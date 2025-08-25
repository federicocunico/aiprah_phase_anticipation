#!/usr/bin/env python3
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable, Iterator, Sequence
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data import BatchSampler
import torchvision.transforms as transforms
from torchvision.transforms import (
    Resize,
    RandomCrop,
    CenterCrop,
    RandomHorizontalFlip,
    ColorJitter,
    RandomRotation,
)

VIDEO_RE = re.compile(r"video\d{2}\.mp4$", re.IGNORECASE)
NUM_CLASSES = 6  # phases 0..5


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _list_videos_with_1fps(root: Path):
    items = []
    for p in sorted(root.iterdir(), key=lambda x: _natural_key(x.name)):
        if p.is_file() and VIDEO_RE.fullmatch(p.name):
            videoname = p.stem
            frames_dir = root / f"{videoname}_1fps"
            if frames_dir.is_dir():
                frames = sorted(
                    [
                        f
                        for f in frames_dir.iterdir()
                        if f.suffix.lower() in (".png", ".jpg", ".jpeg")
                    ],
                    key=lambda x: _natural_key(x.name),
                )
                if frames:
                    items.append((videoname, p, frames_dir, frames))
    return items


def _get_total_frames_and_fps_cv2(video_path: Path) -> Tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()

    if total <= 0:
        cap = cv2.VideoCapture(str(video_path))
        total = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            total += 1
        cap.release()
        if total <= 0:
            raise RuntimeError(f"Could not determine frame count for {video_path}")

    if fps <= 0.0:
        fps = 25.0
    return total, fps


def _map_1fps_index_to_source_frame(
    i_1fps: int, fps: float, total_frames: int, n_1fps: int
) -> int:
    src = int(round(i_1fps * fps))
    return min(max(src, 0), total_frames - 1)


def _show_plot(time_horizon: float | None = 1.0):
    root = "data/peg_and_ring_workflow"
    root_path = Path(root)
    time_unit = "minutes"
    clamp_fn = lambda x: (
        torch.clamp(torch.asarray(x), min=0, max=time_horizon).numpy()
        if time_horizon is not None
        else x
    )

    ds = PegAndRing(root_dir=root, mode="train", seq_len=10, time_unit=time_unit)

    by_video = {}
    for videoname, frames_dir, frame_path, local_idx in ds.index:
        if videoname not in by_video:
            by_video[videoname] = {
                "frames_dir": Path(frames_dir),
                "N": 0,
            }
        by_video[videoname]["N"] = max(by_video[videoname]["N"], local_idx + 1)

    unit_tag = f"{time_unit.lower()}"

    for videoname, meta in by_video.items():
        frames_dir: Path = meta["frames_dir"]
        N = meta["N"]

        gts = ds.ant_cache[videoname]
        if gts.shape[0] != N:
            N = min(N, gts.shape[0])
            gts = gts[:N]

        gts = clamp_fn(gts)

        pred_path = frames_dir / f"{videoname}_pred_1fps_{unit_tag}.npy"
        if pred_path.exists():
            arr = np.load(pred_path)
            if arr.shape[0] != N or arr.shape[1] != 6:
                print(
                    f"[WARN] {videoname}: predictions shape {arr.shape} != ({N},6). Clipping."
                )
                arr = arr[:N, :6]
            arr = clamp_fn(arr)
        else:
            print(f"[WARN] {videoname}: prediction file not found: {pred_path}")
            arr = None

        fig, axs = plt.subplots(6, 1, sharex=True, figsize=(12, 12))
        y_upper = time_horizon * 1.05 if time_horizon is not None else None

        x = np.arange(N)
        for i in range(6):
            if arr is not None:
                axs[i].plot(
                    x, arr[:N, i], linestyle="-", label="Predicted", linewidth=2
                )
            axs[i].plot(
                x, gts[:N, i], linestyle="--", label="Ground Truth", linewidth=2
            )
            axs[i].set_ylabel(f"Phase {i}")
            axs[i].grid(True)
            if y_upper is not None:
                axs[i].set_ylim(0, y_upper)

        axs[-1].set_xlabel("Frame index (1 FPS)")
        fig.suptitle(
            f"Predicted Time to Next Phase — {videoname} ({time_unit}, horizon={time_horizon})"
        )

        handles, labels = axs[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        img_fname = (
            root_path
            / f"{videoname}_anticipation_plot_{unit_tag}_h{time_horizon:g}.png"
        )
        plt.savefig(img_fname, dpi=150)
        plt.close(fig)

        print(f"[OK] Saved plot -> {img_fname}")


class PegAndRing(Dataset):
    """
    Single-class windowed dataloader for Peg & Ring 1-FPS frames + anticipation labels.

    All anticipation values are returned in MINUTES by default (or SECONDS if time_unit="seconds").

    __getitem__ returns:
      frames: FloatTensor [T, 3, H, W]
      metadata: dict (no None values) with keys:
        "video_name": str
        "frames_filepath": [T] list[str]
        "frames_indexes": torch.asarray([int]*T)
        "phase_label": torch.asarray(int)                    # last frame
        "phase_label_dense": torch.asarray([int]*T)
        "time_to_next_phase_dense": np.ndarray [NUM_CLASSES, T] float
        "time_to_next_phase": np.ndarray [NUM_CLASSES] float
        "future_targets": torch.LongTensor [T, F, NUM_CLASSES]
        # optional (only present if rgbd_mode=True AND depths exist for the window):
        "frames_depths": [T] list[str]
        "depths": FloatTensor [T, H, W]
        "frames_rgbd": FloatTensor [T, 4, H, W]
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        seq_len: int = 10,
        fps: int = 1,
        *,
        stride: int = 1,
        time_unit: str = "minutes",
        mean: Tuple[float, float, float] = (0.26728517, 0.32384375, 0.2749076),
        std: Tuple[float, float, float] = (0.15634732, 0.167153, 0.15354523),
        transform: Optional[Callable] = None,
        to_rgb: bool = True,
        dtype: torch.dtype = torch.float32,
        rgbd_mode: bool = False,
        depth_dir_name: Optional[str] = None,
        rgbd_transform: Optional[Callable] = None,
        future_F: int = 1,
        add_depth_to_frames: bool = True,
    ):
        super().__init__()
        assert fps == 1, "Not implemented with FPS != 1"

        self.root = Path(root_dir)
        self.mode = mode.lower().strip()
        assert self.mode in ("train", "val", "test")
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.time_unit = time_unit.lower().strip()
        assert self.time_unit in ("seconds", "minutes")
        self.mean = torch.tensor(mean, dtype=dtype).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=dtype).view(3, 1, 1)
        self.dtype = dtype
        self.to_rgb = to_rgb

        self.rgbd_mode = rgbd_mode
        self.depth_dir_name = depth_dir_name
        self.rgbd_transform = (
            rgbd_transform if rgbd_transform is not None else (lambda x: x)
        )
        self.future_F = int(future_F)
        self.add_depth_to_frames = add_depth_to_frames

        self.train_transform = transforms.Compose(
            [
                Resize((250, 250)),
                RandomCrop(224),
                RandomHorizontalFlip(),
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                Resize((250, 250)),
                CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.transform = (
            self.train_transform
            if (transform is None and self.mode == "train")
            else (self.test_transform if transform is None else transform)
        )

        val_videos = {"video01", "video04"}
        all_entries = _list_videos_with_1fps(self.root)
        if not all_entries:
            raise RuntimeError(
                "No (videoXX.mp4, videoXX_1fps/) pairs with frames found."
            )

        if self.mode == "test":
            print("[W] Test mode not implemented, using 'val'")
            self.mode = "val"

        if self.mode == "train":
            entries = [e for e in all_entries if e[0] not in val_videos]
        else:
            entries = [e for e in all_entries if e[0] in val_videos]

        if not entries:
            raise RuntimeError(f"No entries for split '{self.mode}'")

        self.ant_cache: Dict[str, np.ndarray] = {}
        self.meta: Dict[str, Tuple[int, float]] = {}
        unit_tag = f"{self.time_unit}"

        flat_index: List[Tuple[str, Path, Path, int]] = []
        for videoname, video_path, frames_dir, frame_paths in entries:
            total_frames, fps_real = _get_total_frames_and_fps_cv2(video_path)
            self.meta[videoname] = (total_frames, fps_real)
            n1 = len(frame_paths)

            ant_raw_path = self.root / f"{videoname}_anticipation.npy"
            if not ant_raw_path.exists():
                raise FileNotFoundError(f"Missing anticipation file: {ant_raw_path}")
            ant_raw = np.load(ant_raw_path)
            if ant_raw.ndim != 2 or ant_raw.shape[1] != NUM_CLASSES:
                raise ValueError(
                    f"{ant_raw_path} must be (N,{NUM_CLASSES}), got {ant_raw.shape}"
                )

            cache_path = frames_dir / f"{videoname}_anticipation_1fps_{unit_tag}.npy"
            if cache_path.exists():
                ant_unit = np.load(cache_path)
                if ant_unit.shape != (n1, NUM_CLASSES):
                    ant_unit = self._build_and_cache_unit_matrix(
                        ant_raw, cache_path, fps_real, total_frames, n1
                    )
            else:
                ant_unit = self._build_and_cache_unit_matrix(
                    ant_raw, cache_path, fps_real, total_frames, n1
                )

            self.ant_cache[videoname] = ant_unit.astype(np.float32)

            for local_idx, fp in enumerate(frame_paths):
                flat_index.append((videoname, frames_dir, fp, local_idx))

        self.windows: List[Dict[str, Any]] = []
        by_video: Dict[str, Dict[str, Any]] = {}
        for idx, (videoname, frames_dir, frame_path, local_idx) in enumerate(
            flat_index
        ):
            by_video.setdefault(videoname, {"frames_dir": frames_dir, "items": []})
            by_video[videoname]["items"].append((idx, frame_path, local_idx))

        for videoname, bundle in by_video.items():
            items = sorted(bundle["items"], key=lambda t: t[2])
            N = len(items)
            for start in range(0, max(0, N - self.seq_len + 1), self.stride):
                end = start + self.seq_len
                idxs = items[start:end]
                if len(idxs) < self.seq_len:
                    continue
                frame_idxs = [li for _, _, li in idxs]
                frame_paths = [str(p) for _, p, _ in idxs]

                window: Dict[str, Any] = {}
                if self.rgbd_mode:
                    frames_dir = bundle["frames_dir"]
                    if self.depth_dir_name:
                        depth_dir = frames_dir.parent / self.depth_dir_name.replace(
                            "<videoname>", videoname
                        )
                    else:
                        depth_dir = frames_dir
                    depth_paths_all = [
                        str((depth_dir / f"{Path(p).stem}.npy")) for p in frame_paths
                    ]
                    depth_paths_exist = [
                        dp for dp in depth_paths_all if Path(dp).exists()
                    ]
                    if len(depth_paths_exist) == len(frame_paths):
                        window["frames_depths"] = depth_paths_exist

                ant_1fps = self.ant_cache[videoname]
                gts_window = ant_1fps[frame_idxs]

                phase_dense = []
                for row in gts_window:
                    z = np.where(row == 0.0)[0]
                    phase_dense.append(int(z[0]) if len(z) else int(np.argmin(row)))
                phase_last = int(phase_dense[-1])

                window.update(
                    {
                        "video_name": videoname,
                        "frames_filepath": frame_paths,
                        "frames_indexes": frame_idxs,
                        "phase_label": phase_last,
                        "phase_label_dense": phase_dense,
                        "time_to_next_phase_dense": gts_window.T,
                        "time_to_next_phase": gts_window[-1],
                        "future_targets": torch.zeros(
                            self.seq_len, 1, NUM_CLASSES, dtype=torch.long
                        ),
                    }
                )
                self.windows.append(window)

        self.index = [
            (w["video_name"], Path(self.root / f"{w['video_name']}_1fps"), Path(fp), li)
            for w in self.windows
            for fp, li in zip(w["frames_filepath"], w["frames_indexes"])
        ]

        # --- New: mapping windows -> by video for samplers ---
        self.video_names: List[str] = [vn for vn, *_ in entries]
        self.windows_by_video: Dict[str, List[int]] = {}
        for wi, w in enumerate(self.windows):
            self.windows_by_video.setdefault(w["video_name"], []).append(wi)

        # Ensure per-video window indices are in ascending temporal order
        for vn in self.windows_by_video:
            self.windows_by_video[vn].sort()

    def _build_and_cache_unit_matrix(
        self,
        ant_raw: np.ndarray,
        cache_path: Path,
        fps: float,
        total_frames: int,
        n_1fps: int,
    ) -> np.ndarray:
        L = total_frames - 1
        rows = []
        for i in range(n_1fps):
            src_idx = _map_1fps_index_to_source_frame(i, fps, total_frames, n_1fps)
            raw_row = ant_raw[min(max(src_idx, 0), ant_raw.shape[0] - 1)].astype(
                np.int64
            )

            secs = np.empty(NUM_CLASSES, dtype=np.float64)
            for k, v in enumerate(raw_row):
                if v == 0:
                    secs[k] = 0.0
                elif v == L:
                    secs[k] = L
                else:
                    secs[k] = max(0.0, float(v) / max(fps, 1e-9))

            vals = secs / 60.0 if self.time_unit == "minutes" else secs
            rows.append(vals)

        ant_unit = np.stack(rows, axis=0).astype(np.float32)
        np.save(cache_path, ant_unit)
        return ant_unit

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        meta = copy.deepcopy(self.windows[idx])

        frames = torch.stack(
            [
                self.transform(Image.open(f).convert("RGB"))
                for f in meta["frames_filepath"]
            ],
            dim=0,
        ).contiguous()

        meta["frames_indexes"] = torch.asarray(meta["frames_indexes"], dtype=torch.long)
        meta["phase_label"] = torch.asarray(meta["phase_label"], dtype=torch.float32)
        meta["phase_label_dense"] = torch.asarray(
            meta["phase_label_dense"], dtype=torch.float32
        )
        meta["time_to_next_phase_dense"] = torch.from_numpy(
            meta["time_to_next_phase_dense"]
        )
        meta["time_to_next_phase"] = torch.from_numpy(meta["time_to_next_phase"])

        if (
            self.rgbd_mode
            and ("frames_depths" in meta)
            and len(meta["frames_depths"]) > 0
        ):
            depths = torch.stack(
                [
                    torch.tensor(np.load(fp), dtype=torch.float32)
                    for fp in meta["frames_depths"]
                ],
                dim=0,
            )
            depths = self.rgbd_transform(depths.unsqueeze(1)).squeeze(1)
            meta["depths"] = depths
            if self.add_depth_to_frames:
                frames_rgbd = torch.cat((frames, depths.unsqueeze(1)), dim=1)
                meta["frames_rgbd"] = frames_rgbd

        return frames, meta


# ------------------- New: Samplers that shuffle videos, not windows -------------------


class VideoSequentialSampler(Sampler[int]):
    """
    Yields dataset indices grouped by video. Within a video, indices are yielded in order.
    The *order of videos* is shuffled each epoch if shuffle_videos=True.
    """

    def __init__(
        self,
        dataset: PegAndRing,
        shuffle_videos: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        self.dataset = dataset
        self.shuffle_videos = shuffle_videos
        self.generator = generator

        self.video_list: List[str] = list(dataset.windows_by_video.keys())

    def __iter__(self) -> Iterator[int]:
        g = self.generator if self.generator is not None else torch.Generator()
        order = list(range(len(self.video_list)))
        if self.shuffle_videos:
            perm = torch.randperm(len(order), generator=g).tolist()
            order = [order[i] for i in perm]

        for oi in order:
            vn = self.video_list[oi]
            for idx in self.dataset.windows_by_video[vn]:
                yield idx

    def __len__(self) -> int:
        return len(self.dataset)


class VideoBatchSampler(BatchSampler):
    """
    Batch sampler that:
      - Shuffles the order of videos per epoch (if shuffle_videos=True),
      - Emits batches that NEVER cross video boundaries,
      - Within each video, windows are in order.
    Two modes:
      1) Fixed-size batches: provide batch_size (int). Remainders optionally dropped via drop_last.
      2) One-video-per-batch: set batch_size=None and batch_videos=True (variable batch sizes).
    """

    def __init__(
        self,
        dataset: PegAndRing,
        batch_size: Optional[int] = None,
        drop_last: bool = False,
        shuffle_videos: bool = True,
        generator: Optional[torch.Generator] = None,
        batch_videos: bool = False,
    ):
        if batch_videos and batch_size is not None:
            raise ValueError("Set either batch_videos=True OR batch_size (not both).")
        if not batch_videos and (batch_size is None):
            raise ValueError("Provide batch_size when batch_videos=False.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_videos = shuffle_videos
        self.generator = generator
        self.batch_videos = batch_videos

        self.video_list: List[str] = list(dataset.windows_by_video.keys())

    def __iter__(self) -> Iterator[List[int]]:
        g = self.generator if self.generator is not None else torch.Generator()
        order = list(range(len(self.video_list)))
        if self.shuffle_videos:
            perm = torch.randperm(len(order), generator=g).tolist()
            order = [order[i] for i in perm]

        for oi in order:
            vn = self.video_list[oi]
            inds = self.dataset.windows_by_video[vn]

            if self.batch_videos:
                yield list(inds)
            else:
                # Fixed-size batches within the video
                bs = int(self.batch_size)
                for start in range(0, len(inds), bs):
                    batch = inds[start : start + bs]
                    if len(batch) == bs:
                        yield batch
                    else:
                        if not self.drop_last and len(batch) > 0:
                            yield batch

    def __len__(self) -> int:
        if self.batch_videos:
            return len(self.video_list)
        # Approximate (depends on per-video remainder handling)
        count = 0
        bs = int(self.batch_size)
        for vn in self.video_list:
            n = len(self.dataset.windows_by_video[vn])
            q, r = divmod(n, bs)
            count += q + (0 if (self.drop_last or r == 0) else 1)
        return count


# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    ds_train = PegAndRing(
        root_dir="data/peg_and_ring_workflow",
        mode="train",
        seq_len=16,
        stride=1,
        time_unit="minutes",
    )
    print("train windows:", len(ds_train))

    from torch.utils.data import DataLoader

    """
    # -------- Standard (default) mode: shuffle all windows --------
    dataloader_std = DataLoader(
        ds_train,
        batch_size=32,
        shuffle=True,      # shuffles windows across *all videos*
        num_workers=8,
        pin_memory=False,
    )
    print("Iterating Standard DataLoader (shuffle all windows across all videos)...")
    for i, (frames, meta) in tqdm(enumerate(dataloader_std), total=len(dataloader_std)):
        pass

    # -------- Option A: shuffle videos, fixed-size batches (may cross videos) --------
    gen = torch.Generator()
    gen.manual_seed(42)
    sampler = VideoSequentialSampler(ds_train, shuffle_videos=True, generator=gen)
    dataloader_A = DataLoader(
        ds_train,
        batch_size=32,
        sampler=sampler,    # overrides shuffle
        num_workers=8,
        pin_memory=False,
    )
    print("Iterating Option A (video-shuffled order, fixed-size batches possibly crossing videos)...")
    for i, (frames, meta) in tqdm(enumerate(dataloader_A), total=len(dataloader_A)):
        pass

    # -------- Option B1: one video per batch (variable batch sizes, no mixing) --------
    batch_sampler_video = VideoBatchSampler(
        ds_train,
        batch_size=None,
        batch_videos=True,    # one video = one batch
        shuffle_videos=True,
        generator=gen,
    )
    dataloader_B1 = DataLoader(
        ds_train,
        batch_sampler=batch_sampler_video,
        num_workers=8,
        pin_memory=False,
    )
    print("Iterating Option B1 (one video per batch; batches do not cross videos)...")
    for i, (frames, meta) in tqdm(enumerate(dataloader_B1), total=len(dataloader_B1)):
        pass

    # -------- Option B2: fixed-size batches confined within a video --------
    batch_sampler_fixed = VideoBatchSampler(
        ds_train,
        batch_size=32,
        drop_last=False,
        shuffle_videos=True,
        generator=gen,
        batch_videos=False,   # fixed-size batches
    )
    dataloader_B2 = DataLoader(
        ds_train,
        batch_sampler=batch_sampler_fixed,
        num_workers=8,
        pin_memory=False,
    )
    print("Iterating Option B2 (fixed-size batches that do not cross videos)...")
    for i, (frames, meta) in tqdm(enumerate(dataloader_B2), total=len(dataloader_B2)):
        pass

    """
    gen_train = torch.Generator()
    gen_train.manual_seed(42)
    train_batch_sampler_fixed = VideoBatchSampler(
        ds_train,
        batch_size=32,
        drop_last=False,
        shuffle_videos=True,
        generator=gen_train,
        batch_videos=False,  # fixed-size batches
    )
    dataloader_B2 = DataLoader(
        ds_train,
        batch_sampler=train_batch_sampler_fixed,
        num_workers=8,
        pin_memory=False,
    )

    for i, (frames, meta) in enumerate(dataloader_B2):
        frames_idxs = meta['frames_indexes']
        print( f"Idx: {i}, batch_len={frames.shape[0]} frames_idxs={frames_idxs} video_name={meta['video_name']}")

    # ----- Validation/Test sets (for counts) -----
    ds_val = PegAndRing(
        root_dir="data/peg_and_ring_workflow",
        mode="val",
        seq_len=16,
        stride=1,
        time_unit="minutes",
    )
    print("val windows:", len(ds_val))

    ds_test = PegAndRing(
        root_dir="data/peg_and_ring_workflow",
        mode="test",
        seq_len=16,
        stride=1,
        time_unit="minutes",
    )
    print("test windows:", len(ds_test))

    gen_val = torch.Generator()
    gen_val.manual_seed(42)
    val_batch_sampler_fixed = VideoBatchSampler(
        ds_val,
        batch_size=32,
        drop_last=False,
        shuffle_videos=True,
        generator=gen_val,
        batch_videos=False,  # fixed-size batches
    )
    dataloader_val = DataLoader(
        ds_val,
        batch_sampler=val_batch_sampler_fixed,
        num_workers=8,
        pin_memory=False,
    )

    for i, (frames, meta) in tqdm(enumerate(dataloader_val), total=len(dataloader_val)):
        print( f"Idx: {i}, batch_len={frames.shape[0]} video_name={meta['video_name']}")

    gen_test = torch.Generator()
    gen_test.manual_seed(42)
    test_batch_sampler_fixed = VideoBatchSampler(
        ds_test,
        batch_size=32,
        drop_last=False,
        shuffle_videos=True,
        generator=gen_test,
        batch_videos=False,  # fixed-size batches
    )
    dataloader_test = DataLoader(
        ds_test,
        batch_sampler=test_batch_sampler_fixed,
        num_workers=8,
        pin_memory=False,
    )

    for i, (frames, meta) in tqdm(
        enumerate(dataloader_test), total=len(dataloader_test)
    ):
        pass
