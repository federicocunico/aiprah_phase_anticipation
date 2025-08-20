#!/usr/bin/env python3
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from PIL import Image
import torch
from torch.utils.data import Dataset
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
        # fallback: brute count
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
    # 1fps frames are taken at t=i seconds from t=0 → nearest source frame
    src = int(round(i_1fps * fps))
    return min(max(src, 0), total_frames - 1)


def _show_plot(time_horizon: float | None = 1.0):
    """
    Create a 6-row subplot (phases 0..5) comparing predicted vs ground truth anticipation,
    for each video discovered by the PegAndRing dataset.

    Convention for predictions (per video):
        <root>/<videoname>_1fps/<videoname>_pred_1fps_<time>.npy

    where:
      - <time> is "seconds" or "minutes"

    If a prediction file isn't found, this function will:
      - (TODO) Replace the 'dummy' line with your model inference to produce preds (Nx6),
        or drop a file at the path above.
      - For now it will plot only the ground truth (and warn).
    """

    # --------- user-configurable ---------
    root = "data/peg_and_ring_workflow"
    root_path = Path(root)
    time_unit = "minutes"  # "seconds" or "minutes"
    # time_horizon = 1.0  # cap horizon (in the chosen unit)
    clamp_fn = lambda x: torch.clamp(torch.asarray(x), max=time_horizon).numpy() if time_horizon is not None else x
    # -------------------------------------

    # Instantiate dataset (already computes GT anticipation in requested unit)
    ds = PegAndRing(root_dir=root, mode="train", seq_len=10, time_unit=time_unit)

    # Per-video indexing
    by_video = {}
    for videoname, frames_dir, frame_path, local_idx in ds.index:
        if videoname not in by_video:
            by_video[videoname] = {
                "frames_dir": Path(frames_dir),
                "N": 0,
            }
        by_video[videoname]["N"] = max(by_video[videoname]["N"], local_idx + 1)

    # Unit tag like dataset cache (but no horizon)
    unit_tag = f"{time_unit.lower()}"

    for videoname, meta in by_video.items():
        frames_dir: Path = meta["frames_dir"]
        N = meta["N"]

        # Ground-truth anticipation (Nx6 float)
        gts = ds.ant_cache[videoname]  # already (N,6)
        if gts.shape[0] != N:
            N = min(N, gts.shape[0])
            gts = gts[:N]

        # ---- clamp GT to horizon ----
        gts = clamp_fn(gts)

        # Try to load predictions
        pred_path = frames_dir / f"{videoname}_pred_1fps_{unit_tag}.npy"
        if pred_path.exists():
            arr = np.load(pred_path)
            if arr.shape[0] != N or arr.shape[1] != 6:
                print(
                    f"[WARN] {videoname}: predictions shape {arr.shape} != ({N},6). Clipping."
                )
                arr = arr[:N, :6]
            # ---- clamp predictions to horizon ----
            arr = torch.clamp(torch.from_numpy(arr), max=clamp).numpy()
        else:
            print(f"[WARN] {videoname}: prediction file not found: {pred_path}")
            arr = None

        # ---- Plot: 6 rows, one per phase ----
        fig, axs = plt.subplots(6, 1, sharex=True, figsize=(12, 12))

        # y-axis: horizon + small margin
        y_upper = time_horizon * 1.05

        x = np.arange(N)
        for i in range(6):
            if arr is not None:
                axs[i].plot(
                    x,
                    arr[:N, i],
                    linestyle="-",
                    color="blue",
                    label="Predicted",
                    linewidth=2,
                )
            axs[i].plot(
                x,
                gts[:N, i],
                linestyle="--",
                color="red",
                label="Ground Truth",
                linewidth=2,
            )
            axs[i].set_ylabel(f"Phase {i}")
            axs[i].grid(True)
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
        "time_to_next_phase_dense": np.ndarray [NUM_CLASSES, T] float  (in requested unit)
        "time_to_next_phase": np.ndarray [NUM_CLASSES] float            (in requested unit, last frame)
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
        fps: int = 1,  # frames per second for 1fps frames
        *,
        stride: int = 1,
        time_unit: str = "minutes",  # "seconds" or "minutes"
        mean: Tuple[float, float, float] = (0.26728517, 0.32384375, 0.2749076),
        std: Tuple[float, float, float] = (0.15634732, 0.167153, 0.15354523),
        transform: Optional[Callable] = None,  # PIL -> Tensor [3,H,W]
        to_rgb: bool = True,
        dtype: torch.dtype = torch.float32,
        rgbd_mode: bool = False,
        depth_dir_name: Optional[str] = None,  # e.g. "<videoname>_1fps_depths"
        rgbd_transform: Optional[Callable] = None,  # [T,1,H,W] -> [T,H,W]
        future_F: int = 1,
        add_depth_to_frames: bool = True,
    ):
        super().__init__()
        assert fps == 1, "Not implemented with FPS != 1"

        self.root = Path(root_dir)
        self.mode = mode.lower().strip()
        assert self.mode in (
            "train",
            "val",
            "test",
        ), "mode must be 'train' or 'val' or 'test'"
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.time_unit = time_unit.lower().strip()
        assert self.time_unit in (
            "seconds",
            "minutes",
        ), "time_unit must be 'seconds' or 'minutes'"
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

        # --------- default transforms ---------
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

        # --------- discover files and split ---------
        val_videos = {"video01", "video04", "video06", "video08"}
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

        # --------- load anticipation (unit only) for 1fps frames ----------
        self.ant_cache: Dict[str, np.ndarray] = {}
        self.meta: Dict[str, Tuple[int, float]] = {}  # videoname -> (total_frames,fps)
        unit_tag = f"{self.time_unit}"  # no horizon in cache name

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

        # --------- build sliding windows (NO None in metadata) ----------
        self.windows: List[Dict[str, Any]] = []
        by_video: Dict[str, Dict[str, Any]] = {}
        for idx, (videoname, frames_dir, frame_path, local_idx) in enumerate(
            flat_index
        ):
            by_video.setdefault(videoname, {"frames_dir": frames_dir, "items": []})
            by_video[videoname]["items"].append((idx, frame_path, local_idx))

        for videoname, bundle in by_video.items():
            items = sorted(bundle["items"], key=lambda t: t[2])  # by local_idx
            N = len(items)
            for start in range(0, max(0, N - self.seq_len + 1), self.stride):
                end = start + self.seq_len
                idxs = items[start:end]
                if len(idxs) < self.seq_len:
                    continue
                frame_idxs = [li for _, _, li in idxs]
                frame_paths = [str(p) for _, p, _ in idxs]

                # optional depths: include only if *all* depth files exist (avoid partial lists)
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

                ant_1fps = self.ant_cache[videoname]  # [N1,6]
                gts_window = ant_1fps[frame_idxs]  # [T,6] in requested unit

                # dense phase labels
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
                        "time_to_next_phase_dense": gts_window.T,  # [6,T]
                        "time_to_next_phase": gts_window[-1],  # [6]
                        "future_targets": torch.zeros(
                            self.seq_len, 1, NUM_CLASSES, dtype=torch.long
                        ),
                    }
                )
                self.windows.append(window)

        # final flat index (for completeness/debug)
        self.index = [
            (w["video_name"], Path(self.root / f"{w['video_name']}_1fps"), Path(fp), li)
            for w in self.windows
            for fp, li in zip(w["frames_filepath"], w["frames_indexes"])
        ]

    def _build_and_cache_unit_matrix(
        self,
        ant_raw: np.ndarray,  # (N,6) ints: 0 (during), L (after), distance-in-frames (before)
        cache_path: Path,
        fps: float,
        total_frames: int,
        n_1fps: int,
    ) -> np.ndarray:
        """
        Convert raw anticipation to requested unit for each 1fps frame and cache it.
        No clipping; unit is seconds or minutes only.
        """
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
                    # time to the video end (in seconds) from this frame
                    # secs[k] = max(0.0, (L - src_idx) / max(fps, 1e-9))
                    secs[k] = L
                else:
                    # v is a distance-in-frames to the next occurrence
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

        # frames tensor [T,3,H,W]
        frames = torch.stack(
            [
                self.transform(Image.open(f).convert("RGB"))
                for f in meta["frames_filepath"]
            ],
            dim=0,
        )

        # canonical dtypes (no None)
        meta["frames_indexes"] = torch.asarray(meta["frames_indexes"], dtype=torch.long)
        meta["phase_label"] = torch.asarray(meta["phase_label"], dtype=torch.long)
        meta["phase_label_dense"] = torch.asarray(
            meta["phase_label_dense"], dtype=torch.long
        )
        meta["time_to_next_phase_dense"] = np.asarray(
            meta["time_to_next_phase_dense"], dtype=np.float32
        )
        meta["time_to_next_phase"] = np.asarray(
            meta["time_to_next_phase"], dtype=np.float32
        )

        # optional depths
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
            )  # [T,H,W]
            depths = self.rgbd_transform(depths.unsqueeze(1)).squeeze(1)  # [T,H,W]
            meta["depths"] = depths
            if self.add_depth_to_frames:
                frames_rgbd = torch.cat(
                    (frames, depths.unsqueeze(1)), dim=1
                )  # [T,4,H,W]
                meta["frames_rgbd"] = frames_rgbd

        return frames, meta


if __name__ == "__main__":

    _show_plot()

    ds_train = PegAndRing(
        root_dir="data/peg_and_ring_workflow",
        mode="train",
        seq_len=16,
        stride=1,
        time_unit="minutes",  # default (omit to use minutes)
    )
    print("train windows:", len(ds_train))
    for i in range(len(ds_train)):
        data = ds_train[i]

        print(f"  window {i}: {data}")

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
