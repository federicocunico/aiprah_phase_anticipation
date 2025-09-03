#!/usr/bin/env python3
import os
import time
import random

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
import torchvision.transforms.functional as TF
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


def _list_videos_with_fps(root: Path, fps: int):
    items = []
    for p in sorted(root.iterdir(), key=lambda x: _natural_key(x.name)):
        if p.is_file() and VIDEO_RE.fullmatch(p.name):
            videoname = p.stem
            frames_dir = root / f"{videoname}_{fps}fps"
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


def _show_plot(time_horizon: float | None = 1.0, fps: int = 1):
    """
    Produce two figures per video:
      1) Multi-plot (6 stacked) showing predicted vs GT time-to-next-phase per class.
      2) Completion percentage figure (stacked subplots): GT completion, and predicted
         completion (if prediction file exists). Completions are in [0, 1].
    """
    root = "data/peg_and_ring_workflow"
    root_path = Path(root)
    images_save_path = root_path / "_visualizations"
    images_save_path.mkdir(exist_ok=True, parents=True)
    time_unit = "minutes"

    def _compute_completion_from_times(times: np.ndarray) -> np.ndarray:
        """
        Compute per-frame completion in [0,1] using only the per-class times matrix.
        times: [N, NUM_CLASSES] in chosen units (can include sentinels/negatives).
        Follows the same logic as dataset precomputation and enforces 0 at segment starts.
        """
        N, C = times.shape
        # derive phase per frame
        phase_dense = np.empty(N, dtype=np.int64)
        for t in range(N):
            row = times[t]
            zeros = np.where(row == 0.0)[0]
            phase_dense[t] = int(zeros[0]) if len(zeros) else int(np.argmin(row))

        completion = np.zeros(N, dtype=np.float32)
        start = 0
        TOL = 1e-6

        while start < N:
            p = int(phase_dense[start])
            end = start
            while end + 1 < N and int(phase_dense[end + 1]) == p:
                end += 1
            length = end - start + 1

            completion[start] = 0.0  # exact 0 at segment start

            if length == 1:
                start = end + 1
                continue

            if p >= (NUM_CLASSES - 1):
                # last phase: linear ramp inside segment
                for i in range(1, length):
                    completion[start + i] = float(i) / float(length - 1)
            else:
                total = float(times[start, p + 1])
                valid_total = (total > 0)
                for i in range(1, length):
                    remaining = float(times[start + i, p + 1])
                    if (not valid_total) or (remaining < 0):
                        completion[start + i] = float(i) / float(length - 1)
                    else:
                        num = max(0.0, total - remaining)
                        if abs(remaining - total) <= TOL:
                            num = 0.0
                        comp = num / max(total, 1e-9)
                        completion[start + i] = float(np.clip(comp, 0.0, 1.0))

            start = end + 1

        return completion

    clamp_fn = lambda x, min_val: (
        torch.clamp(torch.asarray(x), min=min_val, max=time_horizon).numpy()
        if time_horizon is not None
        else x
    )

    seq_len = 10
    ds = PegAndRing(
        root_dir=root,
        mode="train",
        seq_len=seq_len,
        time_unit=time_unit,
        fps=fps,
        stride=seq_len,
    )

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

        gts = ds.ant_cache[videoname]  # [N_frames, 6]
        MIN_VAL = gts.min()  # 0 or -1
        if gts.shape[0] != N:
            N = min(N, gts.shape[0])
            gts = gts[:N]

        gts_clamped = clamp_fn(gts, MIN_VAL)

        pred_path = frames_dir / f"{videoname}_pred_1fps_{unit_tag}.npy"
        if pred_path.exists():
            arr = np.load(pred_path)
            if arr.shape[0] != N or arr.shape[1] != 6:
                print(
                    f"[WARN] {videoname}: predictions shape {arr.shape} != ({N},6). Clipping."
                )
                arr = arr[:N, :6]
            arr_clamped = clamp_fn(arr, MIN_VAL)
        else:
            print(f"[WARN] {videoname}: prediction file not found: {pred_path}")
            arr_clamped = None

        # ---------------- Figure 1: Per-class times (stacked) ----------------
        fig, axs = plt.subplots(6, 1, sharex=True, figsize=(12, 12))
        y_upper = time_horizon * 1.05 if time_horizon is not None else None

        x = np.arange(N)
        for i in range(6):
            if arr_clamped is not None:
                axs[i].plot(
                    x, arr_clamped[:N, i], linestyle="-", label="Predicted", linewidth=2
                )
            axs[i].plot(
                x, gts_clamped[:N, i], linestyle="--", label="Ground Truth", linewidth=2
            )
            axs[i].set_ylabel(f"Phase {i}")
            axs[i].grid(True)
            if y_upper is not None:
                axs[i].set_ylim(MIN_VAL, y_upper)

        axs[-1].set_xlabel("Frame index")
        fig.suptitle(
            f"Predicted Time to Next Phase — {videoname} ({time_unit}, horizon={time_horizon} fps={fps})"
        )

        handles, labels = axs[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        img_fname = (
            images_save_path
            / f"{videoname}_anticipation_plot_{unit_tag}_h={time_horizon:g}_fps={fps}.png"
        )
        plt.savefig(img_fname, dpi=150)
        plt.close(fig)
        print(f"[OK] Saved plot -> {img_fname}")

        # ---------------- Figure 2: Completion % (stacked) ----------------
        # GT completion from dataset cache (already aligned to frames)
        compl_gt_full = ds.compl_cache[videoname]
        compl_gt = compl_gt_full[:N]

        # Predicted completion (if predictions present)
        compl_pred = None
        if arr_clamped is not None:
            compl_pred = _compute_completion_from_times(arr_clamped[:N])

        # Build stacked figure: GT and (if present) Pred
        n_rows = 2 if compl_pred is not None else 1
        fig2, axs2 = plt.subplots(n_rows, 1, sharex=True, figsize=(12, 4 * n_rows))
        if n_rows == 1:
            axs2 = [axs2]

        axs2[0].plot(x, compl_gt, linestyle="--", linewidth=2, label="GT Completion")
        axs2[0].set_ylabel("Completion")
        axs2[0].grid(True)
        axs2[0].set_ylim(0.0, 1.0)
        axs2[0].set_title(f"{videoname} — Completion (GT)")

        if compl_pred is not None:
            axs2[1].plot(x, compl_pred, linestyle="-", linewidth=2, label="Pred Completion")
            axs2[1].set_ylabel("Completion")
            axs2[1].grid(True)
            axs2[1].set_ylim(0.0, 1.0)
            axs2[1].set_title(f"{videoname} — Completion (Pred)")

        axs2[-1].set_xlabel("Frame index")

        plt.tight_layout()
        img_fname2 = images_save_path / f"{videoname}_completion_plot_{unit_tag}_fps={fps}.png"
        plt.savefig(img_fname2, dpi=150)
        plt.close(fig2)
        print(f"[OK] Saved plot -> {img_fname2}")


class PegAndRing(Dataset):
    """
    Single-class windowed dataloader for Peg & Ring N-FPS frames + anticipation labels.

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
        # previous-memory features (temporal context of length prev_T):
        "last_phase": torch.asarray(int)                     # at T-1 (or default)
        "last_anticipation": torch.FloatTensor [NUM_CLASSES] # at T-1 (or default)
        "prev_phases": torch.LongTensor [prev_T]             # times T-prev_T..T-1 (padded)
        "prev_anticipation": torch.FloatTensor [prev_T, NUM_CLASSES]
        # optional (only present if rgbd_mode=True AND depths exist for the window):
        "frames_depths": [T] list[str]
        "depths": FloatTensor [T, H, W]
        "frames_rgbd": FloatTensor [T, 4, H, W]
        # added:
        "phase_completition": torch.FloatTensor scalar in [0, 1]
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
        transform: Optional[Callable] = None,  # (kept for API, unused for train)
        to_rgb: bool = True,
        dtype: torch.dtype = torch.float32,
        rgbd_mode: bool = False,
        depth_dir_name: Optional[str] = None,
        rgbd_transform: Optional[Callable] = None,
        future_F: int = 1,
        add_depth_to_frames: bool = True,
        prev_T: int = 0,
        augment: bool = False
    ):
        super().__init__()
        assert fps in [1, 5], "Not implemented with FPS not in [1, 5]"

        self.fps = fps
        self.root = Path(root_dir)
        self.mode = mode.lower().strip()
        assert self.mode in ("train", "val", "test")
        self.augment = augment
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

        # previous memory length
        self.prev_T = int(prev_T)
        assert self.prev_T >= 0

        # --------- Transforms (ensure temporal consistency for a window) ---------
        # We will not use per-frame transforms.Compose for training; instead, we
        # generate *one* set of random params per window and apply it to all frames.
        # For eval/val/test we keep deterministic resize+center-crop.
        self.resize_size = (250, 250)
        self.crop_size = (224, 224)
        self.train_color_jitter = dict(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
        self.train_rotation = 5  # degrees
        self.hflip_p = 0.5
        self.use_train_aug = (self.mode == "train") and (self.augment == True)

        val_videos = {"video01", "video13", "video15", "video22", "video27"}
        all_entries = _list_videos_with_fps(self.root, self.fps)
        if not all_entries:
            raise RuntimeError(
                f"No (videoXX.mp4, videoXX_{self.fps}fps/) pairs with frames found."
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

        start = time.time()
        print(f"Loading {self.mode} split...")

        self.ant_cache: Dict[str, np.ndarray] = {}
        self.compl_cache: Dict[str, np.ndarray] = {}  # per-frame completion in [0,1]
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

            cache_path = (
                frames_dir / f"{videoname}_anticipation_{self.fps}fps_{unit_tag}.npy"
            )
            if cache_path.exists() and False:
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

            # ---------- NEW: pre-compute per-frame completion in [0,1] ----------
            compl = self._precompute_phase_completion_per_frame(ant_unit)
            self.compl_cache[videoname] = compl.astype(np.float32)  # shape [n1]

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
            for start_idx in range(0, max(0, N - self.seq_len + 1), self.stride):
                end_idx = start_idx + self.seq_len
                idxs = items[start_idx:end_idx]
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

                ant_at_fps = self.ant_cache[videoname]        # [N_frames, C]
                gts_window = ant_at_fps[frame_idxs]           # [T, C]

                # determine dense phases from GT anticipation
                phase_dense = []
                for row in gts_window:
                    z = np.where(row == 0.0)[0]
                    phase_dense.append(int(z[0]) if len(z) else int(np.argmin(row)))
                phase_last = int(phase_dense[-1])

                # dt between consecutive sampled frames (in chosen units)
                if self.time_unit == "seconds":
                    dt = 1.0 / float(self.fps)
                else:  # minutes
                    dt = 1.0 / (float(self.fps) * 60.0)

                # last_phase / last_anticipation (at T-1)
                if self.seq_len >= 2:
                    last_phase = int(phase_dense[-2])
                    last_anticipation = gts_window[-2].astype(np.float32)
                else:
                    base = gts_window[0].copy()
                    adj = base + dt
                    adj[0] = 0.0
                    last_phase = 0
                    last_anticipation = adj.astype(np.float32)

                # prev sequences: T-prev_T .. T-1
                prev_len = self.prev_T
                prev_phases_list: List[int] = []
                prev_ants_list: List[np.ndarray] = []
                if prev_len > 0:
                    take_start = max(0, self.seq_len - prev_len)
                    take_end = self.seq_len
                    hist_phases = phase_dense[take_start:take_end]
                    hist_ants = gts_window[take_start:take_end]

                    missing = prev_len - len(hist_phases)
                    if missing > 0:
                        base = gts_window[0].copy()
                        for s in range(missing, 0, -1):
                            vec = (base + s * dt).astype(np.float32)
                            vec[0] = 0.0
                            prev_ants_list.append(vec)
                            prev_phases_list.append(0)

                    for p, a in zip(hist_phases, hist_ants):
                        prev_phases_list.append(int(p))
                        prev_ants_list.append(a.astype(np.float32))

                    prev_anticipation = np.stack(prev_ants_list, axis=0).astype(
                        np.float32
                    )
                    prev_phases = np.asarray(prev_phases_list, dtype=np.int64)
                else:
                    prev_anticipation = np.zeros((0, NUM_CLASSES), dtype=np.float32)
                    prev_phases = np.zeros((0,), dtype=np.int64)

                # Use precomputed per-frame completion (for the *last* frame in the window)
                completion_arr = self.compl_cache[videoname]  # [N_frames]
                completion_last = float(completion_arr[frame_idxs[-1]])

                window.update(
                    {
                        "video_name": videoname,
                        "fps": self.fps,
                        "frames_filepath": frame_paths,
                        "frames_indexes": frame_idxs,
                        "phase_label": phase_last,
                        "phase_label_dense": phase_dense,
                        "time_to_next_phase_dense": gts_window.T,
                        "time_to_next_phase": gts_window[-1],
                        "future_targets": torch.zeros(
                            self.seq_len, 1, NUM_CLASSES, dtype=torch.long
                        ),
                        "last_phase": last_phase,
                        "last_anticipation": last_anticipation,
                        "prev_phases": prev_phases,
                        "prev_anticipation": prev_anticipation,
                        "phase_completition": completion_last,  # scalar float in [0,1]
                    }
                )
                self.windows.append(window)

        self.index = [
            (w["video_name"], Path(self.root / f"{w['video_name']}_1fps"), Path(fp), li)
            for w in self.windows
            for fp, li in zip(w["frames_filepath"], w["frames_indexes"])
        ]

        # --- mapping windows -> by video for samplers ---
        self.video_names: List[str] = [vn for vn, *_ in entries]
        self.windows_by_video: Dict[str, List[int]] = {}
        for wi, w in enumerate(self.windows):
            self.windows_by_video.setdefault(w["video_name"], []).append(wi)

        for vn in self.windows_by_video:
            self.windows_by_video[vn].sort()

        print(f"Loading done in {1/(time.time() - start):.2f} seconds")

    # --------- NEW: precompute completion per frame (using ant_unit only) ----------
    def _precompute_phase_completion_per_frame(self, ant_unit: np.ndarray) -> np.ndarray:
        """
        Given ant_unit [N_frames, NUM_CLASSES], compute per-frame completion in [0,1]
        towards the next phase. Enforces completion[start_of_segment] == 0 exactly.
        """
        N, C = ant_unit.shape
        phase_dense = np.empty(N, dtype=np.int64)
        for t in range(N):
            row = ant_unit[t]
            zeros = np.where(row == 0.0)[0]
            phase_dense[t] = int(zeros[0]) if len(zeros) else int(np.argmin(row))

        completion = np.zeros(N, dtype=np.float32)
        start = 0
        TOL = 1e-6  # tolerance for float jitter

        while start < N:
            p = int(phase_dense[start])
            end = start
            while end + 1 < N and int(phase_dense[end + 1]) == p:
                end += 1
            length = end - start + 1

            # default: 0 at the first frame of the segment
            completion[start] = 0.0

            # If there's only one frame in this segment, keep it at 0 (even for last phase)
            if length == 1:
                start = end + 1
                continue

            if p >= (NUM_CLASSES - 1):
                # Last phase: no p+1 target. Use linear ramp within the segment but start at 0.
                for i in range(1, length):
                    completion[start + i] = float(i) / float(length - 1)
            else:
                total = float(ant_unit[start, p + 1])
                valid_total = (total > 0)
                for i in range(1, length):
                    remaining = float(ant_unit[start + i, p + 1])
                    if (not valid_total) or (remaining < 0):
                        # fallback to linear ramp
                        completion[start + i] = float(i) / float(length - 1)
                    else:
                        # snap tiny jitter so the first step stays near 0
                        num = max(0.0, total - remaining)
                        if abs(remaining - total) <= TOL:
                            num = 0.0
                        comp = num / max(total, 1e-9)
                        completion[start + i] = float(np.clip(comp, 0.0, 1.0))

            start = end + 1

        return completion

    def _map_sample_index_to_source_frame(
        self,
        i_sample: int,
        sample_fps: float,
        src_fps: float,
        total_frames: int,
    ) -> int:
        if total_frames <= 0:
            return 0
        t = i_sample / max(sample_fps, 1e-9)
        src = int(round(t * src_fps))
        return min(max(src, 0), total_frames - 1)

    def _build_and_cache_unit_matrix(
        self,
        ant_raw: np.ndarray,
        cache_path: Path,
        src_fps: float,  # real/video fps from cv2
        total_frames: int,
        n_sampled: int,  # number of frames extracted at self.fps for this video
    ) -> np.ndarray:
        L = total_frames - 1
        rows = []

        for i in range(n_sampled):
            src_idx = self._map_sample_index_to_source_frame(
                i_sample=i,
                sample_fps=float(self.fps),
                src_fps=float(src_fps),
                total_frames=total_frames,
            )

            src_idx = min(max(src_idx, 0), ant_raw.shape[0] - 1)
            raw_row = ant_raw[src_idx].astype(np.int64)

            secs = np.empty(NUM_CLASSES, dtype=np.float64)
            for k, v in enumerate(raw_row):
                if v == 0:
                    secs[k] = 0.0
                elif v == L or v < 0:
                    secs[k] = float(v)
                else:
                    secs[k] = max(0.0, float(v) / max(src_fps, 1e-9))

            vals = secs / 60.0 if self.time_unit == "minutes" else secs
            rows.append(vals)

        ant_unit = np.stack(rows, axis=0).astype(np.float32)
        ant_unit[ant_unit < 0] = -1.0
        np.save(cache_path, ant_unit)
        return ant_unit

    def __len__(self) -> int:
        return len(self.windows)

    # --------- NEW: consistent window-level augmentation ----------
    def _apply_consistent_transform(self, pil_images: List[Image.Image]) -> torch.Tensor:
        """
        Apply the same randomly-sampled augmentation parameters to all frames in a window
        (temporal consistency). Returns a float tensor [T, 3, H, W] normalized.
        """
        # Resize all to a common size first
        imgs = [img.resize(self.resize_size, Image.BILINEAR) for img in pil_images]

        if self.use_train_aug:
            # sample params once
            do_hflip = random.random() < self.hflip_p
            angle = random.uniform(-self.train_rotation, self.train_rotation)

            # Color jitter factors
            cj = self.train_color_jitter
            b = 1.0 + random.uniform(-cj["brightness"], cj["brightness"])
            c = 1.0 + random.uniform(-cj["contrast"], cj["contrast"])
            s = 1.0 + random.uniform(-cj["saturation"], cj["saturation"])
            h = random.uniform(-cj["hue"], cj["hue"])

            # crop params from first frame (on resized image)
            i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
                imgs[0], output_size=self.crop_size
            )

            out = []
            for img in imgs:
                if do_hflip:
                    img = TF.hflip(img)
                img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, expand=False)
                # color jitter sequence: brightness -> contrast -> saturation -> hue
                img = TF.adjust_brightness(img, b)
                img = TF.adjust_contrast(img, c)
                img = TF.adjust_saturation(img, s)
                img = TF.adjust_hue(img, h)
                img = TF.crop(img, top=i, left=j, height=h_crop, width=w_crop)
                tens = TF.to_tensor(img)
                tens = TF.normalize(tens, mean=self.mean.flatten().tolist(), std=self.std.flatten().tolist())
                out.append(tens)
            return torch.stack(out, dim=0)
        else:
            # eval path: deterministic center-crop
            out = []
            for img in imgs:
                img = TF.center_crop(img, self.crop_size)
                tens = TF.to_tensor(img)
                tens = TF.normalize(tens, mean=self.mean.flatten().tolist(), std=self.std.flatten().tolist())
                out.append(tens)
            return torch.stack(out, dim=0)

    def __getitem__(self, idx: int):
        meta = copy.deepcopy(self.windows[idx])

        pil_frames = [
            Image.open(f).convert("RGB") for f in meta["frames_filepath"]
        ]
        frames = self._apply_consistent_transform(pil_frames).contiguous()  # [T, 3, H, W]

        meta["frames_indexes"] = torch.asarray(meta["frames_indexes"], dtype=torch.long)
        meta["phase_label"] = torch.asarray(meta["phase_label"], dtype=torch.float32)
        meta["phase_label_dense"] = torch.asarray(
            meta["phase_label_dense"], dtype=torch.float32
        )
        meta["time_to_next_phase_dense"] = torch.from_numpy(
            meta["time_to_next_phase_dense"]
        )
        meta["time_to_next_phase"] = torch.from_numpy(meta["time_to_next_phase"])

        # new previous-memory tensors
        meta["last_phase"] = torch.asarray(meta["last_phase"], dtype=torch.long)
        meta["last_anticipation"] = torch.from_numpy(
            np.asarray(meta["last_anticipation"], dtype=np.float32)
        )
        meta["prev_phases"] = torch.from_numpy(
            np.asarray(meta["prev_phases"], dtype=np.int64)
        )
        meta["prev_anticipation"] = torch.from_numpy(
            np.asarray(meta["prev_anticipation"], dtype=np.float32)
        )

        # convert precomputed completion scalar to torch tensor in [0,1]
        meta["phase_completition"] = torch.tensor(
            float(meta["phase_completition"]), dtype=torch.float32
        )

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
            # NOTE: if you want consistent augmentation for depths as well,
            # integrate the same params into a depth pipeline here.
            depths = self.rgbd_transform(depths.unsqueeze(1)).squeeze(1)
            meta["depths"] = depths
            if self.add_depth_to_frames:
                frames_rgbd = torch.cat((frames, depths.unsqueeze(1)), dim=1)
                meta["frames_rgbd"] = frames_rgbd

        return frames, meta


# ------------------- Samplers that shuffle videos, not windows -------------------


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
        count = 0
        bs = int(self.batch_size)
        for vn in self.video_list:
            n = len(self.dataset.windows_by_video[vn])
            q, r = divmod(n, bs)
            count += q + (0 if (self.drop_last or r == 0) else 1)
        return count


# -------------------------------------------------------------------------------------

if __name__ == "__main__":

    _show_plot(1, 1)
    _show_plot(1, 5)
    _show_plot(2, 1)
    _show_plot(2, 5)

    ds_train_5fps = PegAndRing(
        root_dir="data/peg_and_ring_workflow",
        mode="train",
        seq_len=16,
        stride=1,
        fps=5,
        time_unit="minutes",
        prev_T=5,
    )
    ds_train = PegAndRing(
        root_dir="data/peg_and_ring_workflow",
        mode="train",
        seq_len=16,
        stride=1,
        fps=1,
        time_unit="minutes",
        prev_T=5,
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
        frames_idxs = meta["frames_indexes"]
        print(
            f"Idx: {i}, batch_len={frames.shape[0]} frames_idxs={frames_idxs} video_name={meta['video_name']}, completion={meta['phase_completition'][:4]}"
        )

    # ----- Validation/Test sets (for counts) -----
    ds_val = PegAndRing(
        root_dir="data/peg_and_ring_workflow",
        mode="val",
        seq_len=16,
        stride=1,
        time_unit="minutes",
        prev_T=5,
    )
    print("val windows:", len(ds_val))

    ds_test = PegAndRing(
        root_dir="data/peg_and_ring_workflow",
        mode="test",
        seq_len=16,
        stride=1,
        time_unit="minutes",
        prev_T=5,
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
        print(f"Idx: {i}, batch_len={frames.shape[0]} video_name={meta['video_name']}")

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