import math
import os
import glob
import shutil
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Phase definitions for HeiChole
phase_dict_key = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]
phase_dict = {name: idx for idx, name in enumerate(phase_dict_key)}


# Utility: crop black borders from endoscopic frames
def crop(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    thresh = cv2.medianBlur(thresh, 19)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return lambda x: x
    x, y, w, h = cv2.boundingRect(coords)
    return lambda img: img[y : y + h, x : x + w]


# Subsample video to target fps and resize frames, saving as zero-padded JPGs
def subsample_video_OLD(video_path: str, target_fps: int, output_dir: str):
    RESIZE = 250
    os.makedirs(output_dir, exist_ok=True)
    cmd = f'ffmpeg -i "{video_path}" -vf "fps={target_fps}" {output_dir}/%06d.jpg -loglevel error'
    os.system(cmd)

    frames = sorted(glob.glob(os.path.join(output_dir, "*.jpg")))
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    # Determine crop from first frame
    first = cv2.imread(frames[0])
    do_crop = crop(first)

    for frame_path in tqdm(
        frames, desc=f"Crop & resize {os.path.basename(output_dir)}"
    ):
        img = cv2.imread(frame_path)
        img = do_crop(img)
        img = cv2.resize(img, (RESIZE, RESIZE))
        cv2.imwrite(frame_path, img)


# Subsample video to target fps and resize frames
def subsample_video(video_path: str, target_fps: int, output_dir: str):
    RESIZE = 250
    os.makedirs(output_dir, exist_ok=True)

    # Compute expected number of frames
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    expected = int(total_frames / orig_fps * target_fps)

    # Attempt with FFmpeg + vsync 0 to avoid dropped frames
    ffmpeg_cmd = f'ffmpeg -i "{video_path}" -vf "fps={target_fps}" -vsync 0 "{output_dir}/%06d.jpg" -loglevel error'
    # Determine selection interval (every Nth frame)
    # interval = max(1, int(round(orig_fps / target_fps)))
    # expected = math.ceil(total_frames / interval)

    # ffmpeg_cmd = (
    #     f'ffmpeg -i "{video_path}" '
    #     f'-vf "select=not(mod(n\,{interval})),setpts=N/({orig_fps}*TB)" '
    #     f'-vsync vfr "{output_dir}/%06d.jpg" -loglevel error'
    # )
    os.system(ffmpeg_cmd)
    frames = sorted(glob.glob(os.path.join(output_dir, "*.jpg")))

    # If FFmpeg output is incorrect, fallback to OpenCV extraction
    if len(frames) != expected:
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        interval = int(round(orig_fps / target_fps))
        idx = 0
        count = 0
        do_crop = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                if do_crop is None:
                    do_crop = crop(frame)
                img = do_crop(frame)
                img = cv2.resize(img, (RESIZE, RESIZE))
                out_path = os.path.join(output_dir, f"{count:06d}.jpg")
                cv2.imwrite(out_path, img)
                count += 1
            idx += 1
        cap.release()
        frames = sorted(glob.glob(os.path.join(output_dir, "*.jpg")))

    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return frames


# Load phase annotation CSV and subsample to match frames
def load_annotation(
    annotation_path: str, subsample_fps: int, original_fps: int = 25
) -> np.ndarray:
    lines = open(annotation_path, "r").read().strip().splitlines()
    anns = []
    for line in lines[1:]:  # skip header
        if not line:
            continue
        parts = line.split(",")
        frame_idx = int(parts[0])
        phase = phase_dict_key[int(parts[1])] if parts[1].isdigit() else parts[1]
        phase_id = phase_dict.get(phase, -1)
        anns.append([frame_idx, phase_id])

    # Subsample: one annotation per subsample interval
    step = original_fps // subsample_fps
    return np.array(anns[::step], dtype=np.int64)


# Build future classification targets
def create_future_classification_targets(
    all_phases: torch.Tensor, seq_len: int, F: int, sampling: int, offset: int
):
    # all_phases: [NUM_CLASSES, TOTAL_FRAMES]
    NUM_CLASSES, TOTAL = all_phases.shape
    targets = torch.zeros((seq_len, F, NUM_CLASSES), dtype=torch.int64)
    for t in range(seq_len):
        base = offset + t
        for f in range(F):
            idx = base + (f + 1) * sampling
            if idx < TOTAL and all_phases[:, idx].eq(0).any():
                targets[t, f] = all_phases[:, idx].eq(0).long()
    return targets


def apply_minor_correction(frames, anns):
    n_frames = len(frames)
    expected = len(anns)
    if 0 < abs(n_frames - expected) <= 5:  # small tolerance for minor mismatches
        # assume we can crop the annotations/frames to match
        if n_frames > expected:
            frames = frames[:expected]
        else:
            anns = anns[:n_frames]

        # recompute
        n_frames = len(frames)
        expected = len(anns)
        assert n_frames == expected, (
            f"Frame count mismatch after minor correction: {n_frames} vs {expected}"
        )

    return frames, anns


class HeiCholeDataset(Dataset):
    def __init__(self, root_dir: str, seq_len: int = 10, fps: int = 1, mode: str = 'train'):
        super().__init__()
        assert mode in ['train', 'val', 'test'], "mode must be 'train', 'val', or 'test'"
        self.mode = mode
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.fps = fps

        # Paths
        self.videos_dir = os.path.join(root_dir, "Videos", "Full")
        self.ann_dir = os.path.join(root_dir, "Annotations", "Phase")
        self.down_dir = os.path.join(root_dir, f"downsampled_fps={fps}")

        # Transforms
        base_transform = [transforms.Resize((250, 250)), transforms.RandomCrop(224)]
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.RandomRotation(5),
        ]
        norm = [
            transforms.ToTensor(),
            transforms.Normalize([0.418, 0.261, 0.259], [0.219, 0.198, 0.193]),
        ]
        self.train_transform = transforms.Compose(base_transform + aug + norm)
        self.test_transform = transforms.Compose(base_transform + norm)

        # Load annotations and prepare videos
        self.video_annotations = {}
        self.video_frames = {}
        ann_files = sorted(glob.glob(os.path.join(self.ann_dir, "*.csv")))
        with_issues = []
        for ann in ann_files:
            name = os.path.splitext(os.path.basename(ann))[0].split(
                "_Annotation_Phase"
            )[0]
            video_path = os.path.join(self.videos_dir, f"{name}.mp4")
            assert os.path.exists(video_path), f"Missing video {video_path}"

            tmp = cv2.VideoCapture(video_path)
            video_fps = int(tmp.get(cv2.CAP_PROP_FPS))
            tmp.release()

            print(
                f"Processing <ann,video>: {ann.split(os.sep)[-1]}, {video_path.split(os.sep)[-1]} (FPS: {video_fps})"
            )

            # Load and subsample annotations
            anns = load_annotation(ann, self.fps, original_fps=video_fps)
            self.video_annotations[name] = anns

            # Ensure frames exist
            out_dir = os.path.join(self.down_dir, name)
            expected = len(anns)
            frames = sorted(glob.glob(os.path.join(out_dir, "*.jpg")))
            n_frames = len(frames)
            print("Expected frames:", expected, "Found frames:", n_frames)
            frames, anns = apply_minor_correction(
                frames, anns
            )
            n_frames = len(frames)
            expected = len(anns)

            if n_frames != expected:
                print("Trying to resample video:", name)
                if os.path.isdir(out_dir):
                    shutil.rmtree(out_dir)
                subsample_video(video_path, fps, out_dir)
                frames = sorted(glob.glob(os.path.join(out_dir, "*.jpg")))

                frames, anns = apply_minor_correction(
                    frames, anns
                )
                n_frames = len(frames)
                expected = len(anns)
                # assert len(frames) == expected, f"Frame count mismatch for {name}"
                if len(frames) != expected:
                    print(
                        "[WARNING] Frame count mismatch for", name, "after resampling."
                    )
                    with_issues.append(name)
                    continue
            self.video_frames[name] = frames

        if with_issues:
            print("[WARNING] Some videos had frame count issues:", with_issues)
            raise RuntimeError(
                "Some videos had frame count issues. Please check the logs."
            )

        # Precompute phase-transition times
        self.phase_transition = self._compute_phase_transition_time()

        # Define splits: 24 videos -> 14 train, 5 val, 5 test
        ann_paths = sorted(glob.glob(os.path.join(self.ann_dir, '*.csv')))
        video_ids = [os.path.splitext(os.path.basename(p))[0].split("_Annotation_Phase")[0] for p in ann_paths]

        assert len(video_ids) == 24, "Expecting 24 videos for HeiChole"
        train_ids = video_ids[:14]
        val_ids   = video_ids[14:19]
        test_ids  = video_ids[19:24]
        if mode=='train': selected_ids = train_ids
        elif mode=='val': selected_ids = val_ids
        else: selected_ids = test_ids

        # Create sliding windows
        self.windows = []
        for name, frames in tqdm(self.video_frames.items(), desc='Windows'):
            if name not in selected_ids:
                continue
            anns = self.video_annotations[name]
            trans = torch.tensor(self.phase_transition[name])  # [NUM_CLASSES, TOTAL]
            total = len(frames)
            for i in range(total - self.seq_len):
                frame_window = frames[i:i+self.seq_len]
                frame_indexes = torch.tensor([int(os.path.basename(f).split('.')[0]) for f in frame_window])
                labels = anns[i:i+self.seq_len]
                phase_label_dense = torch.tensor(labels[:,1])
                phase_label = phase_label_dense[-1].item()
                time_to_next_phase_dense = trans[:, i:i+self.seq_len]
                time_to_next_phase = time_to_next_phase_dense[:, -1]
                future_targets = create_future_classification_targets(trans, self.seq_len, 10, 60, i)
                self.windows.append({
                    'video_name': name,
                    'frames_filepath': frame_window,
                    'frames_indexes': frame_indexes,
                    'phase_label': phase_label,
                    'phase_label_dense': phase_label_dense,
                    'time_to_next_phase_dense': time_to_next_phase_dense,
                    'time_to_next_phase': time_to_next_phase,
                    'future_targets': future_targets,
                })

    def _compute_phase_transition_time(self):
        results = {}
        for name, anns in tqdm(self.video_annotations.items(), desc="PhaseTime"):
            phases = anns[:, 1]
            total = len(phases)
            D = defaultdict(lambda: [0] * total)
            for i, p in enumerate(phases):
                for c in range(len(phase_dict_key)):
                    if c == p:
                        D[c][i] = 0
                    else:
                        nxt = next(
                            (j - i for j in range(i + 1, total) if phases[j] == c),
                            total,
                        )
                        D[c][i] = nxt
            arr = np.array(list(D.values())) * self.fps
            arr = arr / 60  # minutes
            results[name] = arr.tolist()
        return results

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        imgs = [Image.open(p) for p in w['frames_filepath']]
        x = torch.stack([self.train_transform(img) for img in imgs])
        return x, w


def __test__():
    dataset = HeiCholeDataset(root_dir='data/heichole', seq_len=30, fps=1, mode="train")
    print(f"Dataset size: {len(dataset)}")
    x, meta = dataset[0]
    print(meta)

if __name__ == "__main__":
    __test__()
