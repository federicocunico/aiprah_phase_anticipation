import copy
import glob
from collections import defaultdict
import os
import shutil
import math
from torchvision.transforms import ToTensor
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import (
    RandomCrop,
    RandomHorizontalFlip,
    ColorJitter,
    RandomRotation,
    Resize,
    CenterCrop,
    Normalize,
)
from torch.utils.data import Dataset
from tqdm import tqdm


phase_dict = {}
phase_dict_key = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]
for i in range(len(phase_dict_key)):
    phase_dict[phase_dict_key[i]] = i

import os
import glob
import shutil
import subprocess
from typing import Optional

import cv2
from tqdm import tqdm

# assume you already have a `crop` function that returns a callable
# def crop(img): ...


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _extract_with_opencv(video_path: str, target_fps: int, output_dir: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if not orig_fps or orig_fps <= 0:
        # Fallback if FPS metadata is missing; treat as 30 fps
        orig_fps = 30.0

    # Time-accumulator approach gives accurate sampling even when fps ratios are not integer
    next_sample_t = 0.0
    sample_dt = 1.0 / max(1, target_fps)
    frame_idx = 0
    saved_idx = 0

    pbar = tqdm(desc="Extracting frames (OpenCV)", unit="f")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / orig_fps
        if t + 1e-9 >= next_sample_t:
            saved_idx += 1
            out_path = os.path.join(output_dir, f"{saved_idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            next_sample_t += sample_dt
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()


def subsample_video(video_path: str, target_fps: int, output_dir: str):
    """
    Subsample video frames at a given target_fps. If ffmpeg is available, use it;
    otherwise fall back to OpenCV. Then crop borders with `crop(first_frame)` and
    resize to 250x250. Frames are saved as %06d.jpg.
    """
    RESIZE_WIDTH = 250
    RESIZE_HEIGHT = (
        250  # kept for clarity, we resize to square (RESIZE_WIDTH x RESIZE_WIDTH)
    )

    os.makedirs(output_dir, exist_ok=True)

    if _has_ffmpeg():
        # Use ffmpeg for fast extraction
        # -hide_banner/-loglevel reduce noise; -y overwrites if re-running
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            video_path,
            "-vf",
            f"fps={target_fps}",
            os.path.join(output_dir, "%06d.jpg"),
        ]
        subprocess.run(cmd, check=True)
    else:
        # Fallback: OpenCV-based extraction
        _extract_with_opencv(video_path, target_fps, output_dir)

    # Post-process: crop + resize
    frames = sorted(glob.glob(os.path.join(output_dir, "*.jpg")))
    if not frames:
        raise RuntimeError(
            "No frames were extracted; check the input video and target_fps."
        )

    first_frame = cv2.imread(frames[0])
    if first_frame is None:
        raise RuntimeError(f"Failed to read first extracted frame: {frames[0]}")

    lambda_crop = crop(first_frame)

    for frame_path in tqdm(frames, desc="Cropping & Resizing"):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # Skip if already the target size and square
        if frame.shape[0] == frame.shape[1] == RESIZE_WIDTH:
            continue

        # crop black borders
        frame = lambda_crop(frame)
        # resize to square
        frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_WIDTH))
        # save the frame
        cv2.imwrite(frame_path, frame)


def crop(image: np.ndarray) -> np.ndarray:
    binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)
    binary_image2 = cv2.medianBlur(
        binary_image2, 19
    )  # filter the noise, need to adjust the parameter based on the dataset
    x = binary_image2.shape[0]
    y = binary_image2.shape[1]

    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(10, y - 10):
            if binary_image2.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)

    if not edges_x:
        return image

    left = min(edges_x)  # left border
    right = max(edges_x)  # right
    width = right - left
    bottom = min(edges_y)  # bottom
    top = max(edges_y)  # top
    height = top - bottom

    # pre1_picture = image[left:left + width, bottom:bottom + height]
    lambda_crop = lambda x: x[left : left + width, bottom : bottom + height]

    return lambda_crop


def load_annotation(annotation_path: str, subsample_fps: int) -> np.ndarray:
    with open(annotation_path, "r") as f:
        lines = f.readlines()

    annotations = []
    for i, line in enumerate(lines):
        if i == 0:
            continue
        line = line.strip()
        if not line:
            continue
        line = line.split("\t")
        frame_num = int(line[0])
        phase = phase_dict[line[1]]

        annotations.append([frame_num, phase])

    # subsample the annotations
    t = 25 // subsample_fps
    annotations = annotations[::t]

    # todo: compute the distribution of transitions

    # return np.array(annotations)
    return annotations


class Cholec80Dataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        seq_len: int = 10,
        fps: int = 1,
        rgbd: bool = False,
    ):
        super(Cholec80Dataset, self).__init__()

        self.root_dir = root_dir
        self.mode = mode
        self.target_fps = fps
        self.seq_len = seq_len

        self.rgbd_mode = rgbd

        self.train_transform = transforms.Compose(
            [
                Resize((250, 250)),
                RandomCrop(224),
                RandomHorizontalFlip(),
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.41757566, 0.26098573, 0.25888634],
                    [0.21938758, 0.1983, 0.19342837],
                ),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                Resize((250, 250)),
                CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.41757566, 0.26098573, 0.25888634],
                    [0.21938758, 0.1983, 0.19342837],
                ),
            ]
        )

        self.train_rgbd_transform = transforms.Compose(
            [
                Resize((250, 250)),
                RandomCrop(224),
                RandomHorizontalFlip(),
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                RandomRotation(5),
                # transforms.ToTensor(),
                # Normalize(mean=[0.3978372812271118, 0.2633379399776459, 0.2579571008682251, 178.03602600097656], std=[0.23394420742988586, 0.2093241959810257, 0.20661258697509766, 94.19951629638672]),
                Normalize(mean=[178.03602600097656], std=[94.19951629638672]),
            ]
        )

        self.test_rgbd_transform = transforms.Compose(
            [
                Resize((250, 250)),
                CenterCrop(224),
                # transforms.ToTensor(),
                # Normalize(mean=[0.3978372812271118, 0.2633379399776459, 0.2579571008682251, 178.03602600097656], std=[0.23394420742988586, 0.2093241959810257, 0.20661258697509766, 94.19951629638672]),
                Normalize(mean=[178.03602600097656], std=[94.19951629638672]),
            ]
        )

        # if self.mode == "train":
        #     # videos from 1 to 32
        #     video_idxs = list(range(1, 33))
        #     self.transform = self.train_transform

        # elif self.mode == "test":
        #     # videos from 41 to 80
        #     video_idxs = list(range(41, 81))
        #     self.transform = self.test_transform

        # elif self.mode == "val":
        #     # videos from 32 to 40
        #     video_idxs = list(range(32, 41))
        #     self.transform = self.test_transform
        if self.mode == "train":
            # videos from 1 to 45
            video_idxs = list(range(1, 46))
            self.transform = self.train_transform
            self.rgbd_transform = self.train_rgbd_transform

        elif self.mode == "test":
            # videos from 60 to 80
            video_idxs = list(range(61, 81))
            self.transform = self.test_transform
            self.rgbd_transform = self.test_rgbd_transform

        elif self.mode == "val":
            # videos from 45 to 60
            video_idxs = list(range(46, 61))
            self.transform = self.test_transform
            self.rgbd_transform = self.test_rgbd_transform

        # -----------------------------------
        elif self.mode == "demo_train":
            video_idxs = list(range(1, 3))
            self.transform = self.train_transform
            self.rgbd_transform = self.train_rgbd_transform
        elif self.mode == "demo_val":
            video_idxs = list(range(46, 48))
            self.transform = self.test_transform
            self.rgbd_transform = self.test_rgbd_transform
        elif self.mode == "demo_test":
            video_idxs = list(range(61, 65))
            self.transform = self.test_transform
            self.rgbd_transform = self.test_rgbd_transform
        # -----------------------------------

        else:
            video_idxs = list(range(1, 81))
            self.transform = self.test_transform
            self.rgbd_transform = self.test_rgbd_transform

        self.video_paths = [
            f"{self.root_dir}/videos/video{idx:02d}.mp4" for idx in video_idxs
        ]
        self.label_paths = [
            f"{self.root_dir}/phase_annotations/video{idx:02d}-phase.txt"
            for idx in video_idxs
        ]
        num_videos = len(self.video_paths)
        downsampled_dir = f"{self.root_dir}/downsampled_fps={self.target_fps}"

        self.video_paths_frames = {}
        self.video_annotations = {}

        # Pre-load annotations
        for i in range(num_videos):
            video_name = self.video_paths[i].split(os.sep)[-1].replace(".mp4", "")
            annotation_path = self.label_paths[i]
            anns = load_annotation(annotation_path, self.target_fps)
            self.video_annotations[video_name] = anns

        # Transition matrix initialization
        # self.transition_matrix = self._compute_transition_matrix()
        self.phase_transition_time = self._compute_phase_transition_time()

        for i in range(num_videos):
            video_path = self.video_paths[i]
            annotation_path = self.label_paths[i]
            assert os.path.exists(video_path), f"Video file {video_path} does not exist"
            assert os.path.exists(
                annotation_path
            ), f"Label file {annotation_path} does not exist"
            video_name = video_path.split(os.sep)[-1].replace(".mp4", "")
            # calculate the number of frames in the video
            cap = cv2.VideoCapture(video_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)

            # calculate the expected number of frames after applying the target_fps
            expected_num_frames = int(num_frames / video_fps * self.target_fps)

            # find if already downsampled frames exist
            curr_downsampled_dir = os.path.join(downsampled_dir, video_name)
            frame_names = glob.glob(os.path.join(curr_downsampled_dir, "*.jpg"))
            frame_names.sort()

            if (
                not os.path.exists(curr_downsampled_dir)
                or len(frame_names) != expected_num_frames
            ):
                print(
                    f"Video frames not downsampled. Subsampling now... {video_path.split('/')[-1]}"
                )
                if os.path.exists(curr_downsampled_dir):
                    shutil.rmtree(curr_downsampled_dir)
                os.makedirs(curr_downsampled_dir)
                subsample_video(video_path, self.target_fps, curr_downsampled_dir)
                frame_names = glob.glob(os.path.join(curr_downsampled_dir, "*.jpg"))
                frame_names.sort()

            self.video_paths_frames[video_name] = frame_names
            anns = self.video_annotations[video_name]
            if len(anns) != len(frame_names):
                assert len(anns) > len(
                    frame_names
                ), f"Number of annotations is less than number of frames"
                anns = anns[: len(frame_names)]
            self.video_annotations[video_name] = anns

        # create a list of "windows" of frames
        self.F_steps = 10  # number of future steps to predict
        self.F_sampling = 60  # every 60 ticks (1 minute)
        self.windows = []
        for video_name, frame_names in tqdm(
            self.video_paths_frames.items(), desc="Creating windows"
        ):
            windows = []
            all_ph_trans_time = torch.asarray(
                self.phase_transition_time[video_name]
            )  # [NUM_CLASSES, NUM_FRAMES]

            for i in range(len(frame_names) - self.seq_len):
                frame_window = frame_names[i : i + self.seq_len]
                all_frame_labels = self.video_annotations[video_name][
                    i : i + self.seq_len
                ]

                _unsubsampled_frame_n, frame_label = all_frame_labels[
                    -1
                ]  # label for the last frame in the window
                frame_indexes = [
                    int(f.split("/")[-1].replace(".jpg", "")) for f in frame_window
                ]
                ph_trans_time_dense = all_ph_trans_time[:, frame_indexes]
                ph_trans_time = ph_trans_time_dense[:, -1]

                # t = time.time()
                # targets_future = create_future_classification_targets(
                #     all_ph_trans_time, self.seq_len, self.F_steps, self.F_sampling, i
                # )  # [T, F, NUM_CLASSES]
                targets_future = torch.zeros(
                    (self.seq_len, self.F_steps, 7), dtype=torch.int64
                )
                # print("elapsed", time.time() - t)

                ## Test with depths
                depths_window = copy.deepcopy(frame_window)  # [T] list of str
                for j in range(len(depths_window)):
                    depths_window[j] = (
                        depths_window[j]
                        .replace("downsampled_fps=1", "downsampled_fps=1_DAM2")
                        .replace(".jpg", ".npy")
                    )
                ####

                windows.append(
                    {
                        "video_name": video_name,  # str
                        "frames_filepath": frame_window,  # [T] list of str
                        "frames_depths": depths_window,  # [T] list of str
                        "frames_indexes": torch.asarray(
                            frame_indexes
                        ),  # [T] list of int
                        "phase_label": torch.asarray(
                            frame_label
                        ),  # int (label for the last frame in the window)
                        "phase_label_dense": torch.asarray(
                            [f[1] for f in all_frame_labels]
                        ),  # [T] list of int
                        "time_to_next_phase_dense": ph_trans_time_dense,  # [NUM_CLASSES, T] list of float; time to next phase for each frame
                        "time_to_next_phase": ph_trans_time,  # [NUM_CLASSES] list of float; time to next phase for the last frame
                        "future_targets": targets_future,  # [T, F, NUM_CLASSES] list of int; targets for classification
                    }
                )
            self.windows.extend(windows)

        # random sort; no, let the DataLoader shuffle
        # np.random.shuffle(self.windows)

    def _compute_phase_transition_time(self) -> dict[str, dict[int, float]]:
        """
        For a given index i, compute the remaining time to transition to the next phase.
        Time is given by the number of frames until the next phase transition, multiplied by the target_fps.
        Hence, the result is in seconds. Finally, is converted in minutes, for better readability.
        """

        NUM_PHASES = 7
        dst = os.path.join(self.root_dir, f"phase_transition_{self.mode}.pkl")
        if os.path.isfile(dst):
            phase_results = torch.load(dst, weights_only=False)
        else:
            phase_results = {}

            for video_name, annotations in tqdm(
                self.video_annotations.items(), desc="Computing phase annotations"
            ):
                phases = [
                    phase for _, phase in annotations
                ]  # Extract phase annotations
                num_frames = len(phases)
                phase_lists = defaultdict(
                    lambda: [0] * num_frames
                )  # Initialize phase lists with zeros

                # Iterate through each frame
                for i in range(num_frames):
                    current_phase = phases[i]

                    # Mark 0 for the current phase
                    for phase in range(
                        NUM_PHASES
                    ):  # Assuming phases are numbered 1 to 7
                        if phase == current_phase:
                            phase_lists[phase][i] = 0
                        else:
                            # Count how many frames until this phase appears again
                            next_occurrence = next(
                                (
                                    j - i
                                    for j in range(i + 1, num_frames)
                                    if phases[j] == phase
                                ),
                                None,
                            )
                            if next_occurrence is not None:
                                phase_lists[phase][i] = next_occurrence
                            else:
                                # Phase does not appear again, set to maximum (num_frames)
                                phase_lists[phase][i] = num_frames

                # Convert frame counts to timings
                timings = np.asarray(
                    list(phase_lists.values())
                )  # Convert phase lists to numpy array
                timings = timings * self.target_fps  # Convert frame counts to seconds
                timings = timings / 60  # Convert seconds to minutes
                timings = timings.tolist()

                # Save results for the current video
                phase_results[video_name] = timings

            with open(dst, "wb") as f:
                torch.save(phase_results, f)

        # debug plot
        if False:
            fig, ax = plt.subplots(7, 2, figsize=(10, 20))
            for video_name, timings in phase_results.items():
                fig.suptitle(f"Phase transition time for {video_name}")
                # first 7 rows, first column, un-normalized
                # second 7 rows, second column, clamped
                for i in range(7):
                    ax[i, 0].cla()
                    ax[i, 1].cla()
                    ax[i, 0].plot(timings[i], label=f"{phase_dict_key[i]}")
                    ax[i, 0].set_title(f"{phase_dict_key[i]}")
                    ax[i, 0].set_ylabel("Minutes")
                    ax[i, 0].set_xlabel("Frames")
                    ax[i, 0].legend()
                    ax[i, 1].plot(
                        np.clip(timings[i], 0, 5), label=f"{phase_dict_key[i]}"
                    )
                    ax[i, 1].set_title(f"{phase_dict_key[i]}")
                    ax[i, 1].set_ylabel("Minutes")
                    ax[i, 1].set_xlabel("Frames")
                    # ax[i, 1].legend()
                plt.tight_layout()
                plt.savefig(f"data/cholec80/phase_transition_time_{video_name}.png")
            plt.close()
        return phase_results

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        metadata = copy.deepcopy(self.windows[idx])
        frames = torch.stack(
            [self.transform(Image.open(f)) for f in metadata["frames_filepath"]]
        )

        # load depths if available
        if "frames_depths" in metadata and self.rgbd_mode:
            # depths = torch.stack([torch.tensor(np.load(f)) for f in metadata["frames_depths"]])
            depths = torch.stack(
                [torch.tensor(np.load(f)) for f in metadata["frames_depths"]]
            )
            # transform depths
            depths = self.rgbd_transform(depths.unsqueeze(1)).squeeze(1)
            # resize depths to match frames
            # depths = Resize((224, 224))(depths.unsqueeze(1)).squeeze(1)
            metadata["depths"] = depths
            frames_rgbd = torch.cat(
                (frames, depths.unsqueeze(1)), dim=1
            )  # Concatenate RGB and Depth
            metadata["frames_rgbd"] = frames_rgbd

        return frames, metadata


def create_future_classification_targets(
    all_ph_trans_time, seq_len: int, F: int, sampling: int, i: int
):
    """
    Create SWAG targets for classification from dataset entry.

    Args:
        all_ph_trans_time (torch.Tensor): [NUM_CLASSES, TOTAL_FRAMES] tensor of floats;
                                         time to the next phase for each frame.
        seq_len (int): The length of the current window (T).
        F (int): Number of future steps to predict.
        i (int): The current index in the dataset loop.

    Returns:
        torch.Tensor: [seq_len, F, NUM_CLASSES] tensor containing the SWAG targets.
    """
    NUM_CLASSES, TOTAL_FRAMES = all_ph_trans_time.shape

    # Initialize the target tensor
    swag_targets = torch.zeros((seq_len, F, NUM_CLASSES), dtype=torch.int64)

    # Loop through each frame in the current window
    for t in range(seq_len):
        current_index = i + t  # Current frame index

        for f in range(1, F + 1):  # Future steps (1 to F)
            future_index = current_index + f * sampling  # Adjust by the sampling rate

            if future_index < TOTAL_FRAMES:
                # Check if the class is maintained (time-to-next-phase is 0)
                swag_targets[t, f - 1, :] = (
                    all_ph_trans_time[:, future_index] == 0
                ).int()

    return swag_targets


def __test__():
    data_dir = "./data/cholec80"
    dataset = Cholec80Dataset(data_dir, mode="train", seq_len=10, fps=1)
    dataset = Cholec80Dataset(data_dir, mode="val", seq_len=10, fps=1)

    print(f"Number of videos: {len(dataset)}")
    print(dataset[0])

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # debug plots
    if False:
        phase_results = dataset.phase_transition_time
        dtype = torch.bfloat16
        fig, ax = plt.subplots(7, 2, figsize=(10, 20))
        for video_name, timings in phase_results.items():
            timings = torch.tensor(timings, dtype=dtype)
            timings = timings.tolist()
            fig.suptitle(f"Phase transition time for {video_name}")
            # first 7 rows, first column, un-normalized
            # second 7 rows, second column, clamped
            for i in range(7):
                ax[i, 0].cla()
                ax[i, 1].cla()
                ax[i, 0].plot(timings[i], label=f"{phase_dict_key[i]}")
                ax[i, 0].set_title(f"{phase_dict_key[i]}")
                ax[i, 0].set_ylabel("Minutes")
                ax[i, 0].set_xlabel("Frames")
                ax[i, 0].legend()
                ax[i, 1].plot(np.clip(timings[i], 0, 5), label=f"{phase_dict_key[i]}")
                ax[i, 1].set_title(f"{phase_dict_key[i]}")
                ax[i, 1].set_ylabel("Minutes")
                ax[i, 1].set_xlabel("Frames")
                # ax[i, 1].legend()
            plt.tight_layout()
            plt.savefig(
                f"data/cholec80/phase_transition_time_{video_name}_{str(dtype)}.png"
            )
        plt.close()

    for i, batch in enumerate(dataloader):
        frames, metadata = batch

        time_to_next_phase = metadata["time_to_next_phase"]
        # time_to_next_phase = torch.clamp(time_to_next_phase, 0, 5)

        bfloat16 = torch.tensor(time_to_next_phase, dtype=torch.bfloat16)
        float16 = time_to_next_phase.half()
        float32 = torch.tensor(time_to_next_phase, dtype=torch.float32)

        for n in [bfloat16, float16, float32]:
            diff = torch.abs(time_to_next_phase - n)  # diff in minutes
            diff_seconds = diff * 60
            print(
                f"Type: {n.dtype}, Min (s): {diff_seconds.min()}, Max (s): {diff_seconds.max()}"
            )
            # print("Diff in minutes", diff)
            print("Diff in seconds:", diff_seconds)

        print(f"Batch {i}: {frames.shape}")

    # Cholec80Dataset(data_dir, mode="val", seq_len=10, fps=1)
    # Cholec80Dataset(data_dir, mode="test", seq_len=10, fps=1)
    # Cholec80Dataset(data_dir, mode="train", seq_len=10, fps=1)


def __compute_normalization__():
    data_dir = "./data/cholec80"
    dataset = Cholec80Dataset(data_dir, mode="all", seq_len=10, fps=1)

    # initialize accumulators
    rgb_sum = torch.zeros(3)
    rgb_sq = torch.zeros(3)
    depth_sum = 0.0
    depth_sq = 0.0
    rgbd_sum = torch.zeros(4)
    rgbd_sq = torch.zeros(4)
    cnt_rgb = cnt_depth = cnt_rgbd = 0

    for _, metadata in tqdm(dataset, desc="Computing normalization"):
        # load RGB frames as tensors [T,3,H,W]
        frames_t = torch.stack(
            [ToTensor()(Image.open(f)) for f in metadata["frames_filepath"]]
        )
        # load depth maps [T,H,W]
        depths_t = torch.stack(
            [torch.tensor(np.load(f)) for f in metadata["frames_depths"]]
        )

        # build RGBD [T,4,H,W]
        rgbd_t = torch.cat((frames_t, depths_t.unsqueeze(1)), dim=1)

        # RGB stats
        fr = frames_t.view(frames_t.size(0), 3, -1).permute(1, 0, 2).reshape(3, -1)
        rgb_sum += fr.sum(dim=1)
        rgb_sq += (fr * fr).sum(dim=1)
        cnt_rgb += fr.size(1)

        # Depth stats
        d = depths_t.view(-1)
        depth_sum += d.sum().item()
        depth_sq += (d * d).sum().item()
        cnt_depth += d.numel()

        # RGBD stats
        frd = rgbd_t.view(rgbd_t.size(0), 4, -1).permute(1, 0, 2).reshape(4, -1)
        rgbd_sum += frd.sum(dim=1)
        rgbd_sq += (frd * frd).sum(dim=1)
        cnt_rgbd += frd.size(1)

    # final mean / std
    mean_rgb = rgb_sum / cnt_rgb
    std_rgb = (rgb_sq / cnt_rgb - mean_rgb**2).sqrt()
    mean_depth = depth_sum / cnt_depth
    std_depth = math.sqrt(depth_sq / cnt_depth - mean_depth**2)
    mean_rgbd = rgbd_sum / cnt_rgbd
    std_rgbd = (rgbd_sq / cnt_rgbd - mean_rgbd**2).sqrt()

    print("RGB   mean:", mean_rgb, " std:", std_rgb)
    print("Depth mean:", mean_depth, " std:", std_depth)
    print("RGBD  mean:", mean_rgbd, " std:", std_rgbd)

    # save the normalization values
    torch.save(
        {
            "mean_rgb": mean_rgb,
            "std_rgb": std_rgb,
            "mean_depth": mean_depth,
            "std_depth": std_depth,
            "mean_rgbd": mean_rgbd,
            "std_rgbd": std_rgbd,
        },
        os.path.join(data_dir, "normalization_values.pt"),
    )


if __name__ == "__main__":
    # fname = "data/cholec80/normalization_values.pt"
    # data = torch.load(fname, weights_only=False)
    # __compute_normalization__()
    __test__()
