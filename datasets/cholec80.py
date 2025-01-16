import glob
from collections import defaultdict
import os
import shutil
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ColorJitter, RandomRotation, Resize
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


def subsample_video(video_path: str, target_fps: int, output_dir: str):
    """
    Subsample video frames at a given target_fps with rescale to 224x224 using ffmpeg. Saved as 1.zfill(6).jpg
    """
    RESIZE_WIDTH = 250
    RESIZE_HEIGHT = 250

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cmd = f'ffmpeg -i {video_path} -vf "fps={target_fps}" {output_dir}/%06d.jpg'
    os.system(cmd)

    frames = glob.glob(os.path.join(output_dir, "*.jpg"))
    frames.sort()

    first_frame = cv2.imread(frames[0])
    lambda_crop = crop(first_frame)

    for i, frame_path in tqdm(enumerate(frames), desc="Cropping & Resizing"):
        frame = cv2.imread(frame_path)
        if frame.shape[0] == frame.shape[1] == RESIZE_WIDTH:
            continue

        # crop black borders
        frame = lambda_crop(frame)
        # resize to resize width x resize height
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
    def __init__(self, root_dir: str, mode: str = "train", seq_len: int = 10, fps: int = 1):
        super(Cholec80Dataset, self).__init__()

        self.root_dir = root_dir
        self.mode = mode
        self.target_fps = fps
        self.seq_len = seq_len

        self.train_transform = transforms.Compose(
            [
                Resize((250, 250)),
                RandomCrop(224),
                RandomHorizontalFlip(),
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837]),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                Resize((250, 250)),
                RandomCrop(224),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837]),
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

        elif self.mode == "test":
            # videos from 60 to 80
            video_idxs = list(range(61, 81))
            self.transform = self.test_transform

        elif self.mode == "val":
            # videos from 45 to 60
            video_idxs = list(range(46, 61))
            self.transform = self.test_transform

        # -----------------------------------
        elif self.mode == "demo_train":
            video_idxs = list(range(1, 3))
            self.transform = self.train_transform
        elif self.mode == "demo_val":
            video_idxs = list(range(3, 5))
            self.transform = self.test_transform
        elif self.mode == "demo_test":
            video_idxs = list(range(5, 7))
            self.transform = self.test_transform
        # -----------------------------------

        else:
            video_idxs = list(range(1, 81))
            self.transform = self.test_transform

        self.video_paths = [f"{self.root_dir}/videos/video{idx:02d}.mp4" for idx in video_idxs]
        self.label_paths = [f"{self.root_dir}/phase_annotations/video{idx:02d}-phase.txt" for idx in video_idxs]
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
            assert os.path.exists(annotation_path), f"Label file {annotation_path} does not exist"
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

            if not os.path.exists(curr_downsampled_dir) or len(frame_names) != expected_num_frames:
                print(f"Video frames not downsampled. Subsampling now... {video_path.split('/')[-1]}")
                if os.path.exists(curr_downsampled_dir):
                    shutil.rmtree(curr_downsampled_dir)
                os.makedirs(curr_downsampled_dir)
                subsample_video(video_path, self.target_fps, curr_downsampled_dir)
                frame_names = glob.glob(os.path.join(curr_downsampled_dir, "*.jpg"))
                frame_names.sort()

            self.video_paths_frames[video_name] = frame_names
            anns = self.video_annotations[video_name]
            if len(anns) != len(frame_names):
                assert len(anns) > len(frame_names), f"Number of annotations is less than number of frames"
                anns = anns[: len(frame_names)]
            self.video_annotations[video_name] = anns

        # create a list of "windows" of frames
        self.F_steps = 10  # number of future steps to predict
        self.F_sampling = 60  # every 60 ticks (1 minute)
        self.windows = []
        for video_name, frame_names in tqdm(self.video_paths_frames.items(), desc="Creating windows"):
            windows = []
            all_ph_trans_time = torch.asarray(self.phase_transition_time[video_name])  # [NUM_CLASSES, NUM_FRAMES]

            for i in range(len(frame_names) - self.seq_len):
                frame_window = frame_names[i : i + self.seq_len]
                all_frame_labels = self.video_annotations[video_name][i : i + self.seq_len]

                _unsubsampled_frame_n, frame_label = all_frame_labels[-1]  # label for the last frame in the window
                frame_indexes = [int(f.split("/")[-1].replace(".jpg", "")) for f in frame_window]
                ph_trans_time_dense = all_ph_trans_time[:, frame_indexes]
                ph_trans_time = ph_trans_time_dense[:, -1]

                targets_future = create_future_classification_targets(
                    all_ph_trans_time, self.seq_len, self.F_steps, self.F_sampling, i
                )  # [T, F, NUM_CLASSES]
                windows.append(
                    {
                        "video_name": video_name,  # str
                        "frames_filepath": frame_window,  # [T] list of str
                        "frames_indexes": torch.asarray(frame_indexes),  # [T] list of int
                        "phase_label": torch.asarray(frame_label),  # int (label for the last frame in the window)
                        "phase_label_dense": torch.asarray([f[1] for f in all_frame_labels]),  # [T] list of int
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
            return torch.load(dst, weights_only=False)
        else:
            phase_results = {}

            for video_name, annotations in tqdm(self.video_annotations.items(), desc="Computing phase annotations"):
                phases = [phase for _, phase in annotations]  # Extract phase annotations
                num_frames = len(phases)
                phase_lists = defaultdict(lambda: [0] * num_frames)  # Initialize phase lists with zeros

                # Iterate through each frame
                for i in range(num_frames):
                    current_phase = phases[i]

                    # Mark 0 for the current phase
                    for phase in range(NUM_PHASES):  # Assuming phases are numbered 1 to 7
                        if phase == current_phase:
                            phase_lists[phase][i] = 0
                        else:
                            # Count how many frames until this phase appears again
                            next_occurrence = next(
                                (j - i for j in range(i + 1, num_frames) if phases[j] == phase), None
                            )
                            if next_occurrence is not None:
                                phase_lists[phase][i] = next_occurrence
                            else:
                                # Phase does not appear again, set to maximum (num_frames)
                                phase_lists[phase][i] = num_frames

                # Convert frame counts to timings
                timings = np.asarray(list(phase_lists.values()))  # Convert phase lists to numpy array
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
                    ax[i, 1].plot(np.clip(timings[i], 0, 5), label=f"{phase_dict_key[i]}")
                    ax[i, 1].set_title(f"{phase_dict_key[i]}")
                    ax[i, 1].set_ylabel("Minutes")
                    ax[i, 1].set_xlabel("Frames")
                    ax[i, 1].legend()
                plt.savefig(f"data/cholec80/phase_transition_time_{video_name}.png")
            fig.close()
        return phase_results

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        metadata = self.windows[idx]
        frames = torch.stack([self.transform(Image.open(f)) for f in metadata["frames_filepath"]])
        return frames, metadata


def create_future_classification_targets(all_ph_trans_time, seq_len: int, F: int, sampling: int, i: int):
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
                swag_targets[t, f - 1, :] = (all_ph_trans_time[:, future_index] == 0).int()

    return swag_targets


def __test__():
    data_dir = "./data/cholec80"
    dataset = Cholec80Dataset(data_dir, mode="val", seq_len=10, fps=1)

    print(f"Number of videos: {len(dataset)}")
    print(dataset[0])

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for i, batch in enumerate(dataloader):
        frames, metadata = batch
        print(f"Batch {i}")
        print(frames.shape)
        print(metadata["phase_label"])
        break


if __name__ == "__main__":
    __test__()
