import glob
import os
import shutil
import cv2
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

        for i in tqdm(range(num_videos), desc="Loading videos"):
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
        self.windows = []
        for video_name, frame_names in self.video_paths_frames.items():
            windows = []
            for i in range(len(frame_names) - self.seq_len):
                frame_window = frame_names[i : i + self.seq_len]
                all_frame_labels = self.video_annotations[video_name][i : i + self.seq_len]

                # future_window = frame_names[i + self.seq_len : i + self.seq_len*2]
                if i + self.seq_len * 2 >= len(frame_names):
                    all_future_labels = all_frame_labels  # final phase is the same as the last frame
                else:
                    all_future_labels = self.video_annotations[video_name][i + self.seq_len : i + self.seq_len * 2]

                _, future_label = all_future_labels[-1]
                _unsubsampled_frame_n, frame_label = all_frame_labels[-1]  # label for the last frame in the window
                frame_indexes = [int(f.split("/")[-1].replace(".jpg", "")) for f in frame_window]
                all_ph_trans_time = self.phase_transition_time[video_name]
                ph_trans_time_dense = [all_ph_trans_time[t] for t in frame_indexes]
                ph_trans_time = ph_trans_time_dense[-1]
                windows.append(
                    {
                        "video_name": video_name,
                        "frames_filepath": frame_window,
                        "frames_indexes": torch.tensor(frame_indexes),
                        "phase_label": torch.tensor(frame_label),
                        "phase_label_dense": torch.tensor([f[1] for f in all_frame_labels]),
                        "future_phase": torch.tensor(future_label),
                        "future_phase_dense": torch.tensor([f[1] for f in all_future_labels]),
                        "time_to_next_phase_dense": torch.tensor(ph_trans_time_dense),
                        "time_to_next_phase": torch.tensor(ph_trans_time),
                    }
                )
            self.windows.extend(windows)

        # random sort; no, let the DataLoader shuffle
        # np.random.shuffle(self.windows)

    def _compute_transition_matrix(self):
        raise RuntimeError("This function is not used anymore")
        """
        Compute the phase transition probability matrix.

        Returns:
            torch.Tensor: A transition matrix of shape [num_classes, num_classes].
        """
        num_classes = 7  # Number of phases
        transition_counts = torch.zeros((num_classes, num_classes))

        # Count transitions across all annotations
        for annotations in self.video_annotations.values():
            for i in range(1, len(annotations)):
                prev_phase = annotations[i - 1][1]
                next_phase = annotations[i][1]
                transition_counts[prev_phase, next_phase] += 1

        # Normalize to create probabilities
        transition_probs = transition_counts / transition_counts.sum(dim=1, keepdim=True)

        # from matplotlib import pyplot as plt
        # fig, ax = plt.subplots()
        # ax.matshow(transition_probs)
        # for (i, j), z in np.ndenumerate(transition_probs):
        #     ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
        # plt.savefig("transition_probs.png")

        return transition_probs

    def _compute_phase_transition_time(self) -> dict[str, dict[int, float]]:
        """
        For a given index i, compute the remaining time to transition to the next phase.
        Time is given by the number of frames until the next phase transition, multiplied by the target_fps.
        Hence, the result is in seconds. Finally, is converted in minutes, for better readability.
        """
        time_to_next_phase: dict[str, list[float]] = {}  # {video_name: {frame_index: time_to_next_phase}}
        # creating signals for each video
        for video_name, annotations in self.video_annotations.items():
            phases = [phase for _, phase in annotations]
            # compute the number of frames until the next phase transition.
            # The phase transition is considered when the phase changes.
            curr_p = phases[0]
            indexes_of_change = []
            for i in range(1, len(phases)):
                if phases[i] != curr_p:
                    curr_p = phases[i]
                    indexes_of_change.append(i)
            indexes_of_change.append(i + 1)

            # calculate the time to the next phase transition:
            initial = 0
            tmp = [0] * len(phases)
            for i in indexes_of_change:
                n_elements = i - initial
                tmp[initial:i] = list(range(n_elements - 1, -1, -1))
                initial = i

            timings = np.asarray(tmp)
            # now timings are one every 25 second (because self.target_fps is 1)
            # we need to convert it to the seconds
            timings = timings * self.target_fps  # now timings are in seconds
            timings = timings / 60  # now timings are in minutes
            timings = timings.tolist()

            time_to_next_phase[video_name] = timings  # {i: val for (i, val) in enumerate(timings)}

        return time_to_next_phase

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        metadata = self.windows[idx]
        frames = torch.stack([self.transform(Image.open(f)) for f in metadata["frames_filepath"]])
        return frames, metadata


def __test__():
    data_dir = "./data/cholec80"
    dataset = Cholec80Dataset(data_dir, mode="all", seq_len=10, fps=1)

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
