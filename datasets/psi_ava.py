import itertools
import json
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ColorJitter, RandomRotation, Resize, CenterCrop


# def load_phase_dict():
#     ann = "data/PSI-AVA/data_annotations/psi-ava/fold1/coco_anns/train_coco_anns_v3_35s.json"
#     with open(ann, "r") as f:
#         data = json.load(f)
#     phases = data["phase_categories"]
#     phase_dict = {d["id"]: d["name"] for d in phases}
#     # sort by id
#     phase_dict = dict(sorted(phase_dict.items()))
#     return phase_dict
# phase_dict = load_phase_dict()

phase_dict = {
    0: "Idle",  # Idle
    1: "LPIL",  # Left pelvic isolated lymphadenectomy
    2: "RPIL",  # Right pelvic isolated lymphadenectomy
    3: "Retzius_Space",  # Developing the Space of Retzius
    4: "Dorsal_Venous_Complex",  # Ligation of the deep dorsal venous complex
    5: "Id_Bladder_Neck",  # Bladder neck identification and transection
    6: "Seminal_Vesicles",  # Seminal vesicle dissection
    7: "Denonvilliers_Fascia",  # Development of the plane between the prostate and rectum
    8: "Pedicle_Control",  # Prostatic pedicle control
    9: "Severing_Prostate_Urethra",  # Severing of the prostate from the urethra
    10: "Bladder_Neck_Rec",  # Bladder neck reconstruction
}
NUM_PHASES = len(phase_dict.keys())


class PSI_AVA:
    def __init__(self, data_dir: str, seq_len=30, mode: str = "train", time_horizon: int = 5, normalize: bool = True):
        self.time_horizon = time_horizon
        self.seq_len = seq_len
        self.data_dir = Path(data_dir)
        self.phase_dict = phase_dict

        self.annotations_path = self.data_dir / "data_annotations"
        self.images_dir = self.data_dir / "keyframes"

        # get folders in images_path
        self.folders = [f for f in self.images_dir.iterdir() if f.is_dir()]
        self.folders.sort()
        # get all images in each folder
        self.images_paths: dict[str, list[str]] = {}
        for folder in self.folders:
            # get folder name
            folder_name = folder.name
            images = [f for f in folder.iterdir() if f.is_file()]
            images.sort()
            # if target_hz > 1:
            #     images = images[::target_hz]
            # convert to str
            images = [str(f) for f in images]
            self.images_paths[folder_name] = images

        print("Found total of {} iamges".format(sum([len(images) for images in self.images_paths.values()])))
        self.annotations = {}
        train_phases, val_phases, train_transitions, val_transitions = self.load_annots(self.annotations_path)
        self.train_transitions = train_transitions
        self.val_transitions = val_transitions
        self.train_phases = train_phases
        self.val_phases = val_phases

        assert mode in ["train", "val", "all"], "mode must be either train or val"
        self.mode = mode

        # data is already sampled at 1 fps
        # we want to sample at 1 frame every 30 seconds
        self.target_time_interval_seconds = 30

        self.windows = []
        # create window data for each video
        # datastructure
        # {                   {
        #     "video_name": video_name,  # str
        #     "frames_filepath": frame_window,  # [T] list of str
        #     "frames_indexes": torch.asarray(frame_indexes),  # [T] list of int
        #     "phase_label": torch.asarray(frame_label),  # int (label for the last frame in the window)
        #     "phase_label_dense": torch.asarray([f[1] for f in all_frame_labels]),  # [T] list of int
        #     "time_to_next_phase_dense": ph_trans_time_dense,  # [NUM_CLASSES, T] list of float; time to next phase for each frame
        #     "time_to_next_phase": ph_trans_time,  # [NUM_CLASSES] list of float; time to next phase for the last frame
        #     "future_targets": targets_future,  # [T, F, NUM_CLASSES] list of int; targets for classification
        # }
        if mode != "all":
            phase_annots = self.train_phases if mode == "train" else self.val_phases
            transition_annots = self.train_transitions if mode == "train" else self.val_transitions
            self.windows = self.compute_windows(phase_annots, transition_annots)
        elif mode == "all":
            # add val windows to train windows
            win_train = self.compute_windows(self.train_phases, self.train_transitions)
            win_val = self.compute_windows(self.val_phases, self.val_transitions)
            self.windows = win_train + win_val
        else:
            raise ValueError("mode must be either train or val")

        ### TMP ###
        # mean, std = self.compute_mean_std()
        if normalize:
            mean = torch.tensor([0.3884, 0.1847, 0.1569])
            std = torch.tensor([0.2202, 0.2012, 0.1913])
        else:
            mean = torch.tensor([0, 0, 0])
            std = torch.tensor([1, 1, 1])
        ###########

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
        self.transform = self.train_transform if mode == "train" else self.test_transform
        print(f"Loaded {len(self.windows)} windows for {mode} mode")

    def get_frames(self, phase_id):
        if self.mode != "all":
            raise NotImplementedError()
        # FIXME: this is not implemented yet, using train_phases
        phase_annots = self.train_phases
        video_frames = {}
        for video_name, frames in self.images_paths.items():
            if video_name not in phase_annots:
                continue
            ph = torch.tensor(phase_annots[video_name])  # [F]
            F = len(ph)
            frame_names = self.images_paths[video_name][:F]

            # split frames into chunks of consecutive frames with the same phase (phase_id)
            chunks = []
            current_chunk = []
            for i in range(F):
                if ph[i] == phase_id:
                    current_chunk.append(frame_names[i])
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = []
            if current_chunk:
                chunks.append(current_chunk)
            # add chunks to video_frames
            video_frames[video_name] = chunks

        # save each chunk as a separate video
        from moviepy import ImageSequenceClip

        dst_folder = Path("data/PSI-AVA/idle_videos")
        if dst_folder.exists():
            import shutil

            shutil.rmtree(dst_folder)
        dst_folder.mkdir(parents=True, exist_ok=True)

        for video_name, chunks in tqdm(video_frames.items()):
            for i, chunk in enumerate(chunks):
                len_in_s = len(chunk)
                dst = dst_folder / f"{video_name}" / f"{video_name}_length={len_in_s}s_{i}.mp4"
                dst.parent.mkdir(parents=True, exist_ok=True)
                # create a video from the chunk
                clip = ImageSequenceClip(chunk, fps=1)  # data is at 1 hz
                # save the video
                clip.write_videofile(dst, codec="libx264")
                # close the clip
                clip.close()

        # make an istogram of the lengths of the chunks
        lengths = [len(chunk) for video_name, chunks in video_frames.items() for chunk in chunks]
        import matplotlib.pyplot as plt
        
        plt.hist(lengths, bins=100)
        plt.xlabel("Length (frames)")
        plt.ylabel("Count")
        plt.title("Histogram of chunk lengths")
        plt.savefig(dst_folder / "histogram.png")
        plt.close()

    def compute_windows(self, phase_annots, transition_annots):
        """
        After data collection, an anonymization and curation
        process was carried out, removing video segments with non-surgical content and
        potential patient identifiers. As a result, our dataset contains approximately
        20.45 hours of the surgical procedure performed by three expert surgeons. Given
        that phase and step annotations are defined over fixed time intervals, their
        recognition tasks benefit from dense, frame-wise annotations at a rate of
        *** one keyframe per second. ***
        As a result, 73,618 keyframes are annotated with one phase-step
        pair. Due to the fine-grained nature of instruments and atomic actions annotations,
        we sample frames every 35 seconds of video, to obtain 2238 keyframe
        candidates.
        These keyframes are annotated with 5804 instrument instances, with
        corresponding instrument category and atomic action(s).
        """
        windows = []
        video_for_mode = list(transition_annots.keys())
        for video_name in video_for_mode:
            ph = torch.tensor(phase_annots[video_name])  # [F]
            F = len(ph)
            frame_names = self.images_paths[video_name][:F]
            tr = transition_annots[video_name]  # [NUM_CLASSES, F]
            tr_tensor = torch.tensor([tr[i] for i in range(NUM_PHASES)])

            assert len(ph) == F, f"Phase annotations for {video_name} do not match the number of frames"
            assert tr_tensor.shape[1] == F, f"Transition annotations for {video_name} do not match the number of frames"
            assert len(frame_names) == F, f"Frame names for {video_name} do not match the number of frames"

            # Reduce from 1fps to 1 frame each 30 seconds
            ph = ph[:: self.target_time_interval_seconds]
            tr_tensor = tr_tensor[:, :: self.target_time_interval_seconds]
            frame_names = frame_names[:: self.target_time_interval_seconds]
            F = len(ph)

            # create windows of seq_len
            for i in range(F - self.seq_len - 1):
                frame_window = frame_names[i : i + self.seq_len]

                windows.append(
                    {
                        "video_name": video_name,
                        "frames_filepath": frame_window,
                        "frames_indexes": torch.arange(i, i + self.seq_len),  # [T] list of int
                        "phase_label": ph[i + self.seq_len - 1],  # int (label for the last frame in the window)
                        # "phase_label_dense": ph[i : i + self.seq_len],  # [T] list of int
                        "time_to_next_phase_dense": tr_tensor[
                            :, i : i + self.seq_len
                        ],  # [NUM_CLASSES, T] list of float
                        # "time_to_next_phase": tr_tensor[:, i : i + self.seq_len],  # [NUM_CLASSES, T] list of float
                    }
                )
        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        metadata = self.windows[idx]
        frames = torch.stack([self.transform(Image.open(f)) for f in metadata["frames_filepath"]])
        return frames, metadata

    # def compute_mean_std(self):
    #     total_sum = torch.zeros(3)
    #     total_sum_sq = torch.zeros(3)
    #     total_pixels = 0

    #     for folder, images in tqdm(self.images_paths.items(), desc="Computing mean and std"):
    #         for image_path in images:
    #             image = Image.open(image_path).convert("RGB")
    #             image = transforms.ToTensor()(image)  # shape: [C, H, W]
    #             # Sum over height and width for each channel
    #             total_sum += image.sum(dim=[1, 2])
    #             total_sum_sq += (image**2).sum(dim=[1, 2])
    #             total_pixels += image.size(1) * image.size(2)

    #     mean = total_sum / total_pixels
    #     std = torch.sqrt(total_sum_sq / total_pixels - mean**2)
    #     return mean, std
    def compute_mean_std(self):
        from joblib import Parallel, delayed
        from tqdm import tqdm
        from PIL import Image
        import torch
        import torchvision.transforms as transforms

        # Local helper function to process a single image
        def process_image(image_path):
            image = Image.open(image_path).convert("RGB")
            image = transforms.ToTensor()(image)  # shape: [C, H, W]
            sum_per_channel = image.sum(dim=[1, 2])
            sum_sq_per_channel = (image**2).sum(dim=[1, 2])
            num_pixels = image.size(1) * image.size(2)
            return sum_per_channel, sum_sq_per_channel, num_pixels

        # Flatten all image paths into a single list
        all_image_paths = []
        for folder, images in self.images_paths.items():
            all_image_paths.extend(images)

        # Process images in parallel
        results = Parallel(n_jobs=-1)(
            delayed(process_image)(image_path) for image_path in tqdm(all_image_paths, desc="Processing images")
        )

        # Initialize accumulators
        total_sum = torch.zeros(3)
        total_sum_sq = torch.zeros(3)
        total_pixels = 0

        # Aggregate results from each image
        for sum_per_channel, sum_sq_per_channel, num_pixels in results:
            total_sum += sum_per_channel
            total_sum_sq += sum_sq_per_channel
            total_pixels += num_pixels

        # Compute overall mean and standard deviation per channel
        mean = total_sum / total_pixels
        std = torch.sqrt(total_sum_sq / total_pixels - mean**2)
        print("Mean: ", mean)
        print("Std: ", std)
        return mean, std

    def load_annots(self, annotation_path: Path):
        """
        Load annotations from the annotation path and compute both the phase timeline and
        the remaining time until each phase transition (in minutes).

        Args:
            annotation_path (Path): Path to the annotation folder.
            target_hz (int): Target frequency (frames per second) for the images.

        Returns:
            train_phases (dict[str, list[int]]): Phase annotations for each training case.
            val_phases (dict[str, list[int]]): Phase annotations for each validation case.
            train_transitions (dict[str, dict[int, list[float]]]): For each training case, a dict
                mapping each phase (0 to NUM_PHASES-1) to a list (per frame) of remaining times (in minutes)
                until that phase occurs again.
            val_transitions (dict[str, dict[int, list[float]]]): For each validation case, same as above.
        """
        NUM_PHASES = len(phase_dict.keys())
        cached_annots = annotation_path / "cached_psiava_annots.pkl"
        if cached_annots.exists():
            print("Loading cached annotations...")
            train_phases, val_phases, train_transitions, val_transitions = torch.load(cached_annots, weights_only=False)
        else:
            print("Loading annotations from CSV files...")
            # Expected CSV columns
            COLS = [
                "CASE",
                "FRAME",
                "ID_NUMBER",
                "BOX_X1",
                "BOX_Y1",
                "BOX_X2",
                "BOX_Y2",
                "ACTION_1",
                "ACTION_2",
                "ACTION_3",
                "INSTRUMENT",
                "STEP",
                "PHASE",
            ]

            # Load CSV files for train and validation sets.
            train_csv = annotation_path / "psi-ava_extended/fold1/annotations/train.csv"
            val_csv = annotation_path / "psi-ava_extended/fold1/annotations/val.csv"

            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(val_csv)

            # Assign proper column names.
            train_df.columns = COLS
            val_df.columns = COLS

            # Keep only the necessary columns.
            train_df = train_df[["CASE", "FRAME", "PHASE"]]
            val_df = val_df[["CASE", "FRAME", "PHASE"]]

            # Remove duplicate (CASE, FRAME) entries.
            train_df = train_df.drop_duplicates(subset=["CASE", "FRAME"])
            val_df = val_df.drop_duplicates(subset=["CASE", "FRAME"])

            # Sort by CASE and FRAME.
            train_df = train_df.sort_values(by=["CASE", "FRAME"])
            val_df = val_df.sort_values(by=["CASE", "FRAME"])

            # check sequential order of frames
            for case, group in train_df.groupby("CASE"):
                frames = group["FRAME"].tolist()
                if frames != list(range(min(frames), max(frames) + 1)):
                    raise ValueError(
                        f"Frames for case {case} are not sequential: {frames}; are you using the right fold (EXTENDED)?"
                    )
            for case, group in val_df.groupby("CASE"):
                frames = group["FRAME"].tolist()
                if frames != list(range(min(frames), max(frames) + 1)):
                    raise ValueError(
                        f"Frames for case {case} are not sequential: {frames}; are you using the right fold (EXTENDED)?"
                    )

            # Build dictionaries that map each case to an ordered list of phase annotations.
            train_phases: dict[str, list[int]] = {}
            val_phases: dict[str, list[int]] = {}

            for _, row in train_df.iterrows():
                case = row["CASE"]
                phase = row["PHASE"]
                if case not in train_phases:
                    train_phases[case] = []
                train_phases[case].append(phase)

            for _, row in val_df.iterrows():
                case = row["CASE"]
                phase = row["PHASE"]
                if case not in val_phases:
                    val_phases[case] = []
                val_phases[case].append(phase)

            # Frame duration in seconds is determined by the target_hz (frames per second).
            frame_duration_sec = 1 / 35  # The dataset is subsampled at 1 each 35 frames.

            # Compute transitions for each training case.
            train_transitions: dict[str, dict[int, list[float]]] = {}
            for case, phases in train_phases.items():
                train_transitions[case] = self.compute_transitions(phases, NUM_PHASES, frame_duration_sec)

            # Compute transitions for each validation case.
            val_transitions: dict[str, dict[int, list[float]]] = {}
            for case, phases in val_phases.items():
                val_transitions[case] = self.compute_transitions(phases, NUM_PHASES, frame_duration_sec)

            # Save the annotations to a cache file.
            torch.save(
                (train_phases, val_phases, train_transitions, val_transitions),
                cached_annots,
            )

        # ## HELPER FOR DEBUGGING ##
        # timings = {}
        # RECORDINGS_HZ=30
        # for case, phases in train_phases.items():
        #     if case not in timings:
        #         timings[case] = [[] for _ in range(NUM_PHASES)]
        #     for i, phase in enumerate(phases):
        #         timings[case][phase].append(i / RECORDINGS_HZ)
        # for case, phases in val_phases.items():
        #     if case not in timings:
        #         timings[case] = [[] for _ in range(NUM_PHASES)]
        #     for i, phase in enumerate(phases):
        #         timings[case][phase].append(i / RECORDINGS_HZ)
        # ##########################

        # # subsample all (train_phases, val_phases) to 1/target_hz, as well as the (train_transitions, val_transitions)
        # for case, phases in train_phases.items():
        #     train_phases[case] = phases[::target_hz]
        # for case, phases in val_phases.items():
        #     val_phases[case] = phases[::target_hz]
        # for case, transitions in train_transitions.items():
        #     for phase, transition in transitions.items():
        #         train_transitions[case][phase] = transition[::target_hz]
        # for case, transitions in val_transitions.items():
        #     for phase, transition in transitions.items():
        #         val_transitions[case][phase] = transition[::target_hz]

        # debug plot
        if False:
            from matplotlib import pyplot as plt
            import numpy as np

            for video_name, timings in itertools.chain(train_transitions.items(), val_transitions.items()):
                # Create a new figure for each video.
                fig, ax = plt.subplots(NUM_PHASES, 2, figsize=(10, 20))
                fig.suptitle(f"Phase transition time for {video_name}")

                # For each phase, plot the raw transition times (left column)
                # and the clamped values (right column, e.g., values between 0 and 5 minutes).
                for i in range(NUM_PHASES):
                    # Clear axes in case of any previous plots (not strictly necessary when creating new subplots).
                    ax[i, 0].cla()
                    ax[i, 1].cla()

                    phase_name = phase_dict[i].replace("_", " ").title()
                    # Plot raw data.
                    ax[i, 0].plot(timings[i], label=f"{phase_name}")
                    ax[i, 0].set_title(f"{phase_name} (Raw)")
                    ax[i, 0].set_ylabel("Minutes")
                    ax[i, 0].set_xlabel("Frames (1 = 35sec)")
                    ax[i, 0].legend()

                    # Plot clamped data (values between 0 and 5 minutes).
                    ax[i, 1].plot(np.clip(timings[i], 0, 5), label=f"{phase_name}")
                    ax[i, 1].set_title(f"{phase_name} (Clamped)")
                    ax[i, 1].set_ylabel("Minutes")
                    ax[i, 1].set_xlabel("Frames (1 = 35sec)")
                    # Uncomment if you want a legend here:
                    # ax[i, 1].legend()

                plt.tight_layout()
                plt.savefig(f"data/PSI-AVA/phase_transition_time_{video_name}.png")
                plt.close(fig)

        return train_phases, val_phases, train_transitions, val_transitions

    def compute_transitions(
        self, phases_list: list[int], num_phases: int, frame_duration_sec: float
    ) -> dict[int, list[float]]:
        """
        For a given ordered list of phases, compute a dictionary where each key is a phase
        and the corresponding value is a list (one per frame) of the remaining time (in minutes)
        until that phase occurs. If the phase is active at that frame, the remaining time is zero.

        Args:
            phases_list (list[int]): List of phase annotations (one per frame).
            num_phases (int): Total number of phases.
            frame_duration_sec (float): Duration of one frame in seconds.

        Returns:
            dict[int, list[float]]: Mapping from phase to list of remaining times (in minutes).
        """
        num_frames = len(phases_list)
        transitions = {phase: [0.0] * num_frames for phase in range(num_phases)}

        for i in tqdm(range(num_frames), desc="Computing transitions"):
            current_phase = phases_list[i]
            # For each phase, determine how far in the future it occurs.
            for phase in range(num_phases):
                if phase == current_phase:
                    transitions[phase][i] = 0.0
                else:
                    # Look ahead for the next occurrence of this phase.
                    next_occurrence = None
                    for j in range(i + 1, num_frames):
                        if phases_list[j] == phase:
                            next_occurrence = j - i
                            break
                    # If not found, assume the phase won't occur again; use the remaining frames.
                    # transitions[phase][i] = next_occurrence if next_occurrence is not None else (num_frames - i)
                    transitions[phase][i] = next_occurrence if next_occurrence is not None else num_frames

        # Convert the frame differences into minutes.
        # Each frame lasts frame_duration_sec seconds; convert seconds to minutes by dividing by 60.
        factor = frame_duration_sec / 60
        for phase in range(num_phases):
            transitions[phase] = [x * factor for x in transitions[phase]]
        return transitions


def save_idle_videos():
    idle_videos_folder = "data/PSI-AVA/idle_videos"
    idle_videos_folder = Path(idle_videos_folder)
    idle_videos_folder.mkdir(parents=True, exist_ok=True)
    data_dir = "data/PSI-AVA"
    dataset = PSI_AVA(data_dir, mode="all")

    idle_frames = dataset.get_frames(0)


if __name__ == "__main__":
    data_dir = "data/PSI-AVA"
    dataset = PSI_AVA(data_dir, mode="val")

    # save_idle_videos()

    from matplotlib import pyplot as plt

    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    for i in range(5):
        frames, metadata = dataset[i]

        ax1.imshow(frames[0].permute(1, 2, 0))
        ax1.set_title("Frame 0")
        ax2.imshow(frames[-1].permute(1, 2, 0))
        ax2.set_title("Frame -1")
        plt.savefig("tmp.png")

        print(f"Frames shape: {frames.shape}")
        print(f"Metadata: {metadata}")
        print(f"Phase label: {metadata['phase_label']}")
        print(f"Time to next phase dense: {metadata['time_to_next_phase_dense'].shape}")
        # print(f"Time to next phase: {metadata['time_to_next_phase'].shape}")
        # print(f"Future targets: {metadata['future_targets'].shape}")
