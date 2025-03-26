import itertools
import json
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


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


class PSIAVAHelper:
    def __init__(self, data_dir: str, target_hz: int = 1):
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
            if target_hz > 1:
                images = images[::target_hz]
            # convert to str
            images = [str(f) for f in images]
            self.images_paths[folder_name] = images

        print("Found total of {} iamges".format(sum([len(images) for images in self.images_paths.values()])))
        self.annotations = {}
        all_annotations = self.load_annots(self.annotations_path, target_hz)

    def load_annots(self, annotation_path: Path, target_hz: int = 1):
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
            val_csv = annotation_path / "psi-ava/fold1/annotations/val.csv"

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
            frame_duration_sec = 1 / target_hz

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

        # subsample all (train_phases, val_phases) to 1/target_hz, as well as the (train_transitions, val_transitions)
        for case, phases in train_phases.items():
            train_phases[case] = phases[::target_hz]
        for case, phases in val_phases.items():
            val_phases[case] = phases[::target_hz]
        for case, transitions in train_transitions.items():
            for phase, transition in transitions.items():
                train_transitions[case][phase] = transition[::target_hz]
        for case, transitions in val_transitions.items():
            for phase, transition in transitions.items():
                val_transitions[case][phase] = transition[::target_hz]

        # debug plot
        if True:
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
                    ax[i, 0].set_xlabel("Frames")
                    ax[i, 0].legend()

                    # Plot clamped data (values between 0 and 5 minutes).
                    ax[i, 1].plot(np.clip(timings[i], 0, 5), label=f"{phase_name}")
                    ax[i, 1].set_title(f"{phase_name} (Clamped)")
                    ax[i, 1].set_ylabel("Minutes")
                    ax[i, 1].set_xlabel("Frames")
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
                    transitions[phase][i] = next_occurrence if next_occurrence is not None else (num_frames - i)

        # Convert the frame differences into minutes.
        # Each frame lasts frame_duration_sec seconds; convert seconds to minutes by dividing by 60.
        factor = frame_duration_sec / 60
        for phase in range(num_phases):
            transitions[phase] = [x * factor for x in transitions[phase]]
        return transitions


if __name__ == "__main__":
    data_dir = "data/PSI-AVA"
    dataset = PSIAVAHelper(data_dir)
    print(dataset.phase_dict)
