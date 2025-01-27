import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from moviepy import ImageSequenceClip
from tqdm import tqdm

data = pickle.load(open("demo_output.pickle", "rb"))
TARGET = "video02"
video_data = data[TARGET]
subsampling = 30
video_data = video_data[::subsampling]  # every 30th frame, sequence length is 30

x = np.arange(0, len(video_data) * subsampling)

frames = []
pred_anticipated_phase = []
gt_anticipated_phase = []

for i, d in enumerate(video_data):
    for j in range(subsampling):
        # pred_anticip = d["pred_anticipated_phase"][j]
        # gauss noise 7x1
        gauss_noise = np.random.normal(0, 0.5, 7)
        # pred_anticip = np.asarray(d["time_to_next_phase"]) + gauss_noise
        # pred_anticip[pred_anticip < 0] = abs(pred_anticip[pred_anticip < 0])
        gt_anticip = np.asarray(d["time_to_next_phase_dense"])[:, 0]
        pred_anticip = gt_anticip + gauss_noise
        # S = 10
        # r = np.roll(pred_anticip, S)
        # r[:S] = 0
        # pred_anticip = pred_anticip + r

        # clamp to 5
        pred_anticip[pred_anticip > 5] = 5
        pred_anticip[pred_anticip < 0] = 0
        gt_anticip[gt_anticip > 5] = 5
        gt_anticip[gt_anticip < 0] = 0

        tmp = {
            "frame": d["frames_filepath"][j],
            # "pred_current_phase": pred_anticip,
            "gt_current_phase": d["phase_label"],
            "pred_anticipated_phase": pred_anticip.tolist(),
            "gt_anticipated_phase": gt_anticip.tolist(),
        }
        # linear_data.append(tmp)
        frames.append(tmp["frame"])
        # pred_anticipated_phase.append(tmp["pred_current_phase"])
        # gt_current_phase.append(tmp["gt_current_phase"])
        pred_anticipated_phase.append(tmp["pred_anticipated_phase"])
        gt_anticipated_phase.append(tmp["gt_anticipated_phase"])

pred_anticipated_phase = np.asarray(pred_anticipated_phase)
gt_anticipated_phase = np.asarray(gt_anticipated_phase)

fig = plt.figure(figsize=(7, 10))
axes = fig.subplots(7, 1)
# axes = [fig.subplots(1, 1)]

cholec80_phases = [
    "Preparation",
    "Calot Triangle Dissection",
    "Clipping Cutting",
    "Gallbladder Dissection",
    "Gallbladder Packaging",
    "Cleaning Coagulation",
    "Gallbladder Retraction",
]


def set_layout(ax):
    ax.set_title(ph_name)
    ax.set_xlim(0, len(frames))
    ax.set_ylim(-1, 6)
    # ax.grid()
    ax.grid("off")
    # ax.legend()


def get_gt_phase(i):
    return cholec80_phases[np.argmin(gt_anticipated_phase[i])]


# subsample
# frames = frames[::2]

for j in range(len(axes)):
    ph_name = cholec80_phases[j]
    set_layout(axes[j])

# plt.tight_layout()

frames_movie = []
for i in tqdm(range(len(frames))):

    # frame_name = frames[i].split("/")[-1]
    # frame_name = os.path.join(TARGET, frame_name)
    # assert os.path.exists(frame_name), f"{frame_name} does not exist"

    # image = cv2.imread(frame_name, cv2.IMREAD_COLOR)
    # frames_movie.append(image)

    plt.suptitle(f"Current phase: {get_gt_phase(i)}")
    for j, ph_name in enumerate(cholec80_phases):
        ax = axes[j]
        ax.cla()

        # pred_val = pred_anticipated_phase[i][j]
        # gt_val = gt_anticipated_phase[i][j]
        # ax.plot(x[i], pred_val, label="pred", color="blue")
        # ax.plot(x[i], gt_val, label="gt", color="red")
        # axes[j].set_yticks(range(5))
        # axes[j].set_yticklabels(cholec80_phases)

        pred_val = pred_anticipated_phase[: i + 1][:, j]
        gt_val = gt_anticipated_phase[: i + 1][:, j]
        xx = x[: i + 1]

        assert (
            len(xx) == len(pred_val) == len(gt_val)
        ), f"len(xx)={len(xx)}, len(pred_val)={len(pred_val)}, len(gt_val)={len(gt_val)}"

        ax.plot(xx, gt_val, label="gt", color="red")
        # ax.plot(xx, pred_val, label="pred", color="blue")

        set_layout(ax)

    # plt.draw()
    plt.tight_layout()
    # plt.pause(0.1)
    save_fig_fname = f"{TARGET}_res/demo_frame_{str(i).zfill(5)}.png"
    os.makedirs(os.path.dirname(save_fig_fname), exist_ok=True)
    plt.savefig(save_fig_fname )

saved_frames = glob.glob(f"{TARGET}_res/demo_frame_*.png")
saved_frames.sort()
saved_frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in saved_frames]
frames_movie = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_movie]

# save video

ImageSequenceClip(frames_movie, fps=30).write_videofile("demo_video_surgery.mp4")
ImageSequenceClip(saved_frames, fps=30).write_videofile("demo_video_phase.mp4")
