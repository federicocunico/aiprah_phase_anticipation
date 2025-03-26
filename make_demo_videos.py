import glob
import os
import shutil
import cv2
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import pickle
from moviepy import ImageSequenceClip, VideoFileClip, clips_array
from moviepy.video.fx import Resize
from tqdm import tqdm

data = pickle.load(open("demo_output_val.pickle", "rb"))
# TARGET = "video02"  # train
TARGET = "video46"  # val
# TARGET = "video50"  # test
video_data = data[TARGET]

x = np.arange(0, len(video_data))

frames = []
pred_anticipated_phase = []
gt_anticipated_phase = []

for i, d in enumerate(video_data):
    pred_anticip = np.asarray(d["pred_anticipated_phase"]).squeeze()
    # gauss noise 7x1
    # gauss_noise = np.random.normal(0, 0.5, 7)
    # pred_anticip = np.asarray(d["time_to_next_phase"]) + gauss_noise
    # pred_anticip[pred_anticip < 0] = abs(pred_anticip[pred_anticip < 0])
    gt_anticip = np.asarray(d["time_to_next_phase_dense"])[:, 0]
    # pred_anticip = gt_anticip + gauss_noise
    # S = 10
    # r = np.roll(pred_anticip, S)
    # r[:S] = 0
    # pred_anticip = pred_anticip + r

    # clamp to 5
    pred_anticip[pred_anticip > 5] = 5
    pred_anticip[pred_anticip < 0] = 0
    gt_anticip[gt_anticip > 5] = 5
    gt_anticip[gt_anticip < 0] = 0

    seq_len = len(d["frames_filepath"])
    # for j in range(seq_len):
    #     tmp = {
    #         "frame": d["frames_filepath"][j],
    #         "gt_current_phase": d["phase_label"],
    #         "pred_anticipated_phase": pred_anticip.tolist(),
    #         "gt_anticipated_phase": gt_anticip.tolist(),
    #     }
    #     # linear_data.append(tmp)
    #     frames.append(tmp["frame"])
    #     # pred_anticipated_phase.append(tmp["pred_current_phase"])
    #     # gt_current_phase.append(tmp["gt_current_phase"])
    #     pred_anticipated_phase.append(tmp["pred_anticipated_phase"])
    #     gt_anticipated_phase.append(tmp["gt_anticipated_phase"])
    j = seq_len - 1
    tmp = {
        "frame": d["frames_filepath"][j],
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


def set_layout(ax, ph_name):
    ax.set_title(ph_name)
    ax.set_xlim(0, len(frames))
    ax.set_ylim(-1, 6)
    # ax.grid()
    ax.grid("off")
    # ax.legend()
    ax.set_xlabel("time (frames)")
    ax.set_ylabel("time (m)")


def get_gt_phase(i):
    return cholec80_phases[np.argmin(gt_anticipated_phase[i])]


# subsample
# frames = frames[::2]

for j in range(len(axes)):
    ph_name = cholec80_phases[j]
    set_layout(axes[j], ph_name=ph_name)

# plt.tight_layout()
template = f"{TARGET}_res/demo_frame_{{0}}.png"
folder = os.path.dirname(template)
if os.path.isdir(folder):
    shutil.rmtree(folder)
os.makedirs(folder, exist_ok=True)

frames_movie = []


def make_seq(i):
    frame_name = frames[i].split("/")[-1]
    frame_name = os.path.join("data/cholec80/downsampled_fps=1", TARGET, frame_name)
    assert os.path.exists(frame_name), f"{frame_name} does not exist"

    # image = cv2.imread(frame_name, cv2.IMREAD_COLOR)
    frames_movie.append(frame_name)

    plt.suptitle(f"Current phase: {get_gt_phase(i)} (red: pred, blue: gt)")
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

        ax.plot(xx, pred_val, label="pred", color="red")
        ax.plot(xx, gt_val, label="gt", color="blue")

        # ax.legend()

        set_layout(ax, ph_name)

    # plt.draw()
    plt.tight_layout()
    # plt.pause(0.1)
    save_fig_fname = template.format(str(i).zfill(5))
    # os.makedirs(os.path.dirname(save_fig_fname), exist_ok=True)
    plt.savefig(save_fig_fname)


FPS = 15
target_video_rgb = f"{TARGET}_FPS={FPS}_surgery_RGB.mp4"
target_video_phases = f"{TARGET}_FPS={FPS}_phase_plot.mp4"
target_video_final = f"{TARGET}_FPS={FPS}_combined.mp4"

if not os.path.isfile(target_video_rgb) or not os.path.isfile(target_video_phases):
    parallel = False

    if not parallel:
        # not parallel
        for i in tqdm(range(len(frames))):
            make_seq(i)
    else:
        # do in parallel with joblib
        Parallel(n_jobs=8)(delayed(make_seq)(i) for i in range(len(frames)))

    saved_frames = glob.glob(f"{TARGET}_res/*.png")
    saved_frames.sort()
    saved_frames = [
        cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in saved_frames
    ]
    # save video
    ImageSequenceClip(frames_movie, fps=FPS).write_videofile(target_video_rgb)
    ImageSequenceClip(saved_frames, fps=FPS).write_videofile(target_video_phases)


def combine_videos_side_by_side(video1, video2, output):
    clip1 = VideoFileClip(video1)
    clip2 = VideoFileClip(video2)

    # Optional: If the clips have different heights, resize one of them:
    if clip1.h < clip2.h:
        clip1 = clip1.with_effects([Resize(height=clip2.h)])
    elif clip2.h < clip1.h:
        clip2 = clip2.with_effects([Resize(height=clip1.h)])

    # Arrange the two clips side by side in one row
    final_clip = clips_array([[clip1, clip2]])
    final_clip.write_videofile(output)


combine_videos_side_by_side(target_video_rgb, target_video_phases, target_video_final)
