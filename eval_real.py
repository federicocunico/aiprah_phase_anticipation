import torch
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from collections import deque
from datasets.cholec80 import Cholec80Dataset
from models.temporal_model_v5 import TemporalAnticipationModel
from trainer import PhaseAnticipationTrainer
import re


def sorted_nicely(l):
    """Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


num_classes = 7
time_horizon = 5
T = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = Cholec80Dataset(root_dir="./data/cholec80", mode="demo_test", seq_len=T, fps=1)
model = TemporalAnticipationModel(sequence_length=T, num_classes=num_classes, time_horizon=time_horizon)

chkpt = "checkpoints/stage2_horizon=1/e=epoch=45-l=val_loss_anticipation=0.02361581288278103-val_acc=val_acc=0.8193330764770508horizon=1-stage2_model_best.ckpt"
pl_model = PhaseAnticipationTrainer(model=model, loss_criterion=None)
pl_model.load_state_dict(torch.load(chkpt, map_location="cpu")["state_dict"])

pl_model.model.eval()
pl_model.model.to(DEVICE)

transform = d.transform

data_path = "./data/sim/"
test = "real"

images = glob.glob(f"{data_path}/{test}/*.png")
images = sorted_nicely(images)
print(f"Found {len(images)} images.")

# create a buffer of 30 images using a deque
buffer = deque(maxlen=T)

result = []
for img_path in tqdm(images):
    if len(buffer) == T:
        # If buffer is full, run model and clear buffer
        img_batch = torch.stack(list(buffer), dim=0).unsqueeze(0).to(DEVICE)  # [1, T, C, H, W]
        with torch.no_grad():
            output = pl_model.model(img_batch)
        result.append(output.detach().cpu().numpy().tolist())
        buffer.popleft()
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img)
    buffer.append(img_tensor)


print(f"Processed {len(result)}")
# results
print(result)

# save as json
save_path = f"{data_path}/{test}_results.json"

with open(save_path, "w") as f:
    json.dump(result, f)
print(f"Results saved to {save_path}")


# result is a list of lists, each containing the remaining time to next phase for each class (7).
# plot the results as a 7 plots for each continuous phase numbers

# convert results to array of shape (num_frames, num_classes)
arr = np.array(result).squeeze(axis=1)

# create a 7‐row subplot, one for each class
fig, axs = plt.subplots(7, 1, sharex=True, figsize=(12, 14))
for i in range(7):
    axs[i].plot(arr[:, i], marker="o", linestyle="-")
    axs[i].set_ylabel(f"Class {i}")
    axs[i].grid(True)
    axs[i].set_ylim(0, 5.5)

# common labels and title
axs[-1].set_xlabel("Frame index")
fig.suptitle("Predicted Time to Next Phase per Class")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# save the final report image
report_path = f"{data_path}/{test}_results_plot.png"
plt.savefig(report_path)
print(f"Report saved to {report_path}")

# plt.show()
