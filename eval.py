import os
import pickle
import time
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from trainer import PhaseAnticipationTrainer
from models.temporal_model_v5 import TemporalAnticipationModel
from datasets.cholec80 import Cholec80Dataset
import numpy as np


def eval_model():
    torch.set_float32_matmul_precision("high")  # 'high' or 'medium'; for RTX GPUs
    seq_len = 30
    target_fps = 1
    num_classes = 7
    device = torch.device("cuda:0")

    split = "val"
    fname = f"eval_split={split}.pickle"

    model = TemporalAnticipationModel(time_horizon=5, sequence_length=seq_len, num_classes=num_classes)
    # If evaluation output does not exist, run the model on the dataset and save outputs.
    if not os.path.exists(fname):
        dataset = Cholec80Dataset(root_dir="./data/cholec80", mode=split, seq_len=seq_len, fps=target_fps)
        pl_model = PhaseAnticipationTrainer(model=model, loss_criterion=None)

        # load pretrained-checkpoint
        final_model = "checkpoints/e=epoch=23-l=val_loss_anticipation=0.027181584388017654-stage2_model_best.ckpt"
        pl_model.load_state_dict(torch.load(final_model, weights_only=True)["state_dict"])

        # Set appropriate dtype
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        dtype = torch.float32
        model = model.to(device=device, dtype=dtype)

        # Initialize model (dummy forward pass)
        model(torch.randn(1, seq_len, 3, 224, 224).to(device=device, dtype=dtype))

        out = {}
        with torch.no_grad():
            model.eval()
            for i, (frames, metadata) in tqdm(enumerate(dataset), total=len(dataset)):
                frames = frames.to(device=device, dtype=dtype).unsqueeze(0)
                start = time.time()
                model_output = model(frames)
                # Depending on model output format, separate current and anticipated phase predictions
                if isinstance(model_output, tuple):
                    current_phase, anticipated_phase = model_output
                else:
                    anticipated_phase = model_output
                    current_phase = torch.argmin(anticipated_phase, dim=1)

                # For reporting FPS if needed
                fps = 1 / (time.time() - start)
                # Convert metadata tensors to CPU lists
                m = {k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v for k, v in metadata.items()}
                out_data = {
                    "pred_current_phase": current_phase.cpu().numpy().tolist()[0],  # int
                    "pred_anticipated_phase": anticipated_phase.cpu()  #
                    .float()
                    .numpy()
                    .tolist()[0],  # predicted countdown (list of floats, of length=num_classes)
                    "gt_anticipated_phase": torch.clamp(metadata["time_to_next_phase"], 0, model.time_horizon)
                    .cpu()
                    .numpy()
                    .tolist(),  # ground truth (list of floats, of length=num_classes)
                    "gt_current_phase": metadata["phase_label"].cpu().numpy().tolist(),  # int
                    **m,
                }
                video_name = metadata["video_name"]
                if video_name not in out:
                    out[video_name] = []
                out[video_name].append(out_data)

        with open(fname, "wb") as f:
            pickle.dump(out, f)
    else:
        with open(fname, "rb") as f:
            out = pickle.load(f)


    
    
   


    # -------------------------------
    # Evaluation Metrics Computation
    # -------------------------------

    # 1. Phase Classification Metrics
    phase_preds = []
    phase_gts = []
    for video, samples in out.items():
        for sample in samples:
            phase_preds.append(sample["pred_current_phase"])
            # Ensure gt_current_phase is an integer (if it's in a list, take the first element)
            gt_phase = sample["gt_current_phase"]
            if isinstance(gt_phase, list):
                gt_phase = gt_phase[0]
            phase_gts.append(gt_phase)

    accuracy = accuracy_score(phase_gts, phase_preds)
    precision = precision_score(phase_gts, phase_preds, average="macro")
    recall = recall_score(phase_gts, phase_preds, average="macro")
    f1 = f1_score(phase_gts, phase_preds, average="macro")
    conf_mat = confusion_matrix(phase_gts, phase_preds)
    class_report = classification_report(phase_gts, phase_preds, output_dict=True)

    print("Phase Classification Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_mat)
    print("Classification Report:")
    for cls, metrics in class_report.items():
        print(f"{cls}: {metrics}")

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_mat)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()
    plt.xticks(range(num_classes))
    plt.yticks(range(num_classes))
    plt.savefig(f"cholec80_{split}_confusion_matrix.png")
    plt.close()

    # 2. Anticipation Error Metrics (inMAE, oMAE, wMAE, eMAE)
    # Here we assume that the anticipated phase prediction is a countdown time (in minutes)
    # and that both pred and gt are lists (we take the first element if they are lists).
    time_preds = []
    time_gts = []
    for video, samples in out.items():
        for sample in samples:
            gt_time = sample["gt_anticipated_phase"]
            pred_time = sample["pred_anticipated_phase"]
            if isinstance(gt_time, list):
                gt_time = gt_time[0] if len(gt_time) > 0 else None
            if isinstance(pred_time, list):
                pred_time = pred_time[0] if len(pred_time) > 0 else None
            if gt_time is not None and pred_time is not None:
                time_gts.append(gt_time)
                time_preds.append(pred_time)

    time_preds = np.array(time_preds)
    time_gts = np.array(time_gts)
    abs_errors = np.abs(time_preds - time_gts)
    h = model.time_horizon  # The time horizon used in the model, e.g., 5 minutes

    # inMAE: errors for samples where the ground truth event is within the horizon (< h)
    in_indices = time_gts < h
    in_errors = abs_errors[in_indices] if np.any(in_indices) else np.array([])
    inMAE = np.mean(in_errors) if in_errors.size > 0 else 0

    # oMAE: errors for samples where the ground truth event equals the horizon (i.e., out-of-horizon events)
    out_indices = time_gts == h
    # For out-of-horizon events, the expected prediction is h (i.e., the model should predict h)
    out_errors = np.abs(time_preds[out_indices] - h) if np.any(out_indices) else np.array([])
    oMAE = np.mean(out_errors) if out_errors.size > 0 else 0

    wMAE = (inMAE + oMAE) / 2  # as per the paper's definition

    # eMAE: errors for samples where the ground truth event is within 0.1h (very-short-term prediction)

    # eMAE stands for “very-short-term Mean Absolute Error.” It specifically measures the average absolute error for predictions of events that are expected to occur imminently—typically within a very short fraction (e.g., 10%) of the overall time horizon set for anticipation.
    # For example, if the model’s time horizon is 5 minutes, eMAE focuses on evaluating the error for events predicted to happen within the first 0.5 minutes (30 seconds). This metric is crucial in scenarios like robotic-assisted surgery, where accurately predicting imminent events is essential for timely interventions and adjustments. A lower eMAE indicates that the model is very effective at making rapid, short-term predictions, which is important for real-time decision-making during surgical procedures.
    e_indices = time_gts < (0.1 * h)
    e_errors = abs_errors[e_indices] if np.any(e_indices) else np.array([])
    eMAE = np.mean(e_errors) if e_errors.size > 0 else 0

    print("\nAnticipation Evaluation Metrics for Cholec80:")
    print(f"inMAE (in-horizon MAE): {inMAE:.4f}")
    print(f"oMAE (out-of-horizon MAE): {oMAE:.4f}")
    print(f"wMAE (weighted MAE): {wMAE:.4f}")
    print(f"eMAE (very-short-term MAE): {eMAE:.4f}")

    # 3. RSD Evaluation (Remaining Surgical Duration)
    # Since this evaluation is specific to RSD anticipation and the Cholec80 dataset is used for phase anticipation,
    # we do not compute RSD metrics here. For RSD evaluation, you would typically use the Cataract101 dataset.
    # print("\nRSD evaluation is not applicable for the Cholec80 (phase anticipation) dataset.")

    with open(f"results_cholec80_split={split}.txt", "w") as fp:
        fp.write("Phase Classification Evaluation:\n")
        fp.write(f"Accuracy: {accuracy:.4f}\n")
        fp.write(f"Precision: {precision:.4f}\n")
        fp.write(f"Recall: {recall:.4f}\n")
        fp.write(f"F1 Score: {f1:.4f}\n")
        fp.write("Confusion Matrix:\n")
        fp.write(f"{conf_mat}\n")
        fp.write("Classification Report:\n")
        for cls, metrics in class_report.items():
            fp.write(f"{cls}: {metrics}\n")
        fp.write("\nAnticipation Evaluation Metrics for Cholec80:\n")
        fp.write(f"inMAE (in-horizon MAE): {inMAE:.4f}\n")
        fp.write(f"oMAE (out-of-horizon MAE): {oMAE:.4f}\n")
        fp.write(f"wMAE (weighted MAE): {wMAE:.4f}\n")
        fp.write(f"eMAE (very-short-term MAE): {eMAE:.4f}\n")


if __name__ == "__main__":
    eval_model()
