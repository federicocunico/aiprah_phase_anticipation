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
from datasets.peg_and_ring_workflow import PegAndRing
from trainer import PhaseAnticipationTrainer
from models.temporal_model_v5 import TemporalAnticipationModel
from datasets.cholec80 import Cholec80Dataset
import numpy as np


def aggregate_predictions(out, aggregator_window_size=5):
    """
    Aggregate predictions in non-overlapping windows.

    For each video, the number of aggregated samples is
        len(out[video_name]) // aggregator_window_size.

    For predictions:
      - Phase segmentation: majority vote.
      - Phase anticipation: element-wise mean.

    Ground truth values are subsampled from the center sample of each window.

    Parameters:
        out: dict mapping video names to lists of prediction dictionaries.
             Each dictionary must contain:
               - "pred_current_phase": int prediction.
               - "pred_anticipated_phase": list of floats.
               - "gt_current_phase": ground truth int.
               - "gt_anticipated_phase": ground truth list of floats.
               - Optionally, "frame_index": frame/time index.
        aggregator_window_size: size of the non-overlapping window.

    Returns:
        aggregated: dict with the same structure as `out` but with reduced length,
                    where for each video:
                        len(aggregated[video]) = len(out[video]) // aggregator_window_size.
    """
    aggregated = {}
    for video, predictions in out.items():
        # Sort by frame index if available.
        if "frame_index" in predictions[0]:
            predictions = sorted(predictions, key=lambda x: x["frame_index"])
        agg_video = []
        n = len(predictions)
        # Iterate with a step equal to aggregator_window_size.
        for i in range(0, n - aggregator_window_size + 1, aggregator_window_size):
            window = predictions[i : i + aggregator_window_size]
            # Aggregate predictions:
            seg_votes = [pred["pred_current_phase"] for pred in window]
            majority = int(np.bincount(seg_votes).argmax())
            anticipated_array = np.array(
                [pred["pred_anticipated_phase"] for pred in window]
            )
            mean_anticipated = anticipated_array.mean(axis=0).tolist()
            # Subsample GT values from the center sample.
            center = aggregator_window_size // 2
            frame_index = window[center].get("frame_index", i + center)
            gt_current_phase = window[center]["gt_current_phase"]
            gt_anticipated_phase = window[center]["gt_anticipated_phase"]
            agg_video.append(
                {
                    "frame_index": frame_index,
                    "pred_current_phase": majority,
                    "pred_anticipated_phase": mean_anticipated,
                    "gt_current_phase": gt_current_phase,
                    "gt_anticipated_phase": gt_anticipated_phase,
                }
            )
        aggregated[video] = agg_video
    return aggregated


def eval_model(
    split: str,
    horizon: int,
    final_model: str,
    seq_len: int,
    exp_name: str = None,
    num_classes: int = 7,
    dataset: str = "cholec80",
):
    torch.set_float32_matmul_precision("high")  # 'high' or 'medium'; for RTX GPUs
    # seq_len = 30
    target_fps = 1
    device = torch.device("cuda:0")

    if exp_name is None:
        exp_name = f"cholec80"

    fname = f"results/{exp_name}/eval_split={split}_horizon={horizon}.pickle"

    # makedirs of fname
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    model = TemporalAnticipationModel(
        time_horizon=horizon, sequence_length=seq_len, num_classes=num_classes
    )
    # If evaluation output does not exist, run the model on the dataset and save outputs.
    if not os.path.exists(fname):
        if dataset == "cholec80":
            dataset_obj = Cholec80Dataset(
                root_dir="./data/cholec80", mode=split, seq_len=seq_len, fps=target_fps
            )
        elif dataset == "peg_and_ring":
            dataset_obj = PegAndRing(
                root_dir="./data/peg_and_ring_workflow",
                mode=split,
                seq_len=seq_len,
                fps=target_fps,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        pl_model = PhaseAnticipationTrainer(model=model, loss_criterion=None)

        # Load pretrained-checkpoint.
        # final_model = "checkpoints/e=epoch=23-l=val_loss_anticipation=0.027181584388017654-stage2_model_best.ckpt"
        pl_model.load_state_dict(
            torch.load(final_model, weights_only=True)["state_dict"]
        )

        # Set appropriate dtype.
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        dtype = torch.float32
        model = model.to(device=device, dtype=dtype)

        # Dummy forward pass to initialize the model.
        model(torch.randn(1, seq_len, 3, 224, 224).to(device=device, dtype=dtype))

        out = {}
        with torch.no_grad():
            model.eval()
            for i, (frames, metadata) in tqdm(
                enumerate(dataset_obj), total=len(dataset_obj)
            ):
                frames = frames.to(device=device, dtype=dtype).unsqueeze(0)
                start = time.time()
                model_output = model(frames)
                # Separate outputs depending on the model's output format.
                if isinstance(model_output, tuple):
                    current_phase, anticipated_phase = model_output
                else:
                    anticipated_phase = model_output
                    current_phase = torch.argmin(anticipated_phase, dim=1)

                # Calculate FPS if needed.
                fps = 1 / (time.time() - start)
                # Convert metadata tensors to CPU lists.
                m = {
                    k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in metadata.items()
                }
                # Ensure a frame index is available (fall back to the iteration index if not provided).
                frame_index = m.get("frame_index", i)
                out_data = {
                    "frame_index": frame_index,
                    "pred_current_phase": current_phase.cpu()
                    .numpy()
                    .tolist()[0],  # int
                    "pred_anticipated_phase": anticipated_phase.cpu()
                    .float()
                    .numpy()
                    .tolist()[0],  # list of floats, length=num_classes
                    "gt_anticipated_phase": torch.clamp(
                        torch.as_tensor(metadata["time_to_next_phase"]),
                        0,
                        model.time_horizon,
                    )
                    .cpu()
                    .numpy()
                    .tolist(),  # list of floats
                    "gt_current_phase": metadata["phase_label"]
                    .cpu()
                    .numpy()
                    .tolist(),  # int or list with one element
                    **m,
                }
                video_name = m["video_name"]
                if video_name not in out:
                    out[video_name] = []
                out[video_name].append(out_data)

        # Save the raw window-level predictions.
        with open(fname, "wb") as f:
            pickle.dump(out, f)
    else:
        with open(fname, "rb") as f:
            out = pickle.load(f)

    # Aggregate predictions (and subsample GT accordingly) using a sliding window of size 5.
    aggregated_out = aggregate_predictions(out, aggregator_window_size=5)

    # ---------------------------------------------------
    # Evaluation Metrics Computation on Aggregated Output
    # ---------------------------------------------------

    # 1. Phase Classification Metrics
    phase_preds = []
    phase_gts = []
    for video, samples in aggregated_out.items():
        for sample in samples:
            phase_preds.append(sample["pred_current_phase"])
            # Extract ground truth phase as a scalar.
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

    # Plot and save confusion matrix.
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_mat, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()
    plt.xticks(range(num_classes))
    plt.yticks(range(num_classes))
    plt.savefig(
        f"results/{exp_name}/{dataset}_{split}_confusion_matrix_horizon={horizon}.png"
    )
    plt.close()

    # 2. Anticipation Error Metrics (inMAE, oMAE, wMAE, eMAE)
    time_preds = []
    time_gts = []
    for video, samples in aggregated_out.items():
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
    h = model.time_horizon  # e.g., 5 minutes

    in_indices = time_gts < h
    in_errors = abs_errors[in_indices] if np.any(in_indices) else np.array([])
    inMAE = np.mean(in_errors) if in_errors.size > 0 else 0

    out_indices = time_gts == h
    out_errors = (
        np.abs(time_preds[out_indices] - h) if np.any(out_indices) else np.array([])
    )
    oMAE = np.mean(out_errors) if out_errors.size > 0 else 0

    wMAE = (inMAE + oMAE) / 2

    e_indices = time_gts < (0.1 * h)
    e_errors = abs_errors[e_indices] if np.any(e_indices) else np.array([])
    eMAE = np.mean(e_errors) if e_errors.size > 0 else 0

    print(f"\nAnticipation Evaluation Metrics for {dataset}:")
    print(f"inMAE (in-horizon MAE): {inMAE:.4f}")
    print(f"oMAE (out-of-horizon MAE): {oMAE:.4f}")
    print(f"wMAE (weighted MAE): {wMAE:.4f}")
    print(f"eMAE (very-short-term MAE): {eMAE:.4f}")

    # Save evaluation results to a text file.
    with open(
        f"results/{exp_name}/results_{dataset}_split={split}_horizon={horizon}.txt", "w"
    ) as fp:
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
        fp.write(f"\nAnticipation Evaluation Metrics for {dataset}:\n")
        fp.write(f"inMAE (in-horizon MAE): {inMAE:.4f}\n")
        fp.write(f"oMAE (out-of-horizon MAE): {oMAE:.4f}\n")
        fp.write(f"wMAE (weighted MAE): {wMAE:.4f}\n")
        fp.write(f"eMAE (very-short-term MAE): {eMAE:.4f}\n")

    # Creating visualizations
    print("Creating line-plots visualizations...")
    for video_name, video_data in aggregated_out.items():
        img_fname = f"results/{exp_name}/{dataset}_{video_name}_seq_len={seq_len}_split={split}_predicted_time_to_next_phase.png"
        os.makedirs(os.path.dirname(img_fname), exist_ok=True)
        arr = np.array([v["pred_anticipated_phase"] for v in video_data])
        gts = np.array([v["gt_anticipated_phase"] for v in video_data])
        # create a 7‐row subplot, one for each class
        fig, axs = plt.subplots(num_classes, 1, sharex=True, figsize=(12, 14))
        for i in range(num_classes):
            axs[i].plot(
                arr[:, i], linestyle="-", color="blue", label="Predicted", linewidth=2
            )
            axs[i].plot(
                gts[:, i],
                linestyle="--",
                color="red",
                label="Ground Truth",
                linewidth=2,
            )
            axs[i].set_ylabel(f"Phase {i}")
            axs[i].grid(True)
            axs[i].set_ylim(0, model.time_horizon * 1.05)

        # common labels and title
        axs[-1].set_xlabel("Frame index")
        fig.suptitle("Predicted Time to Next Phase")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(img_fname)


if __name__ == "__main__":
    # ## HORIZON = 5
    # eval_model(
    #     "test",
    #     horizon=5,
    #     final_model="checkpoints/e=epoch=23-l=val_loss_anticipation=0.027181584388017654-stage2_model_best.ckpt",
    # )
    # eval_model(
    #     "val",
    #     horizon=5,
    #     final_model="checkpoints/e=epoch=23-l=val_loss_anticipation=0.027181584388017654-stage2_model_best.ckpt",
    # )
    # ## HORIZON = 3
    # eval_model(
    #     "test",
    #     horizon=3,
    #     final_model="checkpoints/stage2_horizon=3/e=epoch=47-l=val_loss_anticipation=0.023974817246198654-val_acc=val_acc=0.8197687864303589_horizon=3-stage2_model_best.ckpt",
    # )
    # eval_model(
    #     "val",
    #     horizon=3,
    #     final_model="checkpoints/stage2_horizon=3/e=epoch=47-l=val_loss_anticipation=0.023974817246198654-val_acc=val_acc=0.8197687864303589_horizon=3-stage2_model_best.ckpt",
    # )
    # ## HORIZON = 1
    # eval_model(
    #     "test",
    #     horizon=1,
    #     final_model="checkpoints/stage2_horizon=1/e=epoch=12-l=val_loss_anticipation=0.022330984473228455-val_acc=val_acc=0.829586386680603horizon=1-stage2_model_best.ckpt",
    # )
    # eval_model(
    #     "val",
    #     horizon=1,
    #     final_model="checkpoints/stage2_horizon=1/e=epoch=12-l=val_loss_anticipation=0.022330984473228455-val_acc=val_acc=0.829586386680603horizon=1-stage2_model_best.ckpt",
    # )

    eval_model(
        "val",
        horizon=5,
        final_model="checkpoints/cholec80_t=5_s=4_stage2_horizon=5/e=epoch=31-l=val_loss_anticipation=0.03390774503350258-val_acc=val_acc=0.7710379958152771_horizon=5-stage2_model_best.ckpt",
        seq_len=4,
        exp_name="cholec80/seq_len=4",
    )

    eval_model(
        "val",
        horizon=5,
        final_model="checkpoints/cholec80_t=5_s=10_stage2_horizon=5/e=epoch=41-l=val_loss_anticipation=0.030996935442090034-val_acc=val_acc=0.7872610092163086_horizon=5-stage2_model_best.ckpt",
        seq_len=10,
        exp_name="cholec80/seq_len=10",
    )

    eval_model(
        "val",
        horizon=5,
        final_model="checkpoints/cholec80_t=5_s=30_stage2_horizon=5/e=epoch=19-l=val_loss_anticipation=0.028951309621334076-val_acc=val_acc=0.8010921478271484_horizon=5-stage2_model_best.ckpt",
        seq_len=30,
        exp_name="cholec80/seq_len=30",
    )
