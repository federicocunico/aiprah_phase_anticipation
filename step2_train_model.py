import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import wandb
from datasets.cholec80 import Cholec80Dataset
from models.membank_model import MemBankResNetLSTM
from models.temporal_model import TemporalResNetLSTM
from memory_bank_utils import create_memorybank, get_long_range_feature_clip, get_long_range_feature_clip_online


def test(sequence_length, num_classes, chkpt_dst, test_loader, device, backbone=None):
    if backbone is None:
        backbone = MemBankResNetLSTM(sequence_length=sequence_length, num_classes=num_classes)
        backbone.to(device)

    model = TemporalResNetLSTM(backbone=backbone, sequence_length=sequence_length, num_classes=num_classes)
    model.to(device)

    model.load_state_dict(torch.load(chkpt_dst, weights_only=True))

    # get training and validation memory banks
    train_set = Cholec80Dataset(root_dir="./data/cholec80", mode="train", seq_len=sequence_length, fps=1)
    val_set = Cholec80Dataset(root_dir="./data/cholec80", mode="val", seq_len=sequence_length, fps=1)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=False, num_workers=30, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=30, pin_memory=True)
    train_mb, val_mb = create_memorybank(
        model=backbone, train_loader=train_loader, val_loader=val_loader, device=device
    )
    mb = torch.cat([torch.as_tensor(train_mb), torch.as_tensor(val_mb)], dim=0)
    mb = mb.to(device)

    mse = nn.MSELoss()

    # test
    model.eval()
    y_true = []
    y_pred = []
    y_true_regr = []
    y_pred_regr = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Test"):
            frames, metadata = batch
            frames = frames.to(device)
            labels = metadata["phase_label"].to(device)
            time_to_next_phase = metadata["time_to_next_phase"]  # .to(device)
            time_to_next_phase = torch.clamp(time_to_next_phase, 0, model.max_anticipation)

            long_range_features = get_long_range_feature_clip_online(model=model, inputs=frames, MB=mb)

            current_phase, anticipated_phase = model(frames, long_range_features)
            anticipated_phase = anticipated_phase.squeeze()  # [B, 1] -> [B]

            _, predicted = torch.max(current_phase, 1)
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(predicted.detach().cpu().numpy())

            y_true_regr.extend(time_to_next_phase.detach().cpu().numpy())
            y_pred_regr.extend(anticipated_phase.detach().cpu().numpy())

            if i % 10 == 0 and i > 0:
                print(f"Test Batch {i}/{len(test_loader)}; partial acc: {accuracy_score(y_true, y_pred):.2f}")

    acc = accuracy_score(y_true, y_pred)
    mse_loss = mse(torch.as_tensor(y_pred_regr), torch.as_tensor(y_true_regr))
    wandb.log({"test_acc": acc})
    wandb.log({"test_mse_loss": mse_loss})
    print(f"Test Accuracy: {acc:.2f}")
    print(f"Test MSE Loss: {mse_loss:.2f}")

    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    with open(chkpt_dst.replace(".pth", ".txt"), "a") as f:
        f.write(f"Test Accuracy: {acc:.2f}\n")
        f.write(f"Test MSE Loss: {mse_loss:.2f}\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"{cm}\n")


def train_model():

    run = wandb.init(project="main_model")
    # -------------------
    # Hyperparameters
    # -------------------
    batch_size = 4  # 16
    seq_len = 10
    epochs = 25
    target_fps = 1
    num_classes = 7
    multitask_strategy = "lambda" # "adaptive_weighting" # "none"
    l1 = 1
    l2 = 0.3
    mb_pretrained_model = "./wandb/run-stage1/checkpoints/membank_best.pth"
    chkpt_dst = os.path.join(run.dir, "..", "checkpoints", "main_model_best.pth")

    device = torch.device("cuda:0")

    # -------------------
    if not os.path.exists(os.path.dirname(chkpt_dst)):
        os.makedirs(os.path.dirname(chkpt_dst))

    torch.manual_seed(0)
    np.random.seed(0)

    train_dataset = Cholec80Dataset(root_dir="./data/cholec80", mode="train", seq_len=seq_len, fps=target_fps)
    val_dataset = Cholec80Dataset(root_dir="./data/cholec80", mode="val", seq_len=seq_len, fps=target_fps)
    test_dataset = Cholec80Dataset(root_dir="./data/cholec80", mode="test", seq_len=seq_len, fps=target_fps)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=30, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=30, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=30, pin_memory=True)

    # -------------------
    # Models
    # -------------------
    backbone = MemBankResNetLSTM(sequence_length=seq_len, num_classes=num_classes)
    backbone.to(device)
    assert os.path.isfile(mb_pretrained_model), "Pretrained model not found"
    backbone.load_state_dict(torch.load(mb_pretrained_model))

    model = TemporalResNetLSTM(backbone=backbone, sequence_length=seq_len, num_classes=num_classes)
    model.to(device)

    train_mb, val_mb = create_memorybank(
        model=backbone, train_loader=train_loader, val_loader=val_loader, device=device
    )
    train_mb = torch.as_tensor(train_mb)
    val_mb = torch.as_tensor(val_mb)
    mb = torch.cat([train_mb, val_mb], dim=0)

    mb = mb.squeeze()  # Shape: [N, nfeats]
    MB_windows = mb.unfold(0, seq_len, 1).permute(0, 2, 1)  # Shape: [N-T+1, T, F]
    MB_windows = torch.nn.functional.normalize(MB_windows, dim=-1)  # Normalize along feature dimension
    mb_norm = MB_windows.to(device)

    # -------------------
    # Train
    # -------------------

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion_phase = nn.CrossEntropyLoss()
    criterion_anticipation = nn.MSELoss()

    l = len(train_loader)

    best_acc = 0
    best_mse_loss = 1e6
    for e in range(epochs):
        model.train()
        for i, batch in tqdm(enumerate(train_loader), total=l, desc=f"Epoch {e}"):
            frames, metadata = batch
            frames = frames.to(device)
            labels = metadata["phase_label"].to(device)
            time_to_next_phase = metadata["time_to_next_phase"].to(device)  # Get transition probabilities
            time_to_next_phase = torch.clamp(time_to_next_phase, 0, model.max_anticipation)

            # long_range_features = get_long_range_feature_clip(inputs=frames, metadata=metadata, MB=train_mb)
            long_range_features = get_long_range_feature_clip_online(model=model, inputs=frames, MB=mb, MB_norm=mb_norm)

            current_phase, anticipated_phase = model(frames, long_range_features)
            anticipated_phase = anticipated_phase.squeeze()  # [B, 1] -> [B]
            # scale anticipated phase, cut if it is greater than model.max_anticipation (default=5 mintues)
            time_to_next_phase = torch.clamp(time_to_next_phase, 0, model.max_anticipation)

            loss_phase: torch.Tensor = criterion_phase(current_phase, labels)
            # loss_anticipation = model.compute_kl_divergence(anticipated_phase, future_transition_probs)
            loss_anticipation: torch.Tensor = criterion_anticipation(anticipated_phase.reshape(-1), time_to_next_phase)

            if multitask_strategy == "adaptive_weighting":
                # Compute dynamic weights based on current loss magnitudes
                weight_phase = 1.0 / (loss_phase.item() + 1e-8)  # Add a small constant to prevent division by zero
                weight_anticipation = 1.0 / (loss_anticipation.item() + 1e-8)

                # Normalize weights
                total_weight = weight_phase + weight_anticipation
                weight_phase /= total_weight
                weight_anticipation /= total_weight

                # Compute weighted loss
                loss = weight_phase * loss_phase + weight_anticipation * loss_anticipation
            elif multitask_strategy == "lambda":
                loss = l1 * loss_phase + l2 * loss_anticipation
            elif multitask_strategy == "none" or multitask_strategy is None:  # none
                loss = loss_phase + loss_anticipation
            else:
                raise ValueError(f"Invalid multitask strategy: {multitask_strategy}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 1000 == 0 and i > 0:
                wandb.log({"phase_loss": loss_phase.item(), "anticipation_loss": loss_anticipation.item()})
                # print(f"Epoch {e}, Batch {i}/{l}; Phase Loss: {loss_phase.item():.2f} Anticipation Loss: {loss_anticipation.item():.2f}")
            # print every 30% of the epoch
            # if i % (l // 3) == 0 and i > 0:
            #     print(f"Epoch {e}, Batch {i}/{l}; Phase Loss: {loss_phase.item():.2f} Anticipation Loss: {loss_anticipation.item():.2f}")

        # validation
        model.eval()
        y_true = []
        y_pred = []
        y_true_regr = []
        y_pred_regr = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Val Epoch {e}"):
                frames, metadata = batch
                frames = frames.to(device)
                labels = metadata["phase_label"].to(device)
                time_to_next_phase = metadata["time_to_next_phase"]  # .to(device)
                time_to_next_phase = torch.clamp(time_to_next_phase, 0, model.max_anticipation)

                # long_range_features = get_long_range_feature_clip(inputs=frames, metadata=metadata, MB=val_mb)
                long_range_features = get_long_range_feature_clip_online(
                    model=model, inputs=frames, MB=mb, MB_norm=mb_norm
                )

                current_phase, anticipated_phase = model(frames, long_range_features)
                anticipated_phase = anticipated_phase.squeeze()  # [B, 1] -> [B]

                _, predicted = torch.max(current_phase, 1)
                y_true.extend(labels.detach().cpu().numpy())
                y_pred.extend(predicted.detach().cpu().numpy())

                y_true_regr.extend(time_to_next_phase.detach().cpu().numpy())
                y_pred_regr.extend(anticipated_phase.detach().cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        mse_loss = criterion_anticipation(torch.as_tensor(y_pred_regr), torch.as_tensor(y_true_regr))
        wandb.log({"val_acc": acc})
        wandb.log({"val_mse_loss": mse_loss})
        print(f"Epoch {e}, Validation Accuracy: {acc:.2f} MSE Loss: {mse_loss:.2f}")
        if acc > best_acc or mse_loss > best_mse_loss:
            print("Saving best model")
            best_acc = acc
            best_mse_loss = mse_loss
            with open(chkpt_dst.replace(".pth", ".txt"), "w") as f:
                f.write(f"Epoch: {e}\n")
                f.write(f"Validation Accuracy: {acc:.2f}\n")
                f.write(f"Validation MSE Loss: {mse_loss:.2f}\n")
            torch.save(model.state_dict(), chkpt_dst)

    test(
        backbone=backbone,
        sequence_length=seq_len,
        num_classes=num_classes,
        chkpt_dst=chkpt_dst,
        test_loader=test_loader,
        device=device,
    )

    run.finish()

    return chkpt_dst


if __name__ == "__main__":
    train_model()

    # batch_size = 16
    # seq_len = 10
    # target_fps = 1
    # num_classes = 7
    # test_dataset = Cholec80Dataset(root_dir="./data/cholec80", mode="test", seq_len=seq_len, fps=target_fps)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=30, pin_memory=True)
    # test(
    #     backbone=None,
    #     sequence_length=10,
    #     num_classes=7,
    #     chkpt_dst="./wandb/run-stage2/checkpoints/main_model_best.pth",
    #     test_loader=test_loader,
    #     device=torch.device("cuda:0"),
    # )
