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
from memory_bank_utils import create_memorybank, get_long_range_feature_clip


def train_model():

    run = wandb.init(project="main_model")
    # -------------------
    # Hyperparameters
    # -------------------
    batch_size = 4
    seq_len = 10
    epochs = 25
    target_fps = 1
    num_classes = 7
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

    train_mb, val_mb = create_memorybank(model=backbone, train_loader=train_loader, val_loader=val_loader, device=device)
    train_mb = torch.as_tensor(train_mb)
    val_mb = torch.as_tensor(val_mb)

    # -------------------
    # Train
    # -------------------

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion_phase = nn.CrossEntropyLoss()

    l = len(train_loader)

    best_acc = 0
    for e in range(epochs):
        model.train()
        for i, batch in tqdm(enumerate(train_loader), total=l, desc=f"Epoch {e}"):
            frames, metadata = batch
            frames = frames.to(device)
            labels = metadata["phase_label"].to(device)
            future_transition_probs = metadata["future_transition_probs"].to(device)  # Get transition probabilities

            long_range_features = get_long_range_feature_clip(inputs=frames, metadata=metadata, MB=train_mb)

            current_phase, anticipated_phase = model(frames, long_range_features)

            loss_phase: torch.Tensor = criterion_phase(current_phase, labels)
            loss_anticipation = model.compute_kl_divergence(anticipated_phase, future_transition_probs)

            loss = loss_phase + loss_anticipation

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # validation
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Val Epoch {e}"):
                frames, metadata = batch
                frames = frames.to(device)
                labels = metadata["phase_label"].to(device)
                future_transition_probs = metadata["future_transition_probs"].to(device)

                long_range_features = get_long_range_feature_clip(inputs=frames, metadata=metadata, MB=val_mb)

                current_phase, anticipated_phase = model(frames, long_range_features)

                _, predicted = torch.max(current_phase, 1)
                y_true.extend(labels.detach().cpu().numpy())
                y_pred.extend(predicted.detach().cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        wandb.log({"val_acc": acc})
        print(f"Epoch {e}, Validation Accuracy: {acc:.2f}")
        if acc > best_acc:
            print("Saving best model")
            best_acc = acc
            with open(chkpt_dst.replace(".pth", ".txt"), "w") as f:
                f.write(f"Epoch: {e}\n")
                f.write(f"Validation Accuracy: {acc:.2f}\n")
            torch.save(model.state_dict(), chkpt_dst)

    # test
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Test Epoch {e}"):
            frames, metadata = batch
            frames = frames.to(device)
            labels = metadata["phase_label"].to(device)
            future_transition_probs = metadata["future_transition_probs"].to(device)

            long_range_features = get_long_range_feature_clip(inputs=frames, metadata=metadata, MB=val_mb)

            current_phase, anticipated_phase = model(frames, long_range_features)

            _, predicted = torch.max(current_phase, 1)
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(predicted.detach().cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    wandb.log({"test_acc": acc})
    print(f"Test Accuracy: {acc:.2f}")

    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    run.finish()



if __name__ == "__main__":
    train_model()
