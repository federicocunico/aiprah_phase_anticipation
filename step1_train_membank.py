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
from memory_bank_utils import create_memorybank
from models.membank_model import MemBankResNetLSTM


def train_membank():

    run = wandb.init(project="membank")

    # -------------------
    # Hyperparameters
    # -------------------
    batch_size = 16
    seq_len = 10
    epochs = 25
    target_fps = 1
    num_classes = 7
    chkpt_dst = os.path.join(run.dir, "..", "checkpoints", "membank_best.pth")
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

    model = MemBankResNetLSTM(sequence_length=seq_len, num_classes=num_classes)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    l = len(train_loader)
    best_acc = 0
    for e in range(epochs):
        batch: tuple[torch.Tensor, dict[str, torch.Tensor]]

        model.train()
        for i, batch in tqdm(enumerate(train_loader), total=l, desc=f"Epoch {e}"):
            frames, metadata = batch
            frames = frames.to(device)
            labels = metadata["phase_label"].to(device)
            y = model(frames)
            y = y[seq_len - 1 :: seq_len]  # get the last prediction of each sequence

            loss: torch.Tensor = criterion(y, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                # print(f"Epoch {e}, Batch {i}/{l}, Loss: {loss.item()}")
                wandb.log({"loss": loss.item()})

        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Val Epoch {e}"):
                frames, metadata = batch
                frames = frames.to(device)
                labels = metadata["phase_label"].to(device)
                y = model(frames)
                y = y[seq_len - 1 :: seq_len]  # get the last prediction of each sequence

                _, predicted = torch.max(y, 1)
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

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Test Epoch {e}"):
            frames, metadata = batch
            frames =frames.to(device)
            labels = metadata["phase_label"].to(device)
            y = model(frames)
            y = y[seq_len - 1 :: seq_len]  # get the last prediction of each sequence

            _, predicted = torch.max(y, 1)
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(predicted.detach().cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"Test Accuracy: {acc:.2f}")
    with open("membank_test_results.txt", "w") as f:
        f.write(f"Epoch: {e}\n")
        f.write(f"Test Accuracy: {acc:.2f}\n")
        f.write(f"Confusion Matrix:\n")
        f.write(str(cm))
    np.save("membank_test_results.npy", cm)

    wandb.log({"test_acc": acc})
    run.finish()

    # load best model
    model.load_state_dict(torch.load(chkpt_dst))
    create_memorybank(model, train_loader, val_loader, device, "./data/cholec80/membank.pth")


if __name__ == "__main__":
    train_membank()
