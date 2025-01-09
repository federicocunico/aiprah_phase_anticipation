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
    mb_pretrained_model = os.path.join(run.dir, "..", "checkpoints", "membank_best.pth")
    device = torch.device("cuda:0")

    # -------------------

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

    MB = create_memorybank(model=backbone, train_loader=train_loader, val_loader=val_loader, device=device)

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
            long_range_features = get_long_range_feature_clip(inputs=frames, metadata=metadata, MB=MB)

            current_phase, anticipated_phase = model(frames, long_range_features)

            _, y = torch.max(current_phase, dim=1)
            loss_phase: torch.Tensor = criterion_phase(y, labels)

            optimizer.zero_grad()
            loss_phase.backward()
            optimizer.step()
