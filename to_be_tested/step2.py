import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Import your loss functions, trainer wrapper and dataset modules
from losses.loss import ModelLoss
from losses.swag_loss import SWAGLoss
from losses.ce_regr_loss import CERegrLoss
from losses.weighted_regr_loss import WeightedMSELoss
from trainer import PhaseAnticipationTrainer
from datasets.cholec80 import Cholec80Dataset

# Import the updated temporal model that uses the pre-trained backbone
from models.temporal_model_v5 import TemporalAnticipationModel
from memory_bank_utils import create_memorybank, get_long_range_feature_clip, get_long_range_feature_clip_online

def train_model():
    seed_everything(0)

    log_dir = "checkpoints"
    wandb_logger = WandbLogger(name="stage2", project="cholec80")

    # -------------------
    # Hyperparameters
    # -------------------
    torch.set_float32_matmul_precision("high")  # For RTX GPUs
    batch_size = 8
    seq_len = 30
    epochs = 200
    target_fps = 1
    num_classes = 7
    multitask_strategy = "none"
    l1 = 1
    l2 = 0.05
    # mb_pretrained_model = "./wandb/run-stage1/checkpoints/membank_best.pth"  # Path to pre-trained weights
    mb_pretrained_model = "./checkpoints/membank_best.pth"  # Path to pre-trained weights

    device = torch.device("cuda:0")

    # -------------------
    # Data Loaders
    # -------------------
    train_dataset = Cholec80Dataset(root_dir="./data/cholec80", mode="train", seq_len=seq_len, fps=target_fps)
    val_dataset   = Cholec80Dataset(root_dir="./data/cholec80", mode="val",   seq_len=seq_len, fps=target_fps)
    test_dataset  = Cholec80Dataset(root_dir="./data/cholec80", mode="test",  seq_len=seq_len, fps=target_fps)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=30, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=30, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=30, pin_memory=True)

    # -------------------
    # Model Initialization
    # -------------------
    # Create the Temporal Anticipation Model.
    # This model uses the pre-trained backbone from stage 1.
    model = TemporalAnticipationModel(time_horizon=5, sequence_length=seq_len, num_classes=num_classes)
    
    # Load the pre-trained memory bank weights into the backbone.
    if os.path.isfile(mb_pretrained_model):
        state_dict = torch.load(mb_pretrained_model, map_location=device)
        model.backbone.load_state_dict(state_dict)
        print("Loaded pre-trained memory bank weights.")
    else:
        raise FileNotFoundError("Pre-trained memory bank model not found.")

    # -------------------
    # Loss, Trainer, and Callbacks
    # -------------------
    loss_criterion = nn.SmoothL1Loss(reduction='mean')  # Or WeightedMSELoss(), etc.
    pl_model = PhaseAnticipationTrainer(model=model, loss_criterion=loss_criterion)

    ckpt_save = ModelCheckpoint(
        dirpath=log_dir,
        filename="e={epoch}-l={val_loss_anticipation}-stage2_model_best",
        monitor="val_loss_anticipation",
        mode="min",
        save_top_k=1,
    )
    trainer = Trainer(
        accelerator="auto",
        precision="bf16-mixed",
        max_epochs=epochs,
        callbacks=[ckpt_save],
        logger=wandb_logger,
    )

    trainer.fit(pl_model, train_loader, val_loader)
    trainer.test(pl_model, test_loader, ckpt_path="best")

if __name__ == "__main__":
    train_model()
