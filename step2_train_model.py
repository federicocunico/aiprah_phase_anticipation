import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from losses.loss import ModelLoss
from losses.swag_loss import SWAGLoss
from losses.ce_regr_loss import CERegrLoss
from losses.weighted_regr_loss import WeightedMSELoss
from trainer import PhaseAnticipationTrainer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from datasets.cholec80 import Cholec80Dataset
from models.membank_model import MemBankResNetLSTM
from models.memory_bank import MemoryBank
from models.temporal_model_v1 import TemporalResNetLSTM

# from models.temporal_model_v2 import TemporalAnticipationModel
# from models.temporal_model_v3 import TemporalAnticipationModel
# from models.temporal_model_v4 import TemporalAnticipationModel
from models.temporal_model_v5 import TemporalAnticipationModel
from memory_bank_utils import (
    create_memorybank,
    get_long_range_feature_clip,
    get_long_range_feature_clip_online,
)


def train_model(time_horizon: int):
    seed_everything(0)

    log_dir = f"checkpoints/stage2_horizon={time_horizon}"
    wandb_logger = WandbLogger(
        name=f"stage2-time_horizon={time_horizon}", project="cholec80"
    )
    # -------------------
    # Hyperparameters
    # -------------------
    torch.set_float32_matmul_precision("high")  # 'high' or 'medium'; for RTX GPUs
    batch_size = 8  # 16  # 16
    seq_len = 30  # 10
    epochs = 50
    target_fps = 1
    num_classes = 7
    # multitask_strategy = "none"  # "adaptive_weighting" # "none"
    # l1 = 1
    # l2 = 0.05
    # mb_pretrained_model = "./wandb/run-stage1/checkpoints/membank_best.pth"

    # device = torch.device("cuda:0")

    train_dataset = Cholec80Dataset(
        root_dir="./data/cholec80", mode="train", seq_len=seq_len, fps=target_fps
    )
    val_dataset = Cholec80Dataset(
        root_dir="./data/cholec80", mode="val", seq_len=seq_len, fps=target_fps
    )
    test_dataset = Cholec80Dataset(
        root_dir="./data/cholec80", mode="test", seq_len=seq_len, fps=target_fps
    )
    # train_dataset = Cholec80Dataset(root_dir="./data/cholec80", mode="demo_train", seq_len=seq_len, fps=target_fps)
    # val_dataset = Cholec80Dataset(root_dir="./data/cholec80", mode="demo_val", seq_len=seq_len, fps=target_fps)
    # test_dataset = Cholec80Dataset(root_dir="./data/cholec80", mode="demo_test", seq_len=seq_len, fps=target_fps)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=30,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=30,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=30,
        pin_memory=True,
    )

    # -------------------
    # Models
    # -------------------
    # backbone = MemBankResNetLSTM(sequence_length=seq_len, num_classes=num_classes)
    # backbone.to(device)
    # assert os.path.isfile(mb_pretrained_model), "Pretrained model not found"
    # backbone.load_state_dict(torch.load(mb_pretrained_model))
    # mb: MemoryBank = create_memorybank(
    #     model=backbone,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     device=device,
    #     membank_size=len(train_dataset),
    # )
    # model = TemporalResNetLSTM(backbone=backbone, memory_bank=mb, sequence_length=seq_len, num_classes=num_classes)

    model = TemporalAnticipationModel(
        time_horizon=time_horizon, sequence_length=seq_len, num_classes=num_classes
    )

    # -------------------
    # Train
    # -------------------
    # loss_criterion = ModelLoss(l1=l1, l2=l2, multitask_strategy=multitask_strategy)
    # loss_criterion = SWAGLoss(10, 5)
    loss_criterion = nn.SmoothL1Loss(reduction="mean")  # WeightedMSELoss()
    pl_model = PhaseAnticipationTrainer(model=model, loss_criterion=loss_criterion)

    ckpt_save = ModelCheckpoint(
        dirpath=f"{log_dir}",
        filename=f"e={{epoch}}-l={{val_loss_anticipation}}-val_acc={{val_acc}}_horizon={time_horizon}-stage2_model_best",
        # monitor="val_loss_anticipation",
        # mode="min",
        monitor="val_acc",
        mode="max",
        save_top_k=3,
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
    import argparse

    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--time_horizon", type=int, default=5, help="Time horizon for anticipation"
    )
    args = parser.parse_args()

    train_model(time_horizon=args.time_horizon)
