import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from eval import eval_model
from losses.loss import ModelLoss
from losses.swag_loss import SWAGLoss
from losses.ce_regr_loss import CERegrLoss
from losses.weighted_regr_loss import WeightedMSELoss
from trainer import PhaseAnticipationTrainer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from datasets.cholec80 import Cholec80Dataset
from datasets.peg_and_ring_workflow import PegAndRing
from datasets.heichole import HeiCholeDataset
# from models.membank_model import MemBankResNetLSTM
# from models.memory_bank import MemoryBank
# from models.temporal_model_v1 import TemporalResNetLSTM

# from models.temporal_model_v2 import TemporalAnticipationModel
# from models.temporal_model_v3 import TemporalAnticipationModel
# from models.temporal_model_v4 import TemporalAnticipationModel
from models.temporal_model_v5 import TemporalAnticipationModel

# from memory_bank_utils import (
#     create_memorybank,
#     get_long_range_feature_clip,
#     get_long_range_feature_clip_online,
# )


def train_model(args):
    seed_everything(0)
    time_horizon = args.time_horizon

    log_dir = f"checkpoints/{args.exp_name}_stage2_horizon={time_horizon}"
    wandb_logger = WandbLogger(
        name=f"{args.exp_name}_seq_len={args.seq_len}-time_horizon={time_horizon}",
        project=args.wandb_project,
    )
    # -------------------
    # Hyperparameters
    # -------------------
    torch.set_float32_matmul_precision("high")  # 'high' or 'medium'; for RTX GPUs
    batch_size = args.batch_size  # 16  # 16
    seq_len = args.seq_len  # 10
    epochs = 50
    target_fps = 1
    num_classes = args.num_classes
    # multitask_strategy = "none"  # "adaptive_weighting" # "none"
    # l1 = 1
    # l2 = 0.05
    # mb_pretrained_model = "./wandb/run-stage1/checkpoints/membank_best.pth"

    # device = torch.device("cuda:0")
    if args.dataset == "cholec80":
        data_dir = "./data/cholec80_fast"
        SWADataset = Cholec80Dataset
    elif args.dataset == "heichole":
        data_dir = "./data/heichole"
        SWADataset = HeiCholeDataset
    elif args.dataset == "peg_and_ring":
        data_dir = "./data/peg_and_ring_workflow"
        SWADataset = PegAndRing
    else:
        expected = ["cholec80", "heichole", "peg_and_ring"]
        raise ValueError(f"Unknown dataset: {args.dataset}, expected one of {expected}")

    train_dataset = SWADataset(
        root_dir=data_dir, mode="train", seq_len=seq_len, fps=target_fps
    )
    val_dataset = SWADataset(
        root_dir=data_dir, mode="val", seq_len=seq_len, fps=target_fps
    )
    test_dataset = SWADataset(
        root_dir=data_dir, mode="test", seq_len=seq_len, fps=target_fps
    )
    # train_dataset = Cholec80Dataset(root_dir="./data/cholec80", mode="demo_train", seq_len=seq_len, fps=target_fps)
    # val_dataset = Cholec80Dataset(root_dir="./data/cholec80", mode="demo_val", seq_len=seq_len, fps=target_fps)
    # test_dataset = Cholec80Dataset(root_dir="./data/cholec80", mode="demo_test", seq_len=seq_len, fps=target_fps)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=30,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=30,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=30,
        pin_memory=False,
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
        devices=[args.gpu],
        precision="bf16-mixed",
        max_epochs=epochs,
        callbacks=[ckpt_save],
        logger=wandb_logger,
        log_every_n_steps=25,
    )

    trainer.fit(pl_model, train_loader, val_loader)
    trainer.test(pl_model, test_loader, ckpt_path="best")

    # get best model
    final_ckpt = ckpt_save.best_model_path

    eval_model(
        "val",
        dataset=args.dataset,
        horizon=time_horizon,
        final_model=final_ckpt,
        seq_len=args.seq_len,
        num_classes=args.num_classes,
        exp_name=f"{args.dataset}/seq_len={args.seq_len}",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU device index to use for training"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="SWA", help="WandB project name"
    )
    parser.add_argument("--exp_name", type=str, default="TEST", help="Experiment name")
    parser.add_argument(
        "--dataset", type=str, default="heichole", help="Dataset to use for training"
    )
    parser.add_argument(
        "--time_horizon", type=int, default=5, help="Time horizon for anticipation"
    )
    parser.add_argument(
        "--seq_len", type=int, default=30, help="Sequence length for the model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--num_classes", type=int, default=7, help="Number of classes for the model"
    )
    args = parser.parse_args()

    train_model(args)
