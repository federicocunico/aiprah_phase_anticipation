#!/usr/bin/env python3
# train_temporal_transformer_plain.py
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import random
from pathlib import Path
from typing import Dict, Any, DefaultDict, Optional, List, Iterator
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, BatchSampler

import matplotlib.pyplot as plt

from datasets.peg_and_ring_workflow import PegAndRing  # your dataset

# from models.temporal_transformer import TemporalTransformerLearnedPE as PlainModel
from models.temporal_windowed_transformer import WindowMemoryTransformer


# ---------------------------
# Fixed training hyperparams
# ---------------------------
SEED = 42
ROOT_DIR = "data/peg_and_ring_workflow"
TIME_UNIT = "minutes"  # dataset outputs in minutes (or "seconds")
NUM_CLASSES = 6

# Temporal windowing
SEQ_LEN = 10
STRIDE = 1

# Batching / epochs
BATCH_SIZE = 24
NUM_WORKERS = 30
EPOCHS = 20

# Optimizer
LR = 1e-4
WEIGHT_DECAY = 1e-4

# Model hyper
TIME_HORIZON = 2.0  # clamp targets/preds for stability in plots/metrics

PRINT_EVERY = 20
CKPT_PATH = Path("best_temporal_transformer_plain.pth")

# Eval output dirs
EVAL_ROOT = Path("eval_outputs")
EVAL_CLS_DIR = EVAL_ROOT / "classification"
EVAL_ANT_DIR = EVAL_ROOT / "anticipation"
EVAL_CLS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_ANT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# ---------------------------
# Evaluation (plain model)
# ---------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, time_horizon: float
) -> Dict[str, Any]:
    model.eval()

    total_mae = 0.0
    total_ce = 0.0
    total_acc = 0.0
    total_samples = 0

    ce_loss = nn.CrossEntropyLoss()

    for _, (frames, meta) in enumerate(loader):
        frames = frames.to(device)  # [B, T, 3, H, W]
        ttnp = meta["time_to_next_phase"].to(device).float()  # [B, C]
        ttnp = torch.clamp(ttnp, min=0.0, max=time_horizon)
        labels = meta["phase_label"].to(device).long()

        reg, logits = model(frames, meta)

        preds_cls = logits.argmax(dim=1)  # [B]
        acc = (preds_cls == labels).float().mean()
        mae = torch.mean(torch.abs(reg - ttnp))  # scalar
        ce = ce_loss(logits, labels)

        bs = frames.size(0)
        total_mae += float(mae.item()) * bs
        total_ce += float(ce.item()) * bs
        total_acc += float(acc.item()) * bs
        total_samples += bs

    return {
        "mae": total_mae / max(1, total_samples),
        "ce": total_ce / max(1, total_samples),
        "acc": total_acc / max(1, total_samples),
        "samples": total_samples,
    }


# ---------------------------
# Visualizations (plain model)
# ---------------------------
@torch.no_grad()
def visualize_phase_timelines_classification(
    model: nn.Module,
    root_dir: str,
    split: str = "test",
    time_unit: str = "minutes",
    batch_size: int = 64,
    num_workers: int = 6,
    out_dir: Path = EVAL_CLS_DIR,
):
    device = next(model.parameters()).device
    model.eval()

    ds = PegAndRing(
        root_dir=root_dir, mode=split, seq_len=SEQ_LEN, stride=1, time_unit=time_unit
    )

    # Fixed-size batches, per-video temporal order, deterministic video order
    gen = torch.Generator()
    gen.manual_seed(SEED)
    batch_sampler = VideoBatchSampler(
        ds,
        batch_size=batch_size,  # <-- always use a fixed batch size
        batch_videos=False,  # <-- never set True (no batch_size=None)
        shuffle_videos=False,  # keep original video order for plotting
        generator=gen,
    )
    loader = DataLoader(
        ds,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    lengths: Dict[str, int] = {
        vn: ds.ant_cache[vn].shape[0] for vn in ds.ant_cache.keys()
    }
    preds: DefaultDict[str, np.ndarray] = defaultdict(lambda: None)
    gts: DefaultDict[str, np.ndarray] = defaultdict(lambda: None)
    for vn, N in lengths.items():
        preds[vn] = -np.ones(N, dtype=np.int32)
        gts[vn] = -np.ones(N, dtype=np.int32)

    for frames, meta in loader:
        frames = frames.to(device)
        _, logits = model(frames, meta)

        pred_phase = logits.argmax(dim=1).detach().cpu().numpy()
        video_names = meta["video_name"]
        idx_last = meta["frames_indexes"][:, -1].cpu().numpy()
        gt_phase = meta["phase_label"].cpu().numpy()

        for vn, idx, p, g in zip(video_names, idx_last, pred_phase, gt_phase):
            idx = int(idx)
            if 0 <= idx < preds[vn].shape[0]:
                preds[vn][idx] = int(p)
            if 0 <= idx < gts[vn].shape[0]:
                gts[vn][idx] = int(g)

    for vn, N in lengths.items():
        pred_arr = preds[vn]
        gt_arr = gts[vn]
        x = np.arange(N)
        valid = (pred_arr >= 0) & (gt_arr >= 0)

        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        ax.step(
            x[valid],
            gt_arr[valid],
            where="post",
            linewidth=2,
            label="GT Phase",
            alpha=0.9,
        )
        ax.step(
            x[valid],
            pred_arr[valid],
            where="post",
            linewidth=2,
            label="Pred Phase",
            alpha=0.9,
        )

        ax.set_title(f"[{split}] Phase timeline — {vn}")
        ax.set_xlabel("Frame index (1 FPS)")
        ax.set_ylabel("Phase (0..5)")
        ax.set_yticks(list(range(NUM_CLASSES)))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right")

        out_path = out_dir / f"{vn}_phase_timeline_{split}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Saved classification timeline -> {out_path}")


@torch.no_grad()
def visualize_anticipation_curves(
    model: nn.Module,
    root_dir: str,
    split: str = "test",
    time_unit: str = "minutes",
    time_horizon: float = TIME_HORIZON,
    batch_size: int = 64,
    num_workers: int = 6,
    out_dir: Path = EVAL_ANT_DIR,
):
    device = next(model.parameters()).device
    model.eval()

    ds = PegAndRing(
        root_dir=root_dir, mode=split, seq_len=SEQ_LEN, stride=1, time_unit=time_unit
    )

    # Fixed-size batches, per-video temporal order, deterministic video order
    gen = torch.Generator()
    gen.manual_seed(SEED)
    batch_sampler = VideoBatchSampler(
        ds,
        batch_size=batch_size,  # <-- always use a fixed batch size
        batch_videos=False,  # <-- never set True (no batch_size=None)
        shuffle_videos=False,  # deterministic video order for clean curves
        generator=gen,
    )
    loader = DataLoader(
        ds,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    lengths: Dict[str, int] = {
        vn: ds.ant_cache[vn].shape[0] for vn in ds.ant_cache.keys()
    }
    preds: DefaultDict[str, np.ndarray] = defaultdict(lambda: None)
    gts: DefaultDict[str, np.ndarray] = defaultdict(lambda: None)
    for vn, N in lengths.items():
        preds[vn] = np.full((N, NUM_CLASSES), np.nan, dtype=np.float32)
        gts[vn] = np.full((N, NUM_CLASSES), np.nan, dtype=np.float32)

    for frames, meta in loader:
        frames = frames.to(device)
        outputs, _ = model(frames, meta)  # logits unused here

        pred = torch.clamp(outputs, min=0.0, max=time_horizon).detach().cpu().numpy()
        gt = (
            torch.clamp(meta["time_to_next_phase"], min=0.0, max=time_horizon)
            .cpu()
            .numpy()
        )

        video_names = meta["video_name"]
        idx_last = meta["frames_indexes"][:, -1].cpu().numpy()

        for vn, idx, p_row, g_row in zip(video_names, idx_last, pred, gt):
            idx = int(idx)
            if 0 <= idx < preds[vn].shape[0]:
                preds[vn][idx, :] = p_row
            if 0 <= idx < gts[vn].shape[0]:
                gts[vn][idx, :] = g_row

    unit_tag = time_unit.lower()
    for vn, N in lengths.items():
        arr = preds[vn]
        gt = gts[vn]
        valid = np.isfinite(arr).all(axis=1) & np.isfinite(gt).all(axis=1)
        x = np.arange(N)[valid]
        arr = arr[valid]
        gt = gt[valid]

        fig, axs = plt.subplots(6, 1, sharex=True, figsize=(12, 12))
        y_upper = time_horizon * 1.05

        for i in range(NUM_CLASSES):
            axs[i].plot(
                x,
                arr[:, i],
                linestyle="-",
                color="blue",
                label="Predicted",
                linewidth=2,
            )
            axs[i].plot(
                x,
                gt[:, i],
                linestyle="--",
                color="red",
                label="Ground Truth",
                linewidth=2,
            )
            axs[i].set_ylabel(f"Phase {i}")
            axs[i].grid(True, linestyle="--", alpha=0.4)
            axs[i].set_ylim(0, y_upper)

        axs[-1].set_xlabel("Frame index (1 FPS)")
        fig.suptitle(
            f"[{split}] Time to Next Phase — {vn} ({unit_tag}, horizon={time_horizon})"
        )

        handles, labels = axs[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = (
            out_dir / f"{vn}_anticipation_{unit_tag}_h{time_horizon:g}_{split}.png"
        )
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Saved anticipation curves -> {out_path}")


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ---------------------------
    # Datasets
    # ---------------------------
    train_ds = PegAndRing(
        root_dir=ROOT_DIR,
        mode="train",
        seq_len=SEQ_LEN,
        stride=STRIDE,
        time_unit=TIME_UNIT,
    )
    val_ds = PegAndRing(
        root_dir=ROOT_DIR,
        mode="val",
        seq_len=SEQ_LEN,
        stride=STRIDE,
        time_unit=TIME_UNIT,
    )
    test_ds = PegAndRing(
        root_dir=ROOT_DIR,
        mode="test",
        seq_len=SEQ_LEN,
        stride=STRIDE,
        time_unit=TIME_UNIT,
    )

    # ---------------------------
    # Dataloaders with video-aware samplers
    # ---------------------------
    gen_train = torch.Generator()
    gen_train.manual_seed(SEED)
    train_batch_sampler_fixed = VideoBatchSampler(
        train_ds,
        batch_size=BATCH_SIZE,
        drop_last=False,
        shuffle_videos=True,
        generator=gen_train,
        batch_videos=False,  # fixed-size batches within a video
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler_fixed,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # For validation and test: one video per batch, deterministic order
    gen_eval = torch.Generator()
    gen_eval.manual_seed(SEED)
    val_batch_sampler = VideoBatchSampler(
        val_ds,
        batch_size=BATCH_SIZE,
        batch_videos=False,
        shuffle_videos=False,
        generator=gen_eval,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    test_batch_sampler = VideoBatchSampler(
        test_ds,
        batch_size=BATCH_SIZE,
        batch_videos=False,
        shuffle_videos=False,
        generator=gen_eval,
    )
    test_loader = DataLoader(
        test_ds,
        batch_sampler=test_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # ---------------------------
    # Model (plain)
    # ---------------------------
    model = WindowMemoryTransformer(
        sequence_length=SEQ_LEN,
        num_classes=NUM_CLASSES,
        time_horizon=TIME_HORIZON,
        backbone="resnet50",
        pretrained_backbone=True,
        freeze_backbone=False,
        d_model=256,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        reasoning_time=10,  # M * SEQ_LEN seconds
        prior_mode="hard",
        reg_activation="softplus",  # or "linear"
    ).to(device)

    # ---------------------------
    # Losses / Optim
    # ---------------------------
    smooth_l1 = nn.SmoothL1Loss(reduction="mean")
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ---------------------------
    # Training loop
    # ---------------------------
    best_val_mae = float("inf")

    if True:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            epoch_mae = 0.0
            epoch_ce = 0.0
            epoch_loss = 0.0
            epoch_acc = 0.0
            seen = 0

            t0 = time.time()
            model.reset_all_memory()

            for it, (frames, meta) in enumerate(train_loader, start=1):
                frames = frames.to(device)  # [B, T, 3, H, W]
                labels = meta["phase_label"].to(device).long()
                ttnp = torch.clamp(
                    meta["time_to_next_phase"].to(device).float(), 0.0, TIME_HORIZON
                )

                reg, logits, aux = model.forward(frames, meta, return_aux=True)

                phase_loss = ce_loss(logits, labels)
                anticipation_loss = smooth_l1(reg, ttnp)
                train_loss = anticipation_loss + phase_loss

                optimizer.zero_grad(set_to_none=True)
                train_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    pred_cls = logits.argmax(dim=1)
                    train_mae = torch.mean(torch.abs(reg - ttnp))
                    train_acc = (pred_cls == labels).float().mean()

                bs = frames.size(0)
                epoch_mae += float(train_mae.item()) * bs
                epoch_ce += float(phase_loss.item()) * bs
                epoch_loss += float(train_loss.item()) * bs
                epoch_acc += float(train_acc.item()) * bs
                seen += bs

                if it % PRINT_EVERY == 0:
                    print(
                        f"[Epoch {epoch:02d} | it {it:04d}] "
                        f"loss={train_loss.item():.4f} | mae={train_mae.item():.4f} | "
                        f"acc={train_acc.item():.4f} | reg={anticipation_loss.item():.4f} | "
                        f"cls={phase_loss.item():.4f}"
                    )

            train_loss_avg = epoch_loss / max(1, seen)
            train_mae_avg = epoch_mae / max(1, seen)
            train_ce_avg = epoch_ce / max(1, seen)
            train_acc_avg = epoch_acc / max(1, seen)
            print(
                f"\nEpoch {epoch:02d} [{time.time()-t0:.1f}s] "
                f"| train_loss={train_loss_avg:.4f} train_mae={train_mae_avg:.4f} "
                f"train_ce={train_ce_avg:.4f} train_acc={train_acc_avg:.4f}"
            )

            # ---------------------------
            # Validation
            # ---------------------------
            val_stats = evaluate(model, val_loader, device, TIME_HORIZON)
            print(
                f"           val_mae={val_stats['mae']:.4f} val_ce={val_stats['ce']:.4f} "
                f"val_acc={val_stats['acc']:.4f} | samples={val_stats['samples']}"
            )

            if val_stats["mae"] < best_val_mae:
                best_val_mae = val_stats["mae"]
                torch.save(model.state_dict(), CKPT_PATH)
                print(
                    f"✅  New best val_mae={best_val_mae:.4f} — saved to: {CKPT_PATH}"
                )

    # ---------------------------
    # Test + Visualizations
    # ---------------------------
    if CKPT_PATH.exists():
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        print(f"\nLoaded best model from: {CKPT_PATH}")

    model.reset_all_memory()
    test_stats = evaluate(model, test_loader, device, TIME_HORIZON)
    print(
        f"\nTEST — mae={test_stats['mae']:.4f} ce={test_stats['ce']:.4f} "
        f"acc={test_stats['acc']:.4f} samples={test_stats['samples']}"
    )
    model.reset_all_memory()

    visualize_phase_timelines_classification(
        model=model,
        root_dir=ROOT_DIR,
        split="test",
        time_unit=TIME_UNIT,
        batch_size=BATCH_SIZE,
        num_workers=min(6, NUM_WORKERS),
        out_dir=EVAL_CLS_DIR,
    )
    model.reset_all_memory()

    visualize_anticipation_curves(
        model=model,
        root_dir=ROOT_DIR,
        split="test",
        time_unit=TIME_UNIT,
        time_horizon=TIME_HORIZON,
        batch_size=BATCH_SIZE,
        num_workers=min(6, NUM_WORKERS),
        out_dir=EVAL_ANT_DIR,
    )


if __name__ == "__main__":
    main()
