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

from datasets.peg_and_ring_workflow import PegAndRing, VideoBatchSampler
from models.temporal_windowed_transformer import WindowMemoryTransformer

# ---------------------------
# Starter config (tuned for 1 FPS)
# ---------------------------
SEED = 42
ROOT_DIR = "data/peg_and_ring_workflow"
TIME_UNIT = "minutes"
NUM_CLASSES = 6

# Temporal windowing
SEQ_LEN = 16          # was 10; more temporal context helps stability at 1 FPS
STRIDE = SEQ_LEN//2

# Batching / epochs
BATCH_SIZE = 32       # requested
NUM_WORKERS = 30
EPOCHS = 40           # give cosine enough room

# Optimizer (group LRs used below; LR here is ignored)
LR = 1e-4
WEIGHT_DECAY = 5e-2   # slightly higher wd improves generalization for ViT-like heads

# Model hyper
TIME_HORIZON = 2.0
D_MODEL = 384
NHEAD = 6
ENC_LAYERS = 4
DEC_LAYERS = 1
FFN = 1536
DROPOUT = 0.10
REASONING_TIME = 8    # ~8 windows ≈ 8 seconds of per-video memory

# Scheduler (warmup + cosine)
WARMUP_EPOCHS = 5
ETA_MIN = 1e-6
CLIP_NORM = 1.0

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
# Visualizations (kept as-is)
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

    gen = torch.Generator()
    gen.manual_seed(SEED)
    batch_sampler = VideoBatchSampler(
        ds,
        batch_size=batch_size,
        batch_videos=False,
        shuffle_videos=False,
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

    gen = torch.Generator()
    gen.manual_seed(SEED)
    batch_sampler = VideoBatchSampler(
        ds,
        batch_size=batch_size,
        batch_videos=False,
        shuffle_videos=False,
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
                label="Predicted",
                linewidth=2,
            )
            axs[i].plot(
                x,
                gt[:, i],
                linestyle="--",
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


# ---------------------------
# Helper: create optimizer with param groups
# ---------------------------
def build_optimizer(model: WindowMemoryTransformer):
    # Backbone vs the rest (different learning rates)
    backbone_params, non_backbone_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if n.startswith("backbone.") else non_backbone_params).append(p)

    param_groups = [
        {"params": backbone_params, "lr": 1e-4, "weight_decay": WEIGHT_DECAY},
        {"params": non_backbone_params, "lr": 3e-4, "weight_decay": WEIGHT_DECAY},
    ]
    return optim.AdamW(param_groups)


def linear_warmup_step(opt: optim.Optimizer, base_lrs: List[float], epoch: int, warmup_epochs: int):
    """Scale each param group lr linearly during warmup."""
    scale = float(epoch + 1) / float(max(1, warmup_epochs))
    for pg, base in zip(opt.param_groups, base_lrs):
        pg["lr"] = base * scale


def get_base_lrs(opt: optim.Optimizer) -> List[float]:
    return [pg["lr"] for pg in opt.param_groups]


def unfreeze_backbone_last_block(model: WindowMemoryTransformer):
    """Unfreeze backbone.layer4 after warmup to increase learning capacity."""
    if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
        for p in model.backbone.layer4.parameters():
            p.requires_grad = True


def main():
    RUN_DEMO_ONLY = False  # set True to run the tiny CPU sanity demo and exit
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ---------------------------
    # Tiny CPU sanity demo (optional)
    # ---------------------------
    if RUN_DEMO_ONLY:
        m = WindowMemoryTransformer(
            sequence_length=SEQ_LEN,
            num_classes=NUM_CLASSES,
            time_horizon=TIME_HORIZON,
            backbone="resnet18",
            pretrained_backbone=False,
            freeze_backbone=False,
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=1,
            dim_feedforward=512,
            dropout=0.1,
            reasoning_time=3,
            prior_mode="hard",
            reg_activation="softplus",
        ).cpu().train()
        x = torch.randn(2, SEQ_LEN, 3, 224, 224)
        meta = {
            "video_name": ["video01", "video01"],
            "frames_indexes": torch.stack([torch.arange(0, SEQ_LEN), torch.arange(SEQ_LEN, 2*SEQ_LEN)], dim=0),
            "phase_label": torch.randint(0, NUM_CLASSES, (2,)),
            "time_to_next_phase": torch.rand(2, NUM_CLASSES),
        }
        y_reg, y_logits = m(x, meta)
        print("Sanity demo shapes:", y_reg.shape, y_logits.shape)
        return

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

    # For validation and test: fixed-size batches, deterministic video order
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
    # Model (starter config)
    # ---------------------------
    model = WindowMemoryTransformer(
        sequence_length=SEQ_LEN,
        num_classes=NUM_CLASSES,
        time_horizon=TIME_HORIZON,
        backbone="resnet50",
        pretrained_backbone=True,
        freeze_backbone=True,      # freeze during warmup for stability
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=ENC_LAYERS,
        num_decoder_layers=DEC_LAYERS,
        dim_feedforward=FFN,
        dropout=DROPOUT,
        reasoning_time=REASONING_TIME,
        prior_mode="hard",
        reg_activation="softplus",
    ).to(device)

    # ---------------------------
    # Losses / Optim / Scheduler (warmup + cosine)
    # ---------------------------
    smooth_l1 = nn.SmoothL1Loss(reduction="mean")
    ce_loss = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model)
    base_lrs = get_base_lrs(optimizer)  # [backbone_lr, head_lr]
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, EPOCHS - WARMUP_EPOCHS), eta_min=ETA_MIN
    )

    def step_scheduler(epoch_idx: int):
        if epoch_idx < WARMUP_EPOCHS:
            linear_warmup_step(optimizer, base_lrs, epoch_idx, WARMUP_EPOCHS)
        else:
            cosine.step()

    # ---------------------------
    # Training loop
    # ---------------------------
    best_val_mae = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_mae = 0.0
        epoch_ce = 0.0
        epoch_loss = 0.0
        epoch_acc = 0.0
        seen = 0

        t0 = time.time()
        model.reset_all_memory()

        # Unfreeze backbone's last block right after warmup (once)
        if epoch == WARMUP_EPOCHS + 1:
            unfreeze_backbone_last_block(model)
            # re-build optimizer to include newly trainable params (keep current lrs)
            optimizer = build_optimizer(model)
            base_lrs = get_base_lrs(optimizer)
            # resume cosine starting where we left off (epoch-1 warmup steps already done)
            cosine = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, EPOCHS - WARMUP_EPOCHS), eta_min=ETA_MIN
            )

        for it, (frames, meta) in enumerate(train_loader, start=1):
            frames = frames.to(device)  # [B, T, 3, H, W]
            labels = meta["phase_label"].to(device).long()
            ttnp = torch.clamp(
                meta["time_to_next_phase"].to(device).float(), 0.0, TIME_HORIZON
            )

            reg, logits = model(frames, meta)

            phase_loss = ce_loss(logits, labels)
            anticipation_loss = smooth_l1(reg, ttnp)
            train_loss = anticipation_loss + phase_loss

            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
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
                lrs = [pg["lr"] for pg in optimizer.param_groups]
                print(
                    f"[Epoch {epoch:02d} | it {it:04d}] "
                    f"loss={train_loss.item():.4f} | mae={train_mae.item():.4f} | "
                    f"acc={train_acc.item():.4f} | reg={anticipation_loss.item():.4f} | "
                    f"cls={phase_loss.item():.4f} | lrs={lrs}"
                )

        # Step scheduler per-epoch
        step_scheduler(epoch - 1)

        # Epoch stats
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
            print(f"✅  New best val_mae={best_val_mae:.4f} — saved to: {CKPT_PATH}")

    # ---------------------------
    # Test + Visualizations (unchanged)
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
