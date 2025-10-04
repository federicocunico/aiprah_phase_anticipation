#!/usr/bin/env python3
# train_temporal_transformer_plain.py
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

from datasets.peg_and_ring_workflow import PegAndRing, VideoBatchSampler  # your dataset
from typing import Tuple, Dict, Any, Optional, Literal, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- backbone (torchvision) ----
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import math

# ---- backbone (torchvision) ----
try:
    import torchvision
    from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
    _HAS_TORCHVISION_WEIGHTS_ENUM = True
except Exception:
    import torchvision  # type: ignore
    _HAS_TORCHVISION_WEIGHTS_ENUM = False


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for temporal sequences."""
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [seq_len, batch_size, d_model]"""
        return x + self.pe[:x.size(0), :]


class WindowMemoryTransformer(nn.Module):
    """
    Simplified transformer architecture focused on anticipation:
    1. Visual backbone per frame
    2. Simple projection to embedding space
    3. Positional encoding
    4. Transformer encoder for temporal modeling
    5. Simple MLP heads for phase classification and anticipation
    
    Key simplifications:
    - No MS-TCN (transformer should handle temporal relationships)
    - No decoder with class queries (direct prediction from sequence representation)
    - Simpler heads
    - Focus on what matters: good temporal modeling and anticipation
    """
    
    def __init__(
        self,
        sequence_length: int,
        num_classes: int = 6,
        time_horizon: float = 2.0,
        *,
        backbone: str = "resnet18",  # Start with lighter backbone
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,  # Don't freeze - let it adapt
        d_model: int = 256,  # Smaller model
        nhead: int = 8,
        num_encoder_layers: int = 6,  # Focus on good temporal modeling
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        self.T = sequence_length
        self.C = num_classes
        self.H = time_horizon
        self.d_model = d_model
        
        # ---- Visual Backbone ----
        if backbone == "resnet18":
            if _HAS_TORCHVISION_WEIGHTS_ENUM:
                bb = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained_backbone else None)
            else:
                bb = torchvision.models.resnet18(pretrained=pretrained_backbone)
            feat_dim = 512
        elif backbone == "resnet50":
            if _HAS_TORCHVISION_WEIGHTS_ENUM:
                bb = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained_backbone else None)
            else:
                bb = torchvision.models.resnet50(pretrained=pretrained_backbone)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Replace final FC with identity
        bb.fc = nn.Identity()
        self.backbone = bb
        
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        # ---- Feature projection ----
        self.feature_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ---- Positional encoding ----
        self.pos_encoding = PositionalEncoding(d_model, max_len=sequence_length)
        
        # ---- Transformer encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False,  # [seq, batch, features]
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # ---- Output heads ----
        # Use the last timestep for current phase classification
        self.phase_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Use all timesteps for anticipation - predict time to each phase
        self.anticipation_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
            nn.Softplus(beta=1.0)  # Ensure positive predictions
        )
        
        # ---- Global pooling for anticipation ----
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly"""
        for m in [self.feature_proj, self.phase_head, self.anticipation_head]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(
        self, 
        frames: torch.Tensor, 
        meta: Optional[Dict[str, Any]] = None, 
        return_aux: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            frames: [B, T, 3, H, W] - video frames
            meta: metadata dict (unused but kept for API compatibility)
            return_aux: whether to return auxiliary outputs
            
        Returns:
            reg: [B, C] - time to next phase for each class (anticipation)
            logits: [B, C] - phase classification logits
            aux: (optional) auxiliary outputs for debugging
        """
        B, T, C_in, H, W = frames.shape
        assert T == self.T, f"Expected sequence length {self.T}, got {T}"
        
        # ---- Extract visual features ----
        # Process all frames through backbone
        x = frames.view(B * T, C_in, H, W)  # [B*T, 3, H, W]
        
        with torch.set_grad_enabled(self.backbone.training):
            visual_features = self.backbone(x)  # [B*T, feat_dim]
        
        # Project to embedding space
        visual_features = visual_features.view(B, T, -1)  # [B, T, feat_dim]
        embeddings = self.feature_proj(visual_features)  # [B, T, d_model]
        
        # ---- Temporal modeling with transformer ----
        # Prepare for transformer: [T, B, d_model]
        embeddings = embeddings.transpose(0, 1)  # [T, B, d_model]
        
        # Add positional encoding
        embeddings = self.pos_encoding(embeddings)  # [T, B, d_model]
        
        # Apply transformer
        # For 1fps videos, we want to model long-range dependencies
        temporal_features = self.transformer(embeddings)  # [T, B, d_model]
        
        # Back to [B, T, d_model]
        temporal_features = temporal_features.transpose(0, 1)  # [B, T, d_model]
        
        # ---- Phase classification ----
        # Use the last timestep for current phase classification
        current_repr = temporal_features[:, -1, :]  # [B, d_model]
        phase_logits = self.phase_head(current_repr)  # [B, num_classes]
        
        # ---- Anticipation prediction ----
        # Use global average pooling over time for anticipation
        # This gives us a representation of the entire sequence context
        pooled_features = temporal_features.mean(dim=1)  # [B, d_model]
        anticipation_raw = self.anticipation_head(pooled_features)  # [B, num_classes]
        
        # Clamp anticipation to valid range
        anticipation = torch.clamp(anticipation_raw, 0.0, self.H)
        
        # Apply row-min constraint (at least one phase should be at 0)
        # This encourages the model to predict that we're currently in one phase
        anticipation = anticipation - anticipation.min(dim=1, keepdim=True)[0]
        anticipation = torch.clamp(anticipation, 0.0, self.H)
        
        if return_aux:
            aux = {
                'current_repr': current_repr.detach(),
                'pooled_features': pooled_features.detach(),
                'temporal_features_mean': temporal_features.detach().mean(dim=1),
                'anticipation_raw': anticipation_raw.detach(),
            }
            return anticipation, phase_logits, aux
        
        return anticipation, phase_logits


# ---------------------------
# Fixed training hyperparams
# ---------------------------
SEED = 42
ROOT_DIR = "data/peg_and_ring_workflow"
TIME_UNIT = "minutes"  # dataset outputs in minutes (or "seconds")
NUM_CLASSES = 6

# Temporal windowing
SEQ_LEN = 16   # MAX=352/BATCHSIZE
STRIDE = 1 # SEQ_LEN//2

# Batching / epochs
BATCH_SIZE = 22
NUM_WORKERS = 30
EPOCHS = 20

# Optimizer
LR = 3e-4
WEIGHT_DECAY = 2e-4

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
    # train_batch_sampler_fixed = VideoBatchSampler(
    #     train_ds,
    #     batch_size=BATCH_SIZE,
    #     drop_last=False,
    #     shuffle_videos=True,
    #     generator=gen_train,
    #     batch_videos=False,  # fixed-size batches within a video
    # )
    # train_loader = DataLoader(
    #     train_ds,
    #     batch_sampler=train_batch_sampler_fixed,
    #     num_workers=NUM_WORKERS,
    #     pin_memory=False,
    # )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
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
        backbone="resnet18",              # Start lighter - easier to train
        pretrained_backbone=True,         # Good starting point
        freeze_backbone=False,            # Let it adapt to your surgical videos
        d_model=384,                      # Bit larger for better representation
        nhead=8,                         # More heads for better attention
        num_encoder_layers=4,            # Deeper for temporal modeling at 1fps
        dim_feedforward=1536,            # 4x d_model (standard ratio)
        dropout=0.15,                    # Slightly higher for regularization
        use_layer_norm=True,             # Better training stability
    ).to(device)
    model.reset_all_memory = lambda: None  # dummy

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
