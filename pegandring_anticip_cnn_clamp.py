#!/usr/bin/env python3
# train_future_anticipation_transformer.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import random
from pathlib import Path
from typing import Dict, Any, DefaultDict, Optional, List, Tuple, Literal
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from datasets.peg_and_ring_workflow import PegAndRing, VideoBatchSampler
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


class TemporalConvBlock(nn.Module):
    """
    Temporal convolutional block with residual connection and causal padding.
    Uses dilated convolutions for different receptive fields.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.use_residual = use_residual and (in_channels == out_channels)
        
        # Causal padding - only look at past and present
        self.pad = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0  # We'll handle padding manually for causality
        )
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=1  # Pointwise conv
        )
        
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Residual connection projection if needed
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T]
        """
        residual = x
        
        # Causal padding - pad only the left side
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))
        
        # First conv with dilation
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Pointwise conv
        x = self.conv2(x)
        x = self.norm2(x)
        
        # Residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            x = x + residual
        
        x = self.activation(x)
        return x


class MultiScaleTemporalCNN(nn.Module):
    """
    Multi-scale temporal CNN with different dilation rates to capture
    both short-term and long-term temporal dependencies.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Different dilation rates for multi-scale modeling
        dilations = [1, 2, 4, 8, 16][:num_layers]
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(
            TemporalConvBlock(
                in_channels, hidden_channels, 
                kernel_size=3, dilation=dilations[0], 
                dropout=dropout, use_residual=False
            )
        )
        
        # Subsequent layers
        for i in range(1, num_layers):
            self.layers.append(
                TemporalConvBlock(
                    hidden_channels, hidden_channels,
                    kernel_size=3, dilation=dilations[i],
                    dropout=dropout, use_residual=True
                )
            )
        
        # Final projection
        self.final_proj = nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.final_norm = nn.BatchNorm1d(hidden_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T]
        """
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_proj(x)
        x = self.final_norm(x)
        x = F.gelu(x)
        
        return x


class SpatialAttentionPool(nn.Module):
    """
    Spatial attention pooling for CNN features.
    Instead of simple average pooling, learn what spatial regions are important.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        
        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] -> [B, C]
        """
        # Compute spatial attention weights
        att_weights = self.spatial_att(x)  # [B, 1, H, W]
        
        # Apply attention
        x_attended = x * att_weights  # [B, C, H, W]
        
        # Global pooling
        x_pooled = self.global_pool(x_attended)  # [B, C, 1, 1]
        x_pooled = x_pooled.flatten(1)  # [B, C]
        
        return x_pooled


class TemporalCNNAnticipation(nn.Module):
    """
    Non-transformer approach using:
    1. Visual backbone with spatial attention pooling
    2. Multi-scale temporal CNN for sequence modeling
    3. Specialized heads for phase classification and anticipation
    
    Key advantages:
    - Faster than transformers (especially for longer sequences)
    - Naturally causal (no future information leakage)
    - Good for 1fps videos with clear temporal structure
    - Easier to interpret and debug
    """
    
    def __init__(
        self,
        sequence_length: int,
        num_classes: int = 6,
        time_horizon: float = 2.0,
        *,
        backbone: str = "resnet18",
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        hidden_channels: int = 256,
        num_temporal_layers: int = 5,
        dropout: float = 0.1,
        use_spatial_attention: bool = True,
    ):
        super().__init__()
        
        self.T = sequence_length
        self.C = num_classes
        self.H = time_horizon
        self.hidden_channels = hidden_channels
        
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
        
        # Remove the final FC layer and average pooling
        self.backbone_features = nn.Sequential(*list(bb.children())[:-2])  # Remove avgpool and fc
        
        if freeze_backbone:
            for p in self.backbone_features.parameters():
                p.requires_grad = False
        
        # ---- Spatial pooling ----
        if use_spatial_attention:
            self.spatial_pool = SpatialAttentionPool(feat_dim)
        else:
            self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        
        # ---- Feature projection ----
        self.feature_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ---- Temporal CNN ----
        self.temporal_cnn = MultiScaleTemporalCNN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_temporal_layers,
            dropout=dropout
        )
        
        # ---- Phase classification head ----
        # Use only the last timestep for current phase
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
        
        # ---- Anticipation head ----
        # Use all timesteps with attention pooling
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.anticipation_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
            nn.Softplus(beta=1.0)  # Ensure positive outputs
        )
        
        # ---- Learnable query for anticipation ----
        self.anticipation_query = nn.Parameter(
            torch.randn(1, 1, hidden_channels) * 0.02
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly"""
        for module in [self.feature_proj, self.phase_head, self.anticipation_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
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
        x = frames.view(B * T, C_in, H, W)  # [B*T, 3, H, W]
        
        with torch.set_grad_enabled(self.backbone_features.training):
            # Get spatial feature maps
            spatial_features = self.backbone_features(x)  # [B*T, feat_dim, h, w]
        
        # Apply spatial pooling
        if hasattr(self.spatial_pool, 'forward'):
            visual_features = self.spatial_pool(spatial_features)  # [B*T, feat_dim]
        else:
            visual_features = self.spatial_pool(spatial_features).flatten(1)  # [B*T, feat_dim]
        
        # Reshape back to sequence
        visual_features = visual_features.view(B, T, -1)  # [B, T, feat_dim]
        
        # Project to hidden space
        features = []
        for t in range(T):
            feat_t = self.feature_proj(visual_features[:, t])  # [B, hidden_channels]
            features.append(feat_t)
        
        features = torch.stack(features, dim=2)  # [B, hidden_channels, T]
        
        # ---- Temporal modeling ----
        temporal_features = self.temporal_cnn(features)  # [B, hidden_channels, T]
        
        # Convert back to [B, T, hidden_channels] for further processing
        temporal_features_seq = temporal_features.transpose(1, 2)  # [B, T, hidden_channels]
        
        # ---- Phase classification (current state) ----
        current_features = temporal_features_seq[:, -1]  # [B, hidden_channels] - last timestep
        phase_logits = self.phase_head(current_features)  # [B, num_classes]
        
        # ---- Anticipation prediction ----
        # Use attention pooling over the entire sequence
        query = self.anticipation_query.expand(B, -1, -1)  # [B, 1, hidden_channels]
        
        attended_features, _ = self.temporal_attention(
            query=query,
            key=temporal_features_seq,
            value=temporal_features_seq
        )  # [B, 1, hidden_channels]
        
        attended_features = attended_features.squeeze(1)  # [B, hidden_channels]
        anticipation_raw = self.anticipation_head(attended_features)  # [B, num_classes]
        
        # Apply constraints
        anticipation = torch.clamp(anticipation_raw, 0.0, self.H)
        
        # Row-min constraint (at least one phase at time 0)
        anticipation = anticipation - anticipation.min(dim=1, keepdim=True)[0]
        anticipation = torch.clamp(anticipation, 0.0, self.H)
        
        if return_aux:
            aux = {
                'current_features': current_features.detach(),
                'attended_features': attended_features.detach(),
                'temporal_features_mean': temporal_features_seq.detach().mean(dim=1),
                'anticipation_raw': anticipation_raw.detach(),
            }
            return anticipation, phase_logits, aux
        
        return anticipation, phase_logits
    
    def reset_all_memory(self):
        """Dummy method for compatibility - this model is stateless"""
        pass


# Compatibility wrapper
class WindowMemoryTransformer(TemporalCNNAnticipation):
    """
    Compatibility wrapper to maintain the same interface as the original model.
    """
    
    def __init__(self, *args, **kwargs):
        # Remove unused parameters from original model
        unused_params = [
            'tcn_stages', 'tcn_dilations', 'reg_activation', 'prior_mode',
            'num_decoder_layers', 'd_model', 'nhead', 'num_encoder_layers',
            'dim_feedforward', 'use_layer_norm'
        ]
        
        for param in unused_params:
            kwargs.pop(param, None)
        
        super().__init__(*args, **kwargs)

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
BATCH_SIZE = 64
NUM_WORKERS = 30
EPOCHS = 20

# Optimizer
LR = 3e-4
WEIGHT_DECAY = 2e-4

# Model hyper
TIME_HORIZON = 1.0  # clamp targets/preds for stability in plots/metrics

PRINT_EVERY = 20
CKPT_PATH = Path("peg_and_ring_cnn_clamp.pth")

# Eval output dirs
EVAL_ROOT = Path("results/eval_outputs_cnn_clamp")
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
        backbone="resnet18",              # Start light
        pretrained_backbone=True,         
        freeze_backbone=False,            
        hidden_channels=256,             # Main hidden dimension
        num_temporal_layers=5,           # 5 dilated conv layers
        dropout=0.1,
        use_spatial_attention=True,      # Learn important spatial regions
    ).to(device)
    # model.reset_all_memory = lambda: None  # dummy

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
                    f"✅  New best val_mae={best_val_mae:.4f} val_acc={val_stats['acc']:.4f} — saved to: {CKPT_PATH}"
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
