#!/usr/bin/env python3
# train_future_anticipation_transformer.py
import gc
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import random
from pathlib import Path
from typing import Dict, Any, DefaultDict, Optional, List, Tuple, Set
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from datasets.peg_and_ring_workflow import PegAndRing, VideoBatchSampler
import torch.nn.functional as F

# ---- backbone (torchvision) ----
try:
    import torchvision
    from torchvision.models import (
        resnet18,
        resnet50,
        ResNet18_Weights,
        ResNet50_Weights,
    )

    _HAS_TORCHVISION_WEIGHTS_ENUM = True
except Exception:
    import torchvision  # type: ignore

    _HAS_TORCHVISION_WEIGHTS_ENUM = False


# -----------------------------
# Building blocks
# -----------------------------
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
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)
        self.pad = (kernel_size - 1) * dilation  # causal left padding

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # manual pad for causality
        )
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T]
        """
        residual = x
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))  # (left, right)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)

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
        dropout: float = 0.1,
    ):
        super().__init__()

        dilations = [2**i for i in range(num_layers)]  # [1,2,4,8,...]

        self.layers = nn.ModuleList()
        # First layer
        self.layers.append(
            TemporalConvBlock(
                in_channels,
                hidden_channels,
                kernel_size=3,
                dilation=dilations[0],
                dropout=dropout,
                use_residual=False,
            )
        )
        # Subsequent layers
        for i in range(1, num_layers):
            self.layers.append(
                TemporalConvBlock(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    dilation=dilations[i],
                    dropout=dropout,
                    use_residual=True,
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
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = self.spatial_att(x)  # [B,1,H,W]
        x = x * att
        x = self.global_pool(x).flatten(1)  # [B,C]
        return x


class GatedFusion(nn.Module):
    """
    Gated lateral fusion for two temporal feature streams aligned in time.
    out = σ(Conv1d([A;B])) * A + (1-σ(...)) * B
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Conv1d(2 * channels, channels, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)

        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a, b], dim=1)  # [B,2C,T]
        g = torch.sigmoid(self.gate(x))  # [B,C,T]
        y = g * a + (1.0 - g) * b
        y = self.proj(y)
        y = self.norm(y)
        return F.gelu(y)


# -----------------------------
# Causal Conformer-style temporal encoder
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        attn_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(1)
        residual = x
        x = self.ln(x)
        out, _ = self.mha(x, x, x, attn_mask=attn_mask, need_weights=False)
        out = self.dropout(out)
        return residual + out


class ConvModule(nn.Module):
    """
    Conformer convolution module (temporal).
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.dw = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
            dilation=dilation,
            padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.ln = nn.LayerNorm(d_model)

        nn.init.xavier_uniform_(self.pw1.weight)
        nn.init.zeros_(self.pw1.bias)
        nn.init.xavier_uniform_(self.dw.weight)
        nn.init.xavier_uniform_(self.pw2.weight)
        nn.init.zeros_(self.pw2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x)  # [B,T,D]
        x = x.transpose(1, 2)  # [B,D,T]

        x = self.pw1(x)  # [B,2D,T]
        a, b = x.chunk(2, dim=1)
        x = a * torch.sigmoid(b)  # [B,D,T]

        pad = (self.kernel_size - 1) * self.dilation
        if pad > 0:
            x = F.pad(x, (pad, 0))
        x = self.dw(x)
        x = self.bn(x)
        x = F.gelu(x)
        x = self.pw2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # [B,T,D]
        return residual + x


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TemporalConformerBlock(nn.Module):
    """
    FFN -> Causal Self-Attention -> Conv Module -> FFN (with residuals)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ffn1 = PositionwiseFFN(d_model, expansion=4, dropout=dropout)
        self.attn = CausalSelfAttention(d_model, num_heads=num_heads, dropout=dropout)
        self.conv = ConvModule(
            d_model, kernel_size=kernel_size, dilation=dilation, dropout=dropout
        )
        self.ffn2 = PositionwiseFFN(d_model, expansion=4, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn1(x)
        x = self.attn(x)
        x = self.conv(x)
        x = self.ffn2(x)
        return x


class TemporalConformerEncoder(nn.Module):
    """
    Stack of TemporalConformerBlock with increasing dilations for multi-scale context.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        dilations = [2**i for i in range(num_layers)]
        self.blocks = nn.ModuleList(
            [
                TemporalConformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    kernel_size=3,
                    dilation=d,
                    dropout=dropout,
                )
                for d in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# -----------------------------
# Slow-Fast Anticipation Model (differentiable constraints)
# -----------------------------
class SlowFastTemporalAnticipation(nn.Module):
    """
    Slow–Fast temporal anticipation model with completion head.

    Returns: anticipation [B,C], phase_logits [B,C], completion [B,1], {}
    """

    def __init__(
        self,
        sequence_length: int,
        num_classes: int = 6,
        time_horizon: float = 2.0,
        *,
        backbone_fast: str = "resnet18",
        backbone_slow: str = "resnet18",
        pretrained_backbone_fast: bool = True,
        pretrained_backbone_slow: bool = True,
        freeze_backbone_fast: bool = False,
        freeze_backbone_slow: bool = False,
        hidden_channels: int = 256,
        num_temporal_layers: int = 5,
        dropout: float = 0.1,
        use_spatial_attention: bool = True,
        attn_heads: int = 8,
        softmin_tau: float | None = None,
        sigmoid_scale: float = 1.0,
        floor_beta: float = 2.0,
    ):
        super().__init__()

        self.T = sequence_length
        self.C = num_classes
        self.H = time_horizon
        self.hidden_channels = hidden_channels

        self.softmin_tau = softmin_tau if softmin_tau is not None else 0.02 * self.H
        self.sigmoid_scale = sigmoid_scale
        self.floor_beta = floor_beta

        # ---- Visual backbones (independent) ----
        self.backbone_fast_features, feat_dim_fast = self._make_backbone(
            backbone_fast, pretrained_backbone_fast, freeze_backbone_fast
        )
        self.backbone_slow_features, feat_dim_slow = self._make_backbone(
            backbone_slow, pretrained_backbone_slow, freeze_backbone_slow
        )

        # ---- Spatial pooling ----
        if use_spatial_attention:
            self.spatial_pool_fast = SpatialAttentionPool(feat_dim_fast)
            self.spatial_pool_slow = SpatialAttentionPool(feat_dim_slow)
        else:
            self.spatial_pool_fast = nn.AdaptiveAvgPool2d(1)
            self.spatial_pool_slow = nn.AdaptiveAvgPool2d(1)

        # ---- Feature projections (per pathway) ----
        self.feature_proj_fast = nn.Sequential(
            nn.Linear(feat_dim_fast, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.feature_proj_slow = nn.Sequential(
            nn.Linear(feat_dim_slow, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- Pathway-specific temporal CNNs ----
        self.temporal_cnn_fast = MultiScaleTemporalCNN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_temporal_layers,
            dropout=dropout,
        )
        self.temporal_cnn_slow = MultiScaleTemporalCNN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_temporal_layers,
            dropout=dropout,
        )

        # ---- Gated fusion on the fast timeline ----
        self.fusion_gate_fast = GatedFusion(hidden_channels)

        # ---- Post-merge causal Conformer encoder ----
        self.temporal_encoder = TemporalConformerEncoder(
            d_model=hidden_channels,
            num_layers=num_temporal_layers,
            num_heads=attn_heads,
            dropout=dropout,
        )

        # ---- Heads ----
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

        # Anticipation via attention pooling
        self.anticipation_query = nn.Parameter(
            torch.randn(1, 1, hidden_channels) * 0.02
        )
        self.temporal_attention_pool = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.anticipation_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

        self.completion_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    # ----- utils -----
    def _make_backbone(self, backbone: str, pretrained: bool, freeze: bool):
        if backbone == "resnet18":
            if _HAS_TORCHVISION_WEIGHTS_ENUM:
                bb = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            else:
                bb = torchvision.models.resnet18(pretrained=pretrained)
            feat_dim = 512
        elif backbone == "resnet50":
            if _HAS_TORCHVISION_WEIGHTS_ENUM:
                bb = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            else:
                bb = torchvision.models.resnet50(pretrained=pretrained)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        features = nn.Sequential(*list(bb.children())[:-2])
        if freeze:
            for p in features.parameters():
                p.requires_grad = False
        return features, feat_dim

    def _init_weights(self):
        for module in [
            self.feature_proj_fast,
            self.feature_proj_slow,
            self.phase_head,
            self.anticipation_head,
            self.completion_head,
        ]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _pool_spatial(self, pool: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if isinstance(pool, SpatialAttentionPool):
            return pool(x)
        return pool(x).flatten(1)

    @staticmethod
    def _softmin(x: torch.Tensor, dim: int, tau: float) -> torch.Tensor:
        return -tau * torch.logsumexp(-x / tau, dim=dim, keepdim=True)

    # ----- forward -----
    def forward(
        self,
        frames: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, C_in, H, W = frames.shape
        assert T == self.T, f"Expected sequence length {self.T}, got {T}"

        # Pathway sampling
        frames_fast = frames
        idx_slow = torch.arange(0, T, 2, device=frames.device)
        frames_slow = frames.index_select(1, idx_slow)
        T_slow = frames_slow.shape[1]

        # Fast features
        xf = frames_fast.view(B * T, C_in, H, W)
        with torch.set_grad_enabled(self.backbone_fast_features.training):
            sf = self.backbone_fast_features(xf)
        vf = self._pool_spatial(self.spatial_pool_fast, sf)
        vf_seq = vf.view(B, T, -1)
        feats_fast = torch.stack(
            [self.feature_proj_fast(vf_seq[:, t]) for t in range(T)], dim=2
        )

        # Slow features
        xs = frames_slow.view(B * T_slow, C_in, H, W)
        with torch.set_grad_enabled(self.backbone_slow_features.training):
            ss = self.backbone_slow_features(xs)
        vs = self._pool_spatial(self.spatial_pool_slow, ss)
        vs_seq = vs.view(B, T_slow, -1)
        feats_slow = torch.stack(
            [self.feature_proj_slow(vs_seq[:, t]) for t in range(T_slow)], dim=2
        )

        # Temporal TCNs
        tf = self.temporal_cnn_fast(feats_fast)
        ts = self.temporal_cnn_slow(feats_slow)

        # Align + fuse
        ts_up = F.interpolate(ts, size=T, mode="linear", align_corners=False)
        fused_fast = self.fusion_gate_fast(tf, ts_up)

        # Conformer temporal encoder
        encoded = self.temporal_encoder(fused_fast.transpose(1, 2))

        # Heads
        current = encoded[:, -1, :]
        phase_logits = self.phase_head(current)
        completion = self.completion_head(current)

        # Anticipation via attention pooling
        query = self.anticipation_query.expand(B, -1, -1)
        pooled, _ = self.temporal_attention_pool(query=query, key=encoded, value=encoded)
        pooled = pooled.squeeze(1)
        raw = self.anticipation_head(pooled)

        if self.sigmoid_scale != 1.0:
            y = self.H * torch.sigmoid(self.sigmoid_scale * raw)
        else:
            y = self.H * torch.sigmoid(raw)

        m = self._softmin(y, dim=1, tau=self.softmin_tau)
        y = y - m
        anticipation = F.softplus(y, beta=self.floor_beta)

        return anticipation, phase_logits, completion, {}


# ---------------------------
# Fixed training hyperparams
# ---------------------------
SEED = 42
ROOT_DIR = "data/peg_and_ring_workflow"
TIME_UNIT = "minutes"
NUM_CLASSES = 6

SEQ_LEN = 16
STRIDE = 1

BATCH_SIZE = 12
NUM_WORKERS = 30
EPOCHS = 50

LR = 1e-4
WEIGHT_DECAY = 2e-4

TIME_HORIZON = 2.0

PRINT_EVERY = 40
CKPT_PATH = Path("peg_and_ring_slowfast_transformer.pth")
LAST_CKPT_PATH = Path("peg_and_ring_slowfast_transformer_last.pth")  # resume point

# control flags
DO_TRAIN = False
RESUME_IF_CRASH = True

EVAL_ROOT = Path("results/eval_outputs_slowfast")
EVAL_CLS_DIR = EVAL_ROOT / "classification"
EVAL_ANT_DIR = EVAL_ROOT / "anticipation"
EVAL_CLS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_ANT_DIR.mkdir(parents=True, exist_ok=True)

# two-train-videos fit visualizations
EVAL_TRAIN_FIT_ROOT = EVAL_ROOT / "train_fit"
EVAL_TRAIN_CLS_TWO_DIR = EVAL_TRAIN_FIT_ROOT / "classification_two"
EVAL_TRAIN_ANT_TWO_DIR = EVAL_TRAIN_FIT_ROOT / "anticipation_two"
EVAL_TRAIN_CLS_TWO_DIR.mkdir(parents=True, exist_ok=True)
EVAL_TRAIN_ANT_TWO_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ---------------------------
# Evaluation (unchanged)
# ---------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, time_horizon: float
) -> Dict[str, Any]:
    model.eval()

    total_mae = 0.0
    total_ce = 0.0
    total_acc = 0.0
    total_cmae = 0.0
    total_samples = 0

    ce_loss = nn.CrossEntropyLoss()

    for _, (frames, meta) in enumerate(loader):
        frames = frames.to(device)
        ttnp = meta["time_to_next_phase"].to(device).float()
        ttnp = torch.clamp(ttnp, min=0.0, max=time_horizon)
        labels = meta["phase_label"].to(device).long()
        completion_gt = meta["phase_completition"].to(device).float().unsqueeze(1)

        reg, logits, completion_pred, _ = model(frames, meta)

        preds_cls = logits.argmax(dim=1)
        acc = (preds_cls == labels).float().mean()
        mae = torch.mean(torch.abs(reg - ttnp))
        ce = ce_loss(logits, labels)
        cmae = torch.mean(torch.abs(completion_pred - completion_gt))

        bs = frames.size(0)
        total_mae += float(mae.item()) * bs
        total_ce += float(ce.item()) * bs
        total_acc += float(acc.item()) * bs
        total_cmae += float(cmae.item()) * bs
        total_samples += bs

    return {
        "mae": total_mae / max(1, total_samples),
        "ce": total_ce / max(1, total_samples),
        "acc": total_acc / max(1, total_samples),
        "compl_mae": total_cmae / max(1, total_samples),
        "samples": total_samples,
    }


# ---------------------------
# Visualizations (added optional filter earlier)
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
    video_filter: Optional[Set[str]] = None,
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

    all_keys = list(ds.ant_cache.keys())
    keys = [vn for vn in all_keys if (video_filter is None or vn in video_filter)]

    lengths: Dict[str, int] = {vn: ds.ant_cache[vn].shape[0] for vn in keys}
    preds: DefaultDict[str, np.ndarray] = defaultdict(lambda: None)
    gts: DefaultDict[str, np.ndarray] = defaultdict(lambda: None)
    for vn, N in lengths.items():
        preds[vn] = -np.ones(N, dtype=np.int32)
        gts[vn] = -np.ones(N, dtype=np.int32)

    for frames, meta in loader:
        video_names = meta["video_name"]
        if video_filter is not None and not any(vn in video_filter for vn in video_names):
            continue

        frames = frames.to(device)
        _, logits, _, _ = model(frames, meta)

        pred_phase = logits.argmax(dim=1).detach().cpu().numpy()
        idx_last = meta["frames_indexes"][:, -1].cpu().numpy()
        gt_phase = meta["phase_label"].cpu().numpy()

        for vn, idx, p, g in zip(video_names, idx_last, pred_phase, gt_phase):
            if video_filter is not None and vn not in video_filter:
                continue
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
        if not np.any(valid):
            continue

        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        ax.step(x[valid], gt_arr[valid], where="post", linewidth=2, label="GT Phase", alpha=0.9)
        ax.step(x[valid], pred_arr[valid], where="post", linewidth=2, label="Pred Phase", alpha=0.9)

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
    video_filter: Optional[Set[str]] = None,
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

    all_keys = list(ds.ant_cache.keys())
    keys = [vn for vn in all_keys if (video_filter is None or vn in video_filter)]

    lengths: Dict[str, int] = {vn: ds.ant_cache[vn].shape[0] for vn in keys}
    preds: DefaultDict[str, np.ndarray] = defaultdict(lambda: None)
    gts: DefaultDict[str, np.ndarray] = defaultdict(lambda: None)
    for vn, N in lengths.items():
        preds[vn] = np.full((N, NUM_CLASSES), np.nan, dtype=np.float32)
        gts[vn] = np.full((N, NUM_CLASSES), np.nan, dtype=np.float32)

    for frames, meta in loader:
        video_names = meta["video_name"]
        if video_filter is not None and not any(vn in video_filter for vn in video_names):
            continue

        frames = frames.to(device)
        outputs, _, _, _ = model(frames, meta)

        pred = torch.clamp(outputs, min=0.0, max=time_horizon).detach().cpu().numpy()
        gt = (
            torch.clamp(meta["time_to_next_phase"], min=0.0, max=time_horizon)
            .cpu()
            .numpy()
        )

        idx_last = meta["frames_indexes"][:, -1].cpu().numpy()

        for vn, idx, p_row, g_row in zip(video_names, idx_last, pred, gt):
            if video_filter is not None and vn not in video_filter:
                continue
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
        if x.size == 0:
            continue
        arr = arr[valid]
        gt = gt[valid]

        fig, axs = plt.subplots(NUM_CLASSES, 1, sharex=True, figsize=(12, 12))
        y_upper = time_horizon * 1.05

        for i in range(NUM_CLASSES):
            axs[i].plot(x, arr[:, i], linestyle="-", linewidth=2, label="Predicted")
            axs[i].plot(x, gt[:, i], linestyle="--", linewidth=2, label="Ground Truth")
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
# Training (function) + resume-if-crash + CosineLR
# ---------------------------
def _save_last_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_val_mae: float,
    path: Path = LAST_CKPT_PATH,
):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_mae": best_val_mae,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        },
        "meta": {"epochs_total": EPOCHS, "time_horizon": TIME_HORIZON},
    }
    torch.save(ckpt, path)
    print(f"💾 Saved last checkpoint (epoch {epoch}) -> {path}")


def _try_resume(
    model: nn.Module, optimizer: optim.Optimizer, scheduler, device: torch.device
) -> Tuple[int, float]:
    """
    Returns (start_epoch, best_val_mae)
    """
    if RESUME_IF_CRASH and LAST_CKPT_PATH.exists():
        try:
            ckpt = torch.load(LAST_CKPT_PATH, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if scheduler is not None and ckpt.get("scheduler_state") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            best_mae = float(ckpt.get("best_val_mae", float("inf")))
            rng = ckpt.get("rng_state", {})
            # if "python" in rng:
            #     random.setstate(rng["python"])
            # if "numpy" in rng:
            #     np.random.set_state(rng["numpy"])
            # if "torch" in rng:
            #     torch.set_rng_state(rng["torch"])
            # if torch.cuda.is_available() and rng.get("cuda") is not None:
            #     try:
            #         torch.cuda.set_rng_state_all(rng["cuda"])
            #     except Exception:
            #         pass

            start_epoch = int(ckpt.get("epoch", 0)) + 1
            print(
                f"🔁 Resume enabled. Loaded last checkpoint at epoch {start_epoch-1} from {LAST_CKPT_PATH}"
            )
            return start_epoch, best_mae
        except Exception as e:
            print(f"⚠️  Failed to resume from {LAST_CKPT_PATH}: {e}")
    return 1, float("inf")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
):
    # Losses / Optim
    reg_loss = nn.SmoothL1Loss(reduction="mean")
    ce_loss = nn.CrossEntropyLoss()
    compl_loss = nn.SmoothL1Loss(beta=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine Annealing LR (epoch-level)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # Resume if requested
    start_epoch, best_val_mae = _try_resume(model, optimizer, scheduler, device)

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_mae = epoch_ce = epoch_loss = epoch_acc = epoch_cmae = 0.0
        seen = 0
        t0 = time.time()

        for it, (frames, meta) in enumerate(train_loader, start=1):
            frames = frames.to(device)
            labels = meta["phase_label"].to(device).long()
            complets_gt = meta["phase_completition"].to(device).float().unsqueeze(1)
            ttnp = torch.clamp(
                meta["time_to_next_phase"].to(device).float(), 0.0, TIME_HORIZON
            )

            reg, logits, completion_pred, _ = model(frames, meta)

            loss_compl = compl_loss(completion_pred, complets_gt)
            loss_cls = ce_loss(logits, labels)
            loss_reg = reg_loss(reg, ttnp)
            loss_total = loss_reg + loss_cls + loss_compl

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            optimizer.step()

            with torch.no_grad():
                pred_cls = logits.argmax(dim=1)
                train_mae = torch.mean(torch.abs(reg - ttnp))
                train_acc = (pred_cls == labels).float().mean()
                train_cmae = torch.mean(torch.abs(completion_pred - complets_gt))

            bs = frames.size(0)
            epoch_mae += float(train_mae.item()) * bs
            epoch_ce += float(loss_cls.item()) * bs
            epoch_loss += float(loss_total.item()) * bs
            epoch_acc += float(train_acc.item()) * bs
            epoch_cmae += float(train_cmae.item()) * bs
            seen += bs

            if it % PRINT_EVERY == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(
                    f"[Epoch {epoch:02d} | it {it:04d}] "
                    f"loss={loss_total.item():.4f} | mae={train_mae.item():.4f} | "
                    f"acc={train_acc.item():.4f} | reg={loss_reg.item():.4f} | "
                    f"cls={loss_cls.item():.4f} | compl={loss_compl.item():.4f} | lr={cur_lr:.2e}"
                )

        if epoch % 5 == 0:
            print(f"GPU mem: {torch.cuda.memory_allocated() / 1e9:.4f}GB")
            print(f"GPU cached: {torch.cuda.memory_reserved() / 1e9:.4f}GB")

            # collect
            gc.collect()
            torch.cuda.empty_cache()

        # step scheduler once per epoch
        scheduler.step()

        train_loss_avg = epoch_loss / max(1, seen)
        train_mae_avg = epoch_mae / max(1, seen)
        train_ce_avg = epoch_ce / max(1, seen)
        train_acc_avg = epoch_acc / max(1, seen)
        train_cmae_avg = epoch_cmae / max(1, seen)
        print(
            f"\nEpoch {epoch:02d} [{time.time()-t0:.1f}s] "
            f"| train_loss={train_loss_avg:.4f} train_mae={train_mae_avg:.4f} "
            f"train_ce={train_ce_avg:.4f} train_acc={train_acc_avg:.4f} "
            f"train_compl_mae={train_cmae_avg:.4f}"
        )

        # Validation
        val_stats = evaluate(model, val_loader, device, TIME_HORIZON)
        print(
            f"           val_mae={val_stats['mae']:.4f} val_ce={val_stats['ce']:.4f} "
            f"val_acc={val_stats['acc']:.4f} val_compl_mae={val_stats['compl_mae']:.4f} "
            f"| samples={val_stats['samples']}"
        )

        # Save best (unchanged)
        if val_stats["mae"] < best_val_mae:
            best_val_mae = val_stats["mae"]
            torch.save(model.state_dict(), CKPT_PATH)
            print(
                f"✅  New best val_mae={best_val_mae:.4f} val_acc={val_stats['acc']:.4f} — saved to: {CKPT_PATH}"
            )

        # Always save last for resume (even if not best)
        _save_last_checkpoint(model, optimizer, scheduler, epoch, best_val_mae, LAST_CKPT_PATH)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Datasets
    train_ds = PegAndRing(
        ROOT_DIR,
        mode="train",
        seq_len=SEQ_LEN,
        stride=STRIDE,
        time_unit=TIME_UNIT,
        augment=True,
    )
    val_ds = PegAndRing(
        ROOT_DIR,
        mode="val",
        seq_len=SEQ_LEN,
        stride=STRIDE,
        time_unit=TIME_UNIT,
        augment=False,
    )
    test_ds = PegAndRing(
        ROOT_DIR,
        mode="test",
        seq_len=SEQ_LEN,
        stride=STRIDE,
        time_unit=TIME_UNIT,
        augment=False,
    )

    # Dataloaders
    gen_train = torch.Generator()
    gen_train.manual_seed(SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )
    global PRINT_EVERY
    PRINT_EVERY = len(train_loader) // 5

    gen_eval = torch.Generator()
    gen_eval.manual_seed(SEED)
    val_loader = DataLoader(
        val_ds,
        batch_sampler=VideoBatchSampler(
            val_ds,
            batch_size=BATCH_SIZE,
            batch_videos=False,
            shuffle_videos=False,
            generator=gen_eval,
        ),
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_sampler=VideoBatchSampler(
            test_ds,
            batch_size=BATCH_SIZE,
            batch_videos=False,
            shuffle_videos=False,
            generator=gen_eval,
        ),
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # Model
    model = SlowFastTemporalAnticipation(
        sequence_length=SEQ_LEN,
        num_classes=NUM_CLASSES,
        time_horizon=TIME_HORIZON,
        backbone_fast="resnet50",
        backbone_slow="resnet50",
        pretrained_backbone_fast=True,
        pretrained_backbone_slow=True,
        freeze_backbone_fast=False,
        freeze_backbone_slow=False,
        hidden_channels=384,
        num_temporal_layers=6,
        dropout=0.1,
        use_spatial_attention=True,
        attn_heads=8,
    ).to(device)

    # ---- Training (now controlled by DO_TRAIN) ----
    if DO_TRAIN:
        train(model, train_loader, val_loader, device)

    # Test + Visualizations
    if CKPT_PATH.exists():
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        print(f"\nLoaded best model from: {CKPT_PATH}")
    elif RESUME_IF_CRASH and LAST_CKPT_PATH.exists():
        ckpt = torch.load(LAST_CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"\nLoaded last (resume) model from: {LAST_CKPT_PATH}")

    test_stats = evaluate(model, test_loader, device, TIME_HORIZON)
    print(
        f"\nTEST — mae={test_stats['mae']:.4f} ce={test_stats['ce']:.4f} "
        f"acc={test_stats['acc']:.4f} compl_mae={test_stats['compl_mae']:.4f} "
        f"samples={test_stats['samples']}"
    )

    # Standard test-set visualizations
    visualize_phase_timelines_classification(
        model=model,
        root_dir=ROOT_DIR,
        split="test",
        time_unit=TIME_UNIT,
        batch_size=BATCH_SIZE,
        num_workers=min(6, NUM_WORKERS),
        out_dir=EVAL_CLS_DIR,
    )

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

    # ---- Fit check on TWO training videos ----
    train_video_names = sorted(list(train_ds.ant_cache.keys()))[:2]
    video_filter = set(train_video_names)
    print(f"\n[Train-Fit] Visualizing videos: {train_video_names}")

    visualize_phase_timelines_classification(
        model=model,
        root_dir=ROOT_DIR,
        split="train",
        time_unit=TIME_UNIT,
        batch_size=BATCH_SIZE,
        num_workers=min(6, NUM_WORKERS),
        out_dir=EVAL_TRAIN_CLS_TWO_DIR,
        video_filter=video_filter,
    )
    visualize_anticipation_curves(
        model=model,
        root_dir=ROOT_DIR,
        split="train",
        time_unit=TIME_UNIT,
        time_horizon=TIME_HORIZON,
        batch_size=BATCH_SIZE,
        num_workers=min(6, NUM_WORKERS),
        out_dir=EVAL_TRAIN_ANT_TWO_DIR,
        video_filter=video_filter,
    )


if __name__ == "__main__":
    main()
