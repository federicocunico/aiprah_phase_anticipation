#!/usr/bin/env python3
"""
Multi-Task Slow-Fast Model for Peg & Ring Workflow

This model combines:
1. Phase Recognition (classification)
2. Phase Anticipation (time-to-next-phase regression)
3. Phase Completion (progress estimation)
4. Dual-Arm Triplet Prediction (verb + target for each arm)

Architecture:
- Slow-Fast dual pathway visual backbone
- Multi-scale temporal CNN for each pathway
- Gated fusion of slow and fast streams
- Causal Conformer encoder for temporal modeling
- Multiple prediction heads for each task

The triplet prediction handles partially annotated data:
- Uses masked loss to only compute loss on annotated frames
- Predicts null-verb/null-target for non-annotated or inactive states
"""

import gc
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import random
from pathlib import Path
from typing import Dict, Any, DefaultDict, Optional, List, Tuple, Set
from collections import defaultdict
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

from datasets.peg_and_ring_workflow import (
    PegAndRing,
    VideoBatchSampler,
    TRIPLET_VERBS,
    TRIPLET_TARGETS,
)

# ---- backbone (torchvision) ----
try:
    from torchvision.models import (
        resnet18,
        resnet50,
        ResNet18_Weights,
        ResNet50_Weights,
    )

    _HAS_TORCHVISION_WEIGHTS_ENUM = True
except Exception:
    import torchvision

    _HAS_TORCHVISION_WEIGHTS_ENUM = False


# =============================================================================================
# Building Blocks (from top-performing slow-fast model)
# =============================================================================================


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
        """x: [B, C, T]"""
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
        """x: [B, C, T]"""
        for layer in self.layers:
            x = layer(x)
        x = self.final_proj(x)
        x = self.final_norm(x)
        x = F.gelu(x)
        return x


class SpatialAttentionPool(nn.Module):
    """Spatial attention pooling for CNN features."""

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


# =============================================================================================
# Causal Conformer-style temporal encoder (from top-performing model)
# =============================================================================================


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
    """Conformer convolution module (temporal)."""

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
    """FFN -> Causal Self-Attention -> Conv Module -> FFN (with residuals)"""

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
    """Stack of TemporalConformerBlock with increasing dilations for multi-scale context."""

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


# =============================================================================================
# Multi-Task Slow-Fast Model with Triplet Prediction
# =============================================================================================


class MultiTaskSlowFastModel(nn.Module):
    """
    Multi-task model combining:
    1. Phase recognition (classification)
    2. Phase anticipation (regression)
    3. Phase completion (progress estimation)
    4. Dual-arm triplet prediction (verb + target for each arm)

    Uses slow-fast architecture for visual feature extraction.
    Handles partially annotated triplet data with masked loss.
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
        # Triplet parameters
        num_verbs: int = len(TRIPLET_VERBS),
        num_targets: int = len(TRIPLET_TARGETS),
        triplet_hidden_dim: int = 128,
    ):
        super().__init__()

        self.T = sequence_length
        self.C = num_classes
        self.H = time_horizon
        self.hidden_channels = hidden_channels
        self.num_verbs = num_verbs
        self.num_targets = num_targets

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

        # =============================================================================================
        # Phase Recognition, Anticipation, and Completion Heads (from top-performing model)
        # =============================================================================================

        # Phase classification head
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
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
            nn.LayerNorm(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

        # Completion head
        self.completion_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid(),
        )

        # =============================================================================================
        # Dual-Arm Triplet Prediction Heads
        # =============================================================================================

        # Separate feature projections for each arm (to specialize)
        self.left_arm_proj = nn.Sequential(
            nn.Linear(hidden_channels, triplet_hidden_dim),
            nn.LayerNorm(triplet_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.right_arm_proj = nn.Sequential(
            nn.Linear(hidden_channels, triplet_hidden_dim),
            nn.LayerNorm(triplet_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Left arm classification heads
        self.left_verb_head = self._make_verb_classifier(triplet_hidden_dim, dropout)
        self.left_target_head = self._make_target_classifier(
            triplet_hidden_dim, dropout
        )

        # Right arm classification heads
        self.right_verb_head = self._make_verb_classifier(triplet_hidden_dim, dropout)
        self.right_target_head = self._make_target_classifier(
            triplet_hidden_dim, dropout
        )

        self._init_weights()

    def _make_backbone(self, backbone: str, pretrained: bool, freeze: bool):
        """Create visual backbone."""
        if backbone == "resnet18":
            if _HAS_TORCHVISION_WEIGHTS_ENUM:
                bb = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            else:
                import torchvision

                bb = torchvision.models.resnet18(pretrained=pretrained)
            feat_dim = 512
        elif backbone == "resnet50":
            if _HAS_TORCHVISION_WEIGHTS_ENUM:
                bb = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            else:
                import torchvision

                bb = torchvision.models.resnet50(pretrained=pretrained)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        features = nn.Sequential(*list(bb.children())[:-2])
        if freeze:
            for p in features.parameters():
                p.requires_grad = False
        return features, feat_dim

    def _make_verb_classifier(self, in_dim: int, dropout: float) -> nn.Module:
        """Create verb classifier."""
        return nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, self.num_verbs),
        )

    def _make_target_classifier(self, in_dim: int, dropout: float) -> nn.Module:
        """Create target classifier (needs more capacity due to more classes)."""
        return nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, self.num_targets),
        )

    def _init_weights(self):
        """Initialize classifier weights."""
        modules_to_init = [
            self.feature_proj_fast,
            self.feature_proj_slow,
            self.phase_head,
            self.anticipation_head,
            self.completion_head,
            self.left_arm_proj,
            self.right_arm_proj,
            self.left_verb_head,
            self.left_target_head,
            self.right_verb_head,
            self.right_target_head,
        ]
        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _pool_spatial(self, pool: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial pooling."""
        if isinstance(pool, SpatialAttentionPool):
            return pool(x)
        return pool(x).flatten(1)

    @staticmethod
    def _softmin(x: torch.Tensor, dim: int, tau: float) -> torch.Tensor:
        """Differentiable softmin for anticipation."""
        return -tau * torch.logsumexp(-x / tau, dim=dim, keepdim=True)

    def forward(
        self,
        frames: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            frames: [B, T, C, H, W] input frames
            meta: metadata dictionary (optional)

        Returns:
            anticipation: [B, num_classes] time-to-next-phase predictions
            phase_logits: [B, num_classes] phase classification logits
            completion: [B, 1] phase completion in [0, 1]
            extras: dict with triplet predictions:
                - left_verb_logits: [B, num_verbs]
                - left_target_logits: [B, num_targets]
                - right_verb_logits: [B, num_verbs]
                - right_target_logits: [B, num_targets]
        """
        B, T, C_in, H, W = frames.shape
        assert T == self.T, f"Expected sequence length {self.T}, got {T}"

        # =============================================================================================
        # Slow-Fast Visual Feature Extraction
        # =============================================================================================

        # Pathway sampling
        frames_fast = frames
        idx_slow = torch.arange(0, T, 2, device=frames.device)
        frames_slow = frames.index_select(1, idx_slow)
        T_slow = frames_slow.shape[1]

        # Fast features
        xf = frames_fast.view(B * T, C_in, H, W)
        with torch.set_grad_enabled(self.backbone_fast_features.training):
            sf = self.backbone_fast_features(xf)
        vf = self._pool_spatial(self.spatial_pool_fast, sf)  # [B*T, feat_dim]
        # Process all timesteps at once to maintain batch size for BatchNorm
        vf_proj = self.feature_proj_fast(vf)  # [B*T, hidden_channels]
        feats_fast = vf_proj.view(B, T, -1).transpose(1, 2)  # [B, hidden_channels, T]

        # Slow features
        xs = frames_slow.view(B * T_slow, C_in, H, W)
        with torch.set_grad_enabled(self.backbone_slow_features.training):
            ss = self.backbone_slow_features(xs)
        vs = self._pool_spatial(self.spatial_pool_slow, ss)  # [B*T_slow, feat_dim]
        # Process all timesteps at once to maintain batch size for BatchNorm
        vs_proj = self.feature_proj_slow(vs)  # [B*T_slow, hidden_channels]
        feats_slow = vs_proj.view(B, T_slow, -1).transpose(
            1, 2
        )  # [B, hidden_channels, T_slow]

        # =============================================================================================
        # Temporal Modeling
        # =============================================================================================

        # Temporal TCNs
        tf = self.temporal_cnn_fast(feats_fast)
        ts = self.temporal_cnn_slow(feats_slow)

        # Align + fuse
        ts_up = F.interpolate(ts, size=T, mode="linear", align_corners=False)
        fused_fast = self.fusion_gate_fast(tf, ts_up)

        # Conformer temporal encoder
        encoded = self.temporal_encoder(
            fused_fast.transpose(1, 2)
        )  # [B, T, hidden_channels]

        # =============================================================================================
        # Phase Recognition, Anticipation, and Completion
        # =============================================================================================

        # Current frame features (last in sequence)
        current = encoded[:, -1, :]  # [B, hidden_channels]

        # Phase classification
        phase_logits = self.phase_head(current)

        # Phase completion
        completion = self.completion_head(current)

        # Anticipation via attention pooling
        query = self.anticipation_query.expand(B, -1, -1)
        pooled, _ = self.temporal_attention_pool(
            query=query, key=encoded, value=encoded
        )
        pooled = pooled.squeeze(1)
        raw = self.anticipation_head(pooled)

        # Apply constraints for anticipation
        if self.sigmoid_scale != 1.0:
            y = self.H * torch.sigmoid(self.sigmoid_scale * raw)
        else:
            y = self.H * torch.sigmoid(raw)

        m = self._softmin(y, dim=1, tau=self.softmin_tau)
        y = y - m
        anticipation = F.softplus(y, beta=self.floor_beta)

        # =============================================================================================
        # Dual-Arm Triplet Prediction
        # =============================================================================================

        # Arm-specific projections
        left_features = self.left_arm_proj(current)  # [B, triplet_hidden_dim]
        right_features = self.right_arm_proj(current)  # [B, triplet_hidden_dim]

        # Left arm predictions
        left_verb_logits = self.left_verb_head(left_features)
        left_target_logits = self.left_target_head(left_features)

        # Right arm predictions
        right_verb_logits = self.right_verb_head(right_features)
        right_target_logits = self.right_target_head(right_features)

        # Package triplet predictions in extras dict
        extras = {
            "left_verb_logits": left_verb_logits,
            "left_target_logits": left_target_logits,
            "right_verb_logits": right_verb_logits,
            "right_target_logits": right_target_logits,
        }

        return anticipation, phase_logits, completion, extras


# =============================================================================================
# Training Configuration
# =============================================================================================

SEED = 42
ROOT_DIR = "data/peg_and_ring_workflow"
TIME_UNIT = "minutes"
NUM_CLASSES = 6
NUM_VERBS = len(TRIPLET_VERBS)
NUM_TARGETS = len(TRIPLET_TARGETS)

SEQ_LEN = 8
STRIDE = 1

BATCH_SIZE = 64
NUM_WORKERS = 30
EPOCHS = 60

LR = 1e-4
WEIGHT_DECAY = 2e-4

TIME_HORIZON = 2.0

# Task weighting for multi-task loss
WEIGHT_ANTICIPATION = 1.0
WEIGHT_PHASE = 1.0
WEIGHT_COMPLETION = 0.5
WEIGHT_TRIPLET_VERB = 0.8
WEIGHT_TRIPLET_TARGET = 0.8

# Two-stage training configuration
TWO_STAGE_TRAINING = (
    True  # If True: stage 1 = phase/anticipation only, stage 2 = add triplets
)
STAGE1_EPOCHS = 50  # Epochs for stage 1 (phase/anticipation pre-training)
STAGE2_EPOCHS = 50  # Epochs for stage 2 (triplet fine-tuning)
FREEZE_BACKBONE_STAGE2 = True  # Freeze visual backbones in stage 2
FREEZE_TEMPORAL_STAGE2 = True  # Freeze temporal encoders in stage 2
STAGE1_LR = 1e-4  # Learning rate for stage 1
STAGE2_LR = 5e-5  # Lower learning rate for stage 2 (fine-tuning)

# Anti-overfitting measures for better validation performance
LABEL_SMOOTHING = 0.1  # Label smoothing for classification tasks
USE_WARMUP = True  # Learning rate warmup
WARMUP_EPOCHS = 5  # Number of warmup epochs
DROPOUT_RATE = 0.15  # Increased from 0.1 to reduce overfitting
USE_MIXUP = False  # Mixup augmentation (disable for temporal data)
GRADIENT_CLIP_VAL = 1.0  # Gradient clipping value

PRINT_EVERY = 40
CKPT_PATH_STAGE1 = Path("peg_and_ring_multitask_stage1_best.pth")
CKPT_PATH_STAGE2 = Path("peg_and_ring_multitask_stage2_best.pth")
LAST_CKPT_PATH_STAGE1 = Path("peg_and_ring_multitask_stage1_last.pth")
LAST_CKPT_PATH_STAGE2 = Path("peg_and_ring_multitask_stage2_last.pth")

# Legacy paths for single-stage training
CKPT_PATH = Path("peg_and_ring_multitask_slowfast_triplets_best.pth")
LAST_CKPT_PATH = Path("peg_and_ring_multitask_slowfast_triplets_last.pth")

# Control flags
DO_TRAIN = True
RESUME_IF_CRASH = True

# Logging
USE_WANDB = True  # Set to True to enable Weights & Biases logging
WANDB_PROJECT = "peg_ring_multitask"
WANDB_RUN_NAME = None  # Auto-generated if None

EVAL_ROOT = Path("results/eval_outputs_multitask")
EVAL_CLS_DIR = EVAL_ROOT / "classification"
EVAL_ANT_DIR = EVAL_ROOT / "anticipation"
EVAL_TRIPLET_DIR = EVAL_ROOT / "triplets"
EVAL_CLS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_ANT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_TRIPLET_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def init_wandb(stage: Optional[str] = None):
    """Initialize Weights & Biases logging."""
    if not USE_WANDB:
        return

    try:
        import wandb

        run_name = WANDB_RUN_NAME
        if run_name is None:
            if TWO_STAGE_TRAINING:
                run_name = f"two_stage_{stage}" if stage else "two_stage"
            else:
                run_name = "single_stage"

        config = {
            "seed": SEED,
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "epochs_total": (
                EPOCHS if not TWO_STAGE_TRAINING else STAGE1_EPOCHS + STAGE2_EPOCHS
            ),
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "time_horizon": TIME_HORIZON,
            "num_verbs": NUM_VERBS,
            "num_targets": NUM_TARGETS,
            "two_stage_training": TWO_STAGE_TRAINING,
            "weight_anticipation": WEIGHT_ANTICIPATION,
            "weight_phase": WEIGHT_PHASE,
            "weight_completion": WEIGHT_COMPLETION,
            "weight_triplet_verb": WEIGHT_TRIPLET_VERB,
            "weight_triplet_target": WEIGHT_TRIPLET_TARGET,
        }

        if TWO_STAGE_TRAINING:
            config.update(
                {
                    "stage1_epochs": STAGE1_EPOCHS,
                    "stage2_epochs": STAGE2_EPOCHS,
                    "stage1_lr": STAGE1_LR,
                    "stage2_lr": STAGE2_LR,
                    "freeze_backbone_stage2": FREEZE_BACKBONE_STAGE2,
                    "freeze_temporal_stage2": FREEZE_TEMPORAL_STAGE2,
                }
            )

        run = wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config=config,
            # resume="allow" if RESUME_IF_CRASH else False,
        )

        print(f"✅ Weights & Biases initialized: {run.name} id={run.id}")
        return run
    except ImportError:
        print("⚠️  wandb not installed. Install with: pip install wandb")
    except Exception as e:
        print(f"⚠️  Failed to initialize wandb: {e}")
    return None


# =============================================================================================
# Triplet Loss with Masking for Partial Annotations
# =============================================================================================


def compute_triplet_loss_with_masking(
    outputs: Dict[str, torch.Tensor],
    left_targets: torch.Tensor,
    right_targets: torch.Tensor,
    ce_criterion: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], int]:
    """
    Compute triplet loss with masking for partial annotations.

    Args:
        outputs: dict with left/right verb/target logits
        left_targets: [B, 3] (verb, subject, target) for left arm
        right_targets: [B, 3] (verb, subject, target) for right arm
        ce_criterion: CrossEntropy loss function

    Returns:
        left_loss: masked loss for left arm (0 if no valid samples)
        right_loss: masked loss for right arm (0 if no valid samples)
        metrics: dict with accuracy metrics
        num_valid: number of samples with valid triplet annotations
    """
    B = left_targets.shape[0]

    # Extract ground truth
    left_verb_gt = left_targets[:, 0]  # verb index
    left_target_gt = left_targets[:, 2]  # target index
    right_verb_gt = right_targets[:, 0]  # verb index
    right_target_gt = right_targets[:, 2]  # target index

    # Identify samples with valid (non-null) triplet annotations
    # null-verb index is 3, null-target index is 13 (last in each list)
    NULL_VERB_IDX = NUM_VERBS - 1  # 3
    NULL_TARGET_IDX = NUM_TARGETS - 1  # 13

    # A sample has valid triplet if EITHER arm is not completely null
    # This allows learning both active and inactive states
    left_has_action = (left_verb_gt != NULL_VERB_IDX) | (
        left_target_gt != NULL_TARGET_IDX
    )
    right_has_action = (right_verb_gt != NULL_VERB_IDX) | (
        right_target_gt != NULL_TARGET_IDX
    )
    has_valid_triplet = left_has_action | right_has_action

    num_valid = has_valid_triplet.sum().item()

    # If no valid triplets in this batch, return zero losses
    if num_valid == 0:
        device = outputs["left_verb_logits"].device
        zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
        metrics = {
            "left_verb_acc": 0.0,
            "left_target_acc": 0.0,
            "right_verb_acc": 0.0,
            "right_target_acc": 0.0,
            "left_verb_acc_active": 0.0,
            "right_verb_acc_active": 0.0,
        }
        return zero_loss, zero_loss, metrics, 0

    # Compute losses only on samples with valid triplets
    valid_idx = has_valid_triplet

    left_verb_loss = ce_criterion(
        outputs["left_verb_logits"][valid_idx], left_verb_gt[valid_idx]
    )
    left_target_loss = ce_criterion(
        outputs["left_target_logits"][valid_idx], left_target_gt[valid_idx]
    )
    right_verb_loss = ce_criterion(
        outputs["right_verb_logits"][valid_idx], right_verb_gt[valid_idx]
    )
    right_target_loss = ce_criterion(
        outputs["right_target_logits"][valid_idx], right_target_gt[valid_idx]
    )

    # Total loss per arm
    left_loss = left_verb_loss + left_target_loss
    right_loss = right_verb_loss + right_target_loss

    # Calculate accuracies (for monitoring) - only on valid samples
    with torch.no_grad():
        left_verb_pred = outputs["left_verb_logits"][valid_idx].argmax(dim=1)
        left_target_pred = outputs["left_target_logits"][valid_idx].argmax(dim=1)
        right_verb_pred = outputs["right_verb_logits"][valid_idx].argmax(dim=1)
        right_target_pred = outputs["right_target_logits"][valid_idx].argmax(dim=1)

        left_verb_gt_valid = left_verb_gt[valid_idx]
        left_target_gt_valid = left_target_gt[valid_idx]
        right_verb_gt_valid = right_verb_gt[valid_idx]
        right_target_gt_valid = right_target_gt[valid_idx]

        left_verb_acc = (left_verb_pred == left_verb_gt_valid).float().mean()
        left_target_acc = (left_target_pred == left_target_gt_valid).float().mean()
        right_verb_acc = (right_verb_pred == right_verb_gt_valid).float().mean()
        right_target_acc = (right_target_pred == right_target_gt_valid).float().mean()

        # Compute "active" accuracies (arms that are actually doing something, not null)
        left_active_valid = (left_verb_gt_valid != NULL_VERB_IDX) | (
            left_target_gt_valid != NULL_TARGET_IDX
        )
        right_active_valid = (right_verb_gt_valid != NULL_VERB_IDX) | (
            right_target_gt_valid != NULL_TARGET_IDX
        )

        if left_active_valid.any():
            left_verb_acc_active = (
                (
                    left_verb_pred[left_active_valid]
                    == left_verb_gt_valid[left_active_valid]
                )
                .float()
                .mean()
            )
        else:
            left_verb_acc_active = torch.tensor(0.0)

        if right_active_valid.any():
            right_verb_acc_active = (
                (
                    right_verb_pred[right_active_valid]
                    == right_verb_gt_valid[right_active_valid]
                )
                .float()
                .mean()
            )
        else:
            right_verb_acc_active = torch.tensor(0.0)

        metrics = {
            "left_verb_acc": float(left_verb_acc),
            "left_target_acc": float(left_target_acc),
            "right_verb_acc": float(right_verb_acc),
            "right_target_acc": float(right_target_acc),
            "left_verb_acc_active": float(left_verb_acc_active),
            "right_verb_acc_active": float(right_verb_acc_active),
        }

    return left_loss, right_loss, metrics, num_valid


# =============================================================================================
# Evaluation
# =============================================================================================


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    time_horizon: float,
    compute_triplets: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate model on all tasks.

    Args:
        model: model to evaluate
        loader: data loader
        device: device
        time_horizon: time horizon for anticipation
        compute_triplets: whether to compute triplet metrics (set False for stage 1)
    """
    model.eval()

    # Phase anticipation and recognition metrics
    total_mae = 0.0
    total_ce = 0.0
    total_acc = 0.0
    total_cmae = 0.0

    # Triplet metrics
    total_left_verb_acc = 0.0
    total_left_target_acc = 0.0
    total_right_verb_acc = 0.0
    total_right_target_acc = 0.0
    total_left_arm_acc = 0.0
    total_right_arm_acc = 0.0
    total_complete_acc = 0.0
    total_triplet_samples = 0

    # For mAP calculation: collect all predictions and ground truths
    all_left_verb_probs = []
    all_left_target_probs = []
    all_right_verb_probs = []
    all_right_target_probs = []
    all_left_verb_gt = []
    all_left_target_gt = []
    all_right_verb_gt = []
    all_right_target_gt = []

    total_samples = 0

    ce_loss = nn.CrossEntropyLoss()

    NULL_VERB_IDX = NUM_VERBS - 1
    NULL_TARGET_IDX = NUM_TARGETS - 1

    for _, (frames, meta) in enumerate(loader):
        frames = frames.to(device)

        # Phase targets
        ttnp = meta["time_to_next_phase"].to(device).float()
        ttnp = torch.clamp(ttnp, min=0.0, max=time_horizon)
        labels = meta["phase_label"].to(device).long()
        completion_gt = meta["phase_completition"].to(device).float().unsqueeze(1)

        # Forward pass
        reg, logits, completion_pred, extras = model(frames, meta)

        # Phase metrics
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

        # Triplet metrics (only if requested and data available)
        if compute_triplets:
            left_targets = meta["triplet_left_classification"].to(device).long()
            right_targets = meta["triplet_right_classification"].to(device).long()

            # Check if this batch has valid triplet annotations
            left_verb_gt = left_targets[:, 0]
            left_target_gt = left_targets[:, 2]
            right_verb_gt = right_targets[:, 0]
            right_target_gt = right_targets[:, 2]

            left_has_action = (left_verb_gt != NULL_VERB_IDX) | (
                left_target_gt != NULL_TARGET_IDX
            )
            right_has_action = (right_verb_gt != NULL_VERB_IDX) | (
                right_target_gt != NULL_TARGET_IDX
            )
            has_valid_triplet = left_has_action | right_has_action

            if has_valid_triplet.any():
                # Only compute metrics on samples with valid triplets
                valid_idx = has_valid_triplet
                num_valid = valid_idx.sum().item()

                # Get predictions
                left_verb_pred = extras["left_verb_logits"][valid_idx].argmax(dim=1)
                left_target_pred = extras["left_target_logits"][valid_idx].argmax(dim=1)
                right_verb_pred = extras["right_verb_logits"][valid_idx].argmax(dim=1)
                right_target_pred = extras["right_target_logits"][valid_idx].argmax(
                    dim=1
                )

                # Get probabilities for mAP (softmax of logits)
                left_verb_probs = torch.softmax(
                    extras["left_verb_logits"][valid_idx], dim=1
                )
                left_target_probs = torch.softmax(
                    extras["left_target_logits"][valid_idx], dim=1
                )
                right_verb_probs = torch.softmax(
                    extras["right_verb_logits"][valid_idx], dim=1
                )
                right_target_probs = torch.softmax(
                    extras["right_target_logits"][valid_idx], dim=1
                )

                # Ground truths
                left_verb_gt_v = left_verb_gt[valid_idx]
                left_target_gt_v = left_target_gt[valid_idx]
                right_verb_gt_v = right_verb_gt[valid_idx]
                right_target_gt_v = right_target_gt[valid_idx]

                # Collect for mAP calculation
                all_left_verb_probs.append(left_verb_probs.cpu().numpy())
                all_left_target_probs.append(left_target_probs.cpu().numpy())
                all_right_verb_probs.append(right_verb_probs.cpu().numpy())
                all_right_target_probs.append(right_target_probs.cpu().numpy())
                all_left_verb_gt.append(left_verb_gt_v.cpu().numpy())
                all_left_target_gt.append(left_target_gt_v.cpu().numpy())
                all_right_verb_gt.append(right_verb_gt_v.cpu().numpy())
                all_right_target_gt.append(right_target_gt_v.cpu().numpy())

                # Accuracies (for logging)
                left_verb_acc = (left_verb_pred == left_verb_gt_v).float().mean()
                left_target_acc = (left_target_pred == left_target_gt_v).float().mean()
                right_verb_acc = (right_verb_pred == right_verb_gt_v).float().mean()
                right_target_acc = (
                    (right_target_pred == right_target_gt_v).float().mean()
                )

                left_arm_acc = (
                    (
                        (left_verb_pred == left_verb_gt_v)
                        & (left_target_pred == left_target_gt_v)
                    )
                    .float()
                    .mean()
                )
                right_arm_acc = (
                    (
                        (right_verb_pred == right_verb_gt_v)
                        & (right_target_pred == right_target_gt_v)
                    )
                    .float()
                    .mean()
                )
                complete_acc = (
                    (
                        (left_verb_pred == left_verb_gt_v)
                        & (left_target_pred == left_target_gt_v)
                        & (right_verb_pred == right_verb_gt_v)
                        & (right_target_pred == right_target_gt_v)
                    )
                    .float()
                    .mean()
                )

                total_left_verb_acc += float(left_verb_acc.item()) * num_valid
                total_left_target_acc += float(left_target_acc.item()) * num_valid
                total_right_verb_acc += float(right_verb_acc.item()) * num_valid
                total_right_target_acc += float(right_target_acc.item()) * num_valid
                total_left_arm_acc += float(left_arm_acc.item()) * num_valid
                total_right_arm_acc += float(right_arm_acc.item()) * num_valid
                total_complete_acc += float(complete_acc.item()) * num_valid
                total_triplet_samples += num_valid

    N = max(1, total_samples)
    N_triplet = max(1, total_triplet_samples)

    results = {
        # Phase metrics
        "mae": total_mae / N,
        "ce": total_ce / N,
        "acc": total_acc / N,
        "compl_mae": total_cmae / N,
        "samples": total_samples,
    }

    # Add triplet metrics if computed
    if compute_triplets and total_triplet_samples > 0:
        # Calculate individual component accuracies
        left_verb_acc = total_left_verb_acc / N_triplet
        left_target_acc = total_left_target_acc / N_triplet
        right_verb_acc = total_right_verb_acc / N_triplet
        right_target_acc = total_right_target_acc / N_triplet
        left_arm_acc = total_left_arm_acc / N_triplet
        right_arm_acc = total_right_arm_acc / N_triplet
        complete_triplet_acc = total_complete_acc / N_triplet

        # Mean accuracy of verbs (mean of left and right verb accuracies)
        mean_verb_acc = (left_verb_acc + right_verb_acc) / 2.0

        # Mean accuracy of targets (mean of left and right target accuracies)
        mean_target_acc = (left_target_acc + right_target_acc) / 2.0

        # Mean accuracy of complete arms (mean of left and right complete arm accuracies)
        mean_arm_acc = (left_arm_acc + right_arm_acc) / 2.0

        # Overall mean triplet accuracy (mean of all 4 individual component accuracies)
        mean_triplet_acc = (
            left_verb_acc + left_target_acc + right_verb_acc + right_target_acc
        ) / 4.0

        # ========================================================================
        # Calculate TRUE mAP (mean Average Precision) using precision-recall
        # ========================================================================

        def calculate_multiclass_map(y_true_list, y_prob_list, num_classes):
            """
            Calculate mAP for multi-class classification using one-vs-rest approach.

            Args:
                y_true_list: list of ground truth arrays
                y_prob_list: list of probability arrays (softmax outputs)
                num_classes: number of classes

            Returns:
                mAP: mean Average Precision across all classes
            """
            if len(y_true_list) == 0:
                return 0.0

            # Concatenate all batches
            y_true = np.concatenate(y_true_list, axis=0)
            y_prob = np.concatenate(y_prob_list, axis=0)

            aps = []
            for class_idx in range(num_classes):
                # Convert to binary classification (class vs rest)
                binary_true = (y_true == class_idx).astype(int)
                class_probs = y_prob[:, class_idx]

                # Only compute AP if both positive and negative samples exist
                if len(np.unique(binary_true)) > 1:
                    try:
                        ap = average_precision_score(binary_true, class_probs)
                        aps.append(ap)
                    except:
                        pass  # Skip if AP computation fails

            return np.mean(aps) if len(aps) > 0 else 0.0

        # Compute mAP for each component
        left_verb_map = calculate_multiclass_map(
            all_left_verb_gt, all_left_verb_probs, NUM_VERBS
        )
        left_target_map = calculate_multiclass_map(
            all_left_target_gt, all_left_target_probs, NUM_TARGETS
        )
        right_verb_map = calculate_multiclass_map(
            all_right_verb_gt, all_right_verb_probs, NUM_VERBS
        )
        right_target_map = calculate_multiclass_map(
            all_right_target_gt, all_right_target_probs, NUM_TARGETS
        )

        # Overall mAPs (mean of left and right)
        mean_verb_map = (left_verb_map + right_verb_map) / 2.0
        mean_target_map = (left_target_map + right_target_map) / 2.0

        # Overall triplet mAP (mean of all 4 component mAPs)
        mean_triplet_map = (
            left_verb_map + left_target_map + right_verb_map + right_target_map
        ) / 4.0

        results.update(
            {
                # Accuracies (for logging only)
                "left_verb_acc": left_verb_acc,
                "left_target_acc": left_target_acc,
                "right_verb_acc": right_verb_acc,
                "right_target_acc": right_target_acc,
                "left_arm_acc": left_arm_acc,
                "right_arm_acc": right_arm_acc,
                "complete_triplet_acc": complete_triplet_acc,
                "mean_verb_acc": mean_verb_acc,
                "mean_target_acc": mean_target_acc,
                "mean_arm_acc": mean_arm_acc,
                "mean_triplet_acc": mean_triplet_acc,
                # mAP metrics (TRUE mean Average Precision for model selection)
                "left_verb_map": left_verb_map,
                "left_target_map": left_target_map,
                "right_verb_map": right_verb_map,
                "right_target_map": right_target_map,
                "mean_verb_map": mean_verb_map,
                "mean_target_map": mean_target_map,
                "mean_triplet_map": mean_triplet_map,  # USE THIS FOR STAGE 2 BEST MODEL SELECTION
                "triplet_samples": total_triplet_samples,
            }
        )
    elif compute_triplets:
        # No triplet data available
        results.update(
            {
                "left_verb_acc": 0.0,
                "left_target_acc": 0.0,
                "right_verb_acc": 0.0,
                "right_target_acc": 0.0,
                "left_arm_acc": 0.0,
                "right_arm_acc": 0.0,
                "complete_triplet_acc": 0.0,
                "mean_verb_acc": 0.0,
                "mean_target_acc": 0.0,
                "mean_arm_acc": 0.0,
                "mean_triplet_acc": 0.0,
                "left_verb_map": 0.0,
                "left_target_map": 0.0,
                "right_verb_map": 0.0,
                "right_target_map": 0.0,
                "mean_verb_map": 0.0,
                "mean_target_map": 0.0,
                "mean_triplet_map": 0.0,
                "triplet_samples": 0,
            }
        )

    return results


# =============================================================================================
# Training Loop
# =============================================================================================


def freeze_model_components(
    model: nn.Module, freeze_backbone: bool = True, freeze_temporal: bool = True
):
    """
    Freeze specific components of the model for stage 2 training.

    Args:
        model: the multi-task model
        freeze_backbone: whether to freeze visual backbones
        freeze_temporal: whether to freeze temporal encoders and phase/anticipation heads
    """
    if freeze_backbone:
        print("🔒 Freezing visual backbones...")
        for param in model.backbone_fast_features.parameters():
            param.requires_grad = False
        for param in model.backbone_slow_features.parameters():
            param.requires_grad = False
        for param in model.spatial_pool_fast.parameters():
            param.requires_grad = False
        for param in model.spatial_pool_slow.parameters():
            param.requires_grad = False

    if freeze_temporal:
        print("🔒 Freezing temporal encoders and phase/anticipation heads...")
        for param in model.feature_proj_fast.parameters():
            param.requires_grad = False
        for param in model.feature_proj_slow.parameters():
            param.requires_grad = False
        for param in model.temporal_cnn_fast.parameters():
            param.requires_grad = False
        for param in model.temporal_cnn_slow.parameters():
            param.requires_grad = False
        for param in model.fusion_gate_fast.parameters():
            param.requires_grad = False
        for param in model.temporal_encoder.parameters():
            param.requires_grad = False
        for param in model.phase_head.parameters():
            param.requires_grad = False
        for param in model.anticipation_head.parameters():
            param.requires_grad = False
        for param in model.anticipation_query.data:
            pass  # This is a parameter, not a module
        model.anticipation_query.requires_grad = False
        for param in model.temporal_attention_pool.parameters():
            param.requires_grad = False
        for param in model.completion_head.parameters():
            param.requires_grad = False

    # Print trainable parameters summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(
        f"📊 Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)"
    )
    print(
        f"📊 Frozen parameters: {total_params-trainable_params:,} ({100*(total_params-trainable_params)/total_params:.1f}%)"
    )


def _save_last_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_val_mae: float,
    path: Path = LAST_CKPT_PATH,
):
    """Save checkpoint for crash recovery."""
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
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        "meta": {
            "epochs_total": EPOCHS,
            "time_horizon": TIME_HORIZON,
            "num_verbs": NUM_VERBS,
            "num_targets": NUM_TARGETS,
        },
    }
    torch.save(ckpt, path)
    print(f"💾 Saved last checkpoint (epoch {epoch}) -> {path}")


def _try_resume(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    last_ckpt_path: Path = LAST_CKPT_PATH,
) -> Tuple[int, float]:
    """Try to resume from last checkpoint. Returns (start_epoch, best_val_mae)"""
    if RESUME_IF_CRASH and last_ckpt_path.exists():
        try:
            ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if scheduler is not None and ckpt.get("scheduler_state") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            best_mae = float(ckpt.get("best_val_mae", float("inf")))

            start_epoch = int(ckpt.get("epoch", 0)) + 1
            print(
                f"🔁 Resume enabled. Loaded last checkpoint at epoch {start_epoch-1} from {last_ckpt_path}"
            )
            return start_epoch, best_mae
        except Exception as e:
            print(f"⚠️  Failed to resume from {last_ckpt_path}: {e}")
    return 1, float("inf")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    stage: int = 0,  # 0=single-stage, 1=stage1(phase/anticip), 2=stage2(triplets)
    wandb_run: Optional[Any] = None,
):
    """
    Main training loop for multi-task model.

    Args:
        model: model to train
        train_loader: training data loader
        val_loader: validation data loader
        device: device
        stage: training stage (0=single-stage, 1=stage1, 2=stage2)
    """
    # Determine training configuration based on stage
    if stage == 0:  # Single-stage training
        compute_triplets = True
        epochs = EPOCHS
        lr = LR
        ckpt_path = CKPT_PATH
        last_ckpt_path = LAST_CKPT_PATH
        stage_prefix = ""
        print("📋 Single-stage training: all tasks simultaneously")
    elif stage == 1:  # Stage 1: phase/anticipation only
        compute_triplets = False
        epochs = STAGE1_EPOCHS
        lr = STAGE1_LR
        ckpt_path = CKPT_PATH_STAGE1
        last_ckpt_path = LAST_CKPT_PATH_STAGE1
        stage_prefix = "stage1_"
        print("📋 Stage 1: Pre-training phase recognition & anticipation")
    elif stage == 2:  # Stage 2: add triplets
        compute_triplets = True
        epochs = STAGE2_EPOCHS
        lr = STAGE2_LR
        ckpt_path = CKPT_PATH_STAGE2
        last_ckpt_path = LAST_CKPT_PATH_STAGE2
        stage_prefix = "stage2_"
        print("📋 Stage 2: Fine-tuning with triplet recognition")
        # Freeze components if requested
        freeze_model_components(model, FREEZE_BACKBONE_STAGE2, FREEZE_TEMPORAL_STAGE2)
    else:
        raise ValueError(f"Invalid stage: {stage}")

    # Loss functions with label smoothing for better generalization
    reg_loss = nn.SmoothL1Loss(reduction="mean")
    ce_loss = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )  # Add label smoothing
    ce_triplet = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)  # For triplets
    compl_loss = nn.SmoothL1Loss(beta=0.3)

    optimizer = optim.AdamW(
        filter(
            lambda p: p.requires_grad, model.parameters()
        ),  # Only optimize trainable params
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )

    # Learning rate scheduler with warmup for better convergence
    if USE_WARMUP:
        # Warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS
        )
        # Main scheduler (Cosine Annealing)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - WARMUP_EPOCHS, eta_min=1e-6
        )
        # Combine them
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[WARMUP_EPOCHS],
        )
        print(f"Using warmup scheduler: {WARMUP_EPOCHS} epochs warmup")
    else:
        # Cosine Annealing LR (epoch-level)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

    # Resume if requested
    start_epoch, best_val_mae = _try_resume(
        model, optimizer, scheduler, device, last_ckpt_path
    )

    # For stage 2, track best mean triplet mAP instead of best MAE
    best_val_metric = (
        0.0 if stage == 2 else float("inf")
    )  # mean_triplet_map (maximize) vs MAE (minimize)
    if stage != 2:
        best_val_metric = best_val_mae  # Use resumed MAE value for stage 0/1

    for epoch in range(start_epoch, epochs + 1):
        model.train()

        # Track losses
        epoch_loss_total = 0.0
        epoch_loss_reg = 0.0
        epoch_loss_ce = 0.0
        epoch_loss_compl = 0.0
        epoch_loss_triplet = 0.0

        # Track metrics
        epoch_mae = 0.0
        epoch_acc = 0.0
        epoch_cmae = 0.0
        epoch_left_verb_acc = 0.0
        epoch_right_verb_acc = 0.0
        epoch_left_target_acc = 0.0
        epoch_right_target_acc = 0.0
        epoch_triplet_samples = 0

        seen = 0
        t0 = time.time()

        for it, (frames, meta) in enumerate(train_loader, start=1):
            frames = frames.to(device)

            # Phase targets
            labels = meta["phase_label"].to(device).long()
            complets_gt = meta["phase_completition"].to(device).float().unsqueeze(1)
            ttnp = torch.clamp(
                meta["time_to_next_phase"].to(device).float(), 0.0, TIME_HORIZON
            )

            # Forward pass
            reg, logits, completion_pred, extras = model(frames, meta)

            # Compute phase losses (only in stage 0 and 1, not in stage 2 where they're frozen)
            if stage != 2:
                loss_reg = reg_loss(reg, ttnp)
                loss_cls = ce_loss(logits, labels)
                loss_compl = compl_loss(completion_pred, complets_gt)
            else:
                # Stage 2: phase components are frozen, don't compute their losses
                loss_reg = torch.tensor(0.0, device=device)
                loss_cls = torch.tensor(0.0, device=device)
                loss_compl = torch.tensor(0.0, device=device)

            # Triplet losses (only if compute_triplets=True)
            loss_triplet = torch.tensor(0.0, device=device)
            triplet_metrics = {"left_verb_acc": 0.0, "right_verb_acc": 0.0}
            num_triplet_samples = 0

            if compute_triplets:
                left_targets = meta["triplet_left_classification"].to(device).long()
                right_targets = meta["triplet_right_classification"].to(device).long()

                (
                    left_triplet_loss,
                    right_triplet_loss,
                    triplet_metrics,
                    num_triplet_samples,
                ) = compute_triplet_loss_with_masking(
                    extras, left_targets, right_targets, ce_triplet
                )
                loss_triplet = (left_triplet_loss + right_triplet_loss) / 2.0

            # Total weighted loss
            if stage == 2:
                # Stage 2: only triplet loss
                loss_total = (
                    loss_triplet
                    if num_triplet_samples > 0
                    else torch.tensor(0.0, device=device, requires_grad=True)
                )
            else:
                # Stage 0/1: phase losses (and optionally triplet)
                loss_total = (
                    WEIGHT_ANTICIPATION * loss_reg
                    + WEIGHT_PHASE * loss_cls
                    + WEIGHT_COMPLETION * loss_compl
                )
                if compute_triplets and num_triplet_samples > 0:
                    loss_total += (
                        (WEIGHT_TRIPLET_VERB + WEIGHT_TRIPLET_TARGET)
                        / 2.0
                        * loss_triplet
                    )

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()

            # Gradient clipping to prevent exploding gradients (improves stability)
            if GRADIENT_CLIP_VAL > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)

            optimizer.step()

            # Calculate metrics
            with torch.no_grad():
                pred_cls = logits.argmax(dim=1)
                train_mae = torch.mean(torch.abs(reg - ttnp))
                train_acc = (pred_cls == labels).float().mean()
                train_cmae = torch.mean(torch.abs(completion_pred - complets_gt))

            bs = frames.size(0)
            epoch_loss_total += float(loss_total.item()) * bs
            epoch_loss_reg += float(loss_reg.item()) * bs
            epoch_loss_ce += float(loss_cls.item()) * bs
            epoch_loss_compl += float(loss_compl.item()) * bs
            epoch_loss_triplet += float(loss_triplet.item()) * num_triplet_samples

            epoch_mae += float(train_mae.item()) * bs
            epoch_acc += float(train_acc.item()) * bs
            epoch_cmae += float(train_cmae.item()) * bs

            # Track triplet metrics
            if num_triplet_samples > 0:
                epoch_left_verb_acc += (
                    triplet_metrics.get("left_verb_acc", 0.0) * num_triplet_samples
                )
                epoch_right_verb_acc += (
                    triplet_metrics.get("right_verb_acc", 0.0) * num_triplet_samples
                )
                epoch_left_target_acc += (
                    triplet_metrics.get("left_target_acc", 0.0) * num_triplet_samples
                )
                epoch_right_target_acc += (
                    triplet_metrics.get("right_target_acc", 0.0) * num_triplet_samples
                )
                epoch_triplet_samples += num_triplet_samples

            seen += bs

            epoch_mae += float(train_mae.item()) * bs
            epoch_acc += float(train_acc.item()) * bs
            epoch_cmae += float(train_cmae.item()) * bs
            epoch_left_verb_acc += (
                triplet_metrics["left_verb_acc"] * num_triplet_samples
            )
            epoch_right_verb_acc += (
                triplet_metrics["right_verb_acc"] * num_triplet_samples
            )
            epoch_triplet_samples += num_triplet_samples

            seen += bs

            if it % PRINT_EVERY == 0:
                cur_lr = scheduler.get_last_lr()[0]
                if compute_triplets and num_triplet_samples > 0:
                    triplet_str = f" | trip_L={triplet_metrics['left_verb_acc']:.3f} | trip_R={triplet_metrics['right_verb_acc']:.3f}"
                else:
                    triplet_str = ""
                print(
                    f"[Epoch {epoch:02d} | it {it:04d}] "
                    f"loss={loss_total.item():.4f} | mae={train_mae.item():.4f} | "
                    f"acc={train_acc.item():.4f}{triplet_str} | lr={cur_lr:.2e}"
                )

        if epoch % 5 == 0:
            print(f"GPU mem: {torch.cuda.memory_allocated() / 1e9:.4f}GB")
            print(f"GPU cached: {torch.cuda.memory_reserved() / 1e9:.4f}GB")
            gc.collect()
            torch.cuda.empty_cache()

        # Step scheduler once per epoch
        scheduler.step()

        # Print epoch summary
        N = max(1, seen)
        N_triplet = max(1, epoch_triplet_samples)

        summary = (
            f"\nEpoch {epoch:02d} [{time.time()-t0:.1f}s] LR={optimizer.param_groups[0]['lr']:.2e} "
            f"| loss={epoch_loss_total/N:.4f} mae={epoch_mae/N:.4f} "
            f"acc={epoch_acc/N:.4f} compl_mae={epoch_cmae/N:.4f}"
        )
        if compute_triplets and epoch_triplet_samples > 0:
            summary += (
                f" | triplet={epoch_loss_triplet/N_triplet:.4f} "
                f"L_verb={epoch_left_verb_acc/N_triplet:.3f} R_verb={epoch_right_verb_acc/N_triplet:.3f} "
                f"L_tgt={epoch_left_target_acc/N_triplet:.3f} R_tgt={epoch_right_target_acc/N_triplet:.3f}"
            )
        print(summary)

        # Validation
        val_stats = evaluate(
            model, val_loader, device, TIME_HORIZON, compute_triplets=compute_triplets
        )
        val_summary = (
            f"           val_mae={val_stats['mae']:.4f} val_acc={val_stats['acc']:.4f} "
            f"val_compl_mae={val_stats['compl_mae']:.4f}"
        )
        if compute_triplets and val_stats.get("triplet_samples", 0) > 0:
            val_summary += (
                f" | mAP={val_stats['mean_triplet_map']:.3f} "
                f"verb_mAP={val_stats['mean_verb_map']:.3f} target_mAP={val_stats['mean_target_map']:.3f} "
                f"(acc: mean={val_stats['mean_triplet_acc']:.3f} complete={val_stats['complete_triplet_acc']:.3f})"
            )
        print(val_summary)

        # Log to wandb if enabled with proper stage prefixes
        if USE_WANDB:
            try:
                log_dict = {
                    f"{stage_prefix}epoch": epoch,
                    f"{stage_prefix}lr": optimizer.param_groups[0]["lr"],
                    # Training metrics
                    f"{stage_prefix}train_loss": epoch_loss_total / N,
                    f"{stage_prefix}train_mae": epoch_mae / N,
                    f"{stage_prefix}train_acc": epoch_acc / N,
                    f"{stage_prefix}train_compl_mae": epoch_cmae / N,
                    # Validation metrics
                    f"{stage_prefix}val_mae": val_stats["mae"],
                    f"{stage_prefix}val_acc": val_stats["acc"],
                    f"{stage_prefix}val_compl_mae": val_stats["compl_mae"],
                }

                # Add triplet training metrics if available
                if compute_triplets and epoch_triplet_samples > 0:
                    log_dict.update(
                        {
                            f"{stage_prefix}train_triplet_loss": epoch_loss_triplet
                            / N_triplet,
                            f"{stage_prefix}train_left_verb_acc": epoch_left_verb_acc
                            / N_triplet,
                            f"{stage_prefix}train_right_verb_acc": epoch_right_verb_acc
                            / N_triplet,
                            f"{stage_prefix}train_left_target_acc": epoch_left_target_acc
                            / N_triplet,
                            f"{stage_prefix}train_right_target_acc": epoch_right_target_acc
                            / N_triplet,
                        }
                    )

                # Add triplet validation metrics if available
                if compute_triplets and val_stats.get("triplet_samples", 0) > 0:
                    log_dict.update(
                        {
                            # mAP metrics (for model selection in stage 2)
                            f"{stage_prefix}val_mean_triplet_map": val_stats[
                                "mean_triplet_map"
                            ],
                            f"{stage_prefix}val_mean_verb_map": val_stats[
                                "mean_verb_map"
                            ],
                            f"{stage_prefix}val_mean_target_map": val_stats[
                                "mean_target_map"
                            ],
                            f"{stage_prefix}val_left_verb_map": val_stats[
                                "left_verb_map"
                            ],
                            f"{stage_prefix}val_left_target_map": val_stats[
                                "left_target_map"
                            ],
                            f"{stage_prefix}val_right_verb_map": val_stats[
                                "right_verb_map"
                            ],
                            f"{stage_prefix}val_right_target_map": val_stats[
                                "right_target_map"
                            ],
                            # Accuracy metrics (for monitoring)
                            f"{stage_prefix}val_mean_triplet_acc": val_stats[
                                "mean_triplet_acc"
                            ],
                            f"{stage_prefix}val_mean_verb_acc": val_stats[
                                "mean_verb_acc"
                            ],
                            f"{stage_prefix}val_mean_target_acc": val_stats[
                                "mean_target_acc"
                            ],
                            f"{stage_prefix}val_mean_arm_acc": val_stats[
                                "mean_arm_acc"
                            ],
                            f"{stage_prefix}val_left_verb_acc": val_stats[
                                "left_verb_acc"
                            ],
                            f"{stage_prefix}val_right_verb_acc": val_stats[
                                "right_verb_acc"
                            ],
                            f"{stage_prefix}val_left_target_acc": val_stats[
                                "left_target_acc"
                            ],
                            f"{stage_prefix}val_right_target_acc": val_stats[
                                "right_target_acc"
                            ],
                            f"{stage_prefix}val_left_arm_acc": val_stats[
                                "left_arm_acc"
                            ],
                            f"{stage_prefix}val_right_arm_acc": val_stats[
                                "right_arm_acc"
                            ],
                            f"{stage_prefix}val_complete_triplet_acc": val_stats[
                                "complete_triplet_acc"
                            ],
                        }
                    )

                # Use explicit run if provided (more robust in some environments)
                if wandb_run is not None:
                    print(
                        f"[W&B DEBUG] Using provided run for logging: run_id={getattr(wandb_run, 'id', None)} stage={stage_prefix} epoch={epoch} keys={list(log_dict.keys())[:6]}"
                    )
                    wandb_run.log(log_dict, step=epoch)
                else:
                    print(
                        f"[W&B DEBUG] Using global wandb.log for stage={stage_prefix} epoch={epoch} keys={list(log_dict.keys())[:6]}"
                    )
                    wandb.log(log_dict, step=epoch)
            except Exception:
                import traceback

                print("Error logging to wandb (traceback):")
                traceback.print_exc()

        # Save best model based on appropriate metric
        # Stage 2: maximize mean triplet mAP (TRUE mean Average Precision)
        # Stage 0/1: minimize MAE
        is_best = False
        if stage == 2:
            # Stage 2: use mean triplet mAP (higher is better) - TRUE mAP, not mean accuracy
            current_metric = val_stats.get("mean_triplet_map", 0.0)
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                is_best = True
                print(
                    f"✅  New best mean_triplet_mAP={best_val_metric:.4f} — saved to: {ckpt_path}"
                )
        else:
            # Stage 0/1: use MAE (lower is better)
            current_metric = val_stats["mae"]
            if current_metric < best_val_metric:
                best_val_metric = current_metric
                is_best = True
                print(
                    f"✅  New best val_mae={best_val_metric:.4f} — saved to: {ckpt_path}"
                )

        if is_best:
            torch.save(model.state_dict(), ckpt_path)

        # Always save last for resume (using MAE for compatibility)
        _save_last_checkpoint(
            model, optimizer, scheduler, epoch, val_stats["mae"], last_ckpt_path
        )

    print(f"\n🎉 Stage {stage if stage > 0 else 'training'} completed!")
    return ckpt_path  # Return path to best checkpoint

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()

        # Track losses
        epoch_loss_total = 0.0
        epoch_loss_reg = 0.0
        epoch_loss_ce = 0.0
        epoch_loss_compl = 0.0
        epoch_loss_triplet = 0.0

        # Track metrics
        epoch_mae = 0.0
        epoch_acc = 0.0
        epoch_cmae = 0.0
        epoch_left_verb_acc = 0.0
        epoch_right_verb_acc = 0.0

        seen = 0
        t0 = time.time()

        for it, (frames, meta) in enumerate(train_loader, start=1):
            frames = frames.to(device)

            # Phase targets
            labels = meta["phase_label"].to(device).long()
            complets_gt = meta["phase_completition"].to(device).float().unsqueeze(1)
            ttnp = torch.clamp(
                meta["time_to_next_phase"].to(device).float(), 0.0, TIME_HORIZON
            )

            # Triplet targets
            left_targets = meta["triplet_left_classification"].to(device).long()
            right_targets = meta["triplet_right_classification"].to(device).long()

            # Forward pass
            reg, logits, completion_pred, extras = model(frames, meta)

            # Compute losses
            loss_reg = reg_loss(reg, ttnp)
            loss_cls = ce_loss(logits, labels)
            loss_compl = compl_loss(completion_pred, complets_gt)

            # Triplet losses with masking
            left_triplet_loss, right_triplet_loss, triplet_metrics = (
                compute_triplet_loss_with_masking(
                    extras, left_targets, right_targets, ce_triplet
                )
            )
            loss_triplet = (left_triplet_loss + right_triplet_loss) / 2.0

            # Total weighted loss
            loss_total = (
                WEIGHT_ANTICIPATION * loss_reg
                + WEIGHT_PHASE * loss_cls
                + WEIGHT_COMPLETION * loss_compl
                + (WEIGHT_TRIPLET_VERB + WEIGHT_TRIPLET_TARGET) / 2.0 * loss_triplet
            )

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            optimizer.step()

            # Calculate metrics
            with torch.no_grad():
                pred_cls = logits.argmax(dim=1)
                train_mae = torch.mean(torch.abs(reg - ttnp))
                train_acc = (pred_cls == labels).float().mean()
                train_cmae = torch.mean(torch.abs(completion_pred - complets_gt))

            bs = frames.size(0)
            epoch_loss_total += float(loss_total.item()) * bs
            epoch_loss_reg += float(loss_reg.item()) * bs
            epoch_loss_ce += float(loss_cls.item()) * bs
            epoch_loss_compl += float(loss_compl.item()) * bs
            epoch_loss_triplet += float(loss_triplet.item()) * bs

            epoch_mae += float(train_mae.item()) * bs
            epoch_acc += float(train_acc.item()) * bs
            epoch_cmae += float(train_cmae.item()) * bs
            epoch_left_verb_acc += triplet_metrics["left_verb_acc"] * bs
            epoch_right_verb_acc += triplet_metrics["right_verb_acc"] * bs

            seen += bs

            if it % PRINT_EVERY == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(
                    f"[Epoch {epoch:02d} | it {it:04d}] "
                    f"loss={loss_total.item():.4f} | mae={train_mae.item():.4f} | "
                    f"acc={train_acc.item():.4f} | trip_L={triplet_metrics['left_verb_acc']:.3f} | "
                    f"trip_R={triplet_metrics['right_verb_acc']:.3f} | lr={cur_lr:.2e}"
                )

        if epoch % 5 == 0:
            print(f"GPU mem: {torch.cuda.memory_allocated() / 1e9:.4f}GB")
            print(f"GPU cached: {torch.cuda.memory_reserved() / 1e9:.4f}GB")
            gc.collect()
            torch.cuda.empty_cache()

        # Step scheduler once per epoch
        scheduler.step()

        # Print epoch summary
        N = max(1, seen)
        print(
            f"\nEpoch {epoch:02d} [{time.time()-t0:.1f}s] "
            f"| loss={epoch_loss_total/N:.4f} mae={epoch_mae/N:.4f} "
            f"acc={epoch_acc/N:.4f} compl_mae={epoch_cmae/N:.4f} "
            f"triplet={epoch_loss_triplet/N:.4f} "
            f"L_verb={epoch_left_verb_acc/N:.3f} R_verb={epoch_right_verb_acc/N:.3f}"
        )

        # Validation
        val_stats = evaluate(model, val_loader, device, TIME_HORIZON)
        print(
            f"           val_mae={val_stats['mae']:.4f} val_acc={val_stats['acc']:.4f} "
            f"val_compl_mae={val_stats['compl_mae']:.4f} | "
            f"val_L_verb={val_stats['left_verb_acc']:.3f} val_R_verb={val_stats['right_verb_acc']:.3f} "
            f"val_L_arm={val_stats['left_arm_acc']:.3f} val_R_arm={val_stats['right_arm_acc']:.3f} "
            f"val_complete={val_stats['complete_triplet_acc']:.3f}"
        )

        # Save best model based on val MAE
        if val_stats["mae"] < best_val_mae:
            best_val_mae = val_stats["mae"]
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"✅  New best val_mae={best_val_mae:.4f} — saved to: {CKPT_PATH}")

        # Always save last for resume
        _save_last_checkpoint(model, optimizer, scheduler, epoch, best_val_mae)

    print("\n🎉 Training completed!")


# =============================================================================================
# Main
# =============================================================================================


def main():
    print("🚀 Starting Multi-Task Slow-Fast Training")
    print(f"Dataset: {ROOT_DIR}")
    print(f"Tasks: Phase Recognition + Anticipation + Completion + Dual-Arm Triplets")
    print(f"Triplet dimensions: Verbs={NUM_VERBS}, Targets={NUM_TARGETS}")

    if TWO_STAGE_TRAINING:
        print(f"\n🔬 Two-Stage Training Mode:")
        print(
            f"  Stage 1: Phase/Anticipation ({STAGE1_EPOCHS} epochs @ LR={STAGE1_LR})"
        )
        print(
            f"  Stage 2: Triplet Fine-tuning ({STAGE2_EPOCHS} epochs @ LR={STAGE2_LR})"
        )
        print(f"    - Freeze backbone: {FREEZE_BACKBONE_STAGE2}")
        print(f"    - Freeze temporal: {FREEZE_TEMPORAL_STAGE2}")
    else:
        print(f"\n🔬 Single-Stage Multi-Task Training ({EPOCHS} epochs @ LR={LR})")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create datasets based on training mode
    print("\n📁 Loading datasets...")

    if TWO_STAGE_TRAINING:
        # Stage 1: Use ALL data (force_triplets=False) for phase/anticipation training
        print("  Stage 1 datasets: ALL data (with and without triplet annotations)")
        train_ds_stage1 = PegAndRing(
            ROOT_DIR,
            mode="train",
            seq_len=SEQ_LEN,
            stride=STRIDE,
            time_unit=TIME_UNIT,
            augment=True,
            force_triplets=False,  # Use ALL data for phase/anticipation
        )

        val_ds_stage1 = PegAndRing(
            ROOT_DIR,
            mode="val",
            seq_len=SEQ_LEN,
            stride=STRIDE,
            time_unit=TIME_UNIT,
            augment=False,
            force_triplets=False,  # Use ALL data
        )

        print(f"  Stage 1 train samples: {len(train_ds_stage1)}")
        print(f"  Stage 1 val samples: {len(val_ds_stage1)}")

        # Stage 2: Use ONLY triplet-annotated data (force_triplets=True)
        print("  Stage 2 datasets: ONLY triplet-annotated data")
        train_ds_stage2 = PegAndRing(
            ROOT_DIR,
            mode="train",
            seq_len=SEQ_LEN,
            stride=STRIDE,
            time_unit=TIME_UNIT,
            augment=True,
            force_triplets=True,  # Only triplet-annotated data
        )

        val_ds_stage2 = PegAndRing(
            ROOT_DIR,
            mode="val",
            seq_len=SEQ_LEN,
            stride=STRIDE,
            time_unit=TIME_UNIT,
            augment=False,
            force_triplets=True,  # Only triplet-annotated data
        )

        print(f"  Stage 2 train samples: {len(train_ds_stage2)}")
        print(f"  Stage 2 val samples: {len(val_ds_stage2)}")

    else:
        # Single-stage: Use ALL data, handle triplet annotations dynamically
        print("  Single-stage: ALL data (triplet loss masked when not available)")
        train_ds = PegAndRing(
            ROOT_DIR,
            mode="train",
            seq_len=SEQ_LEN,
            stride=STRIDE,
            time_unit=TIME_UNIT,
            augment=True,
            force_triplets=False,  # Use ALL data, mask triplet loss as needed
        )

        val_ds = PegAndRing(
            ROOT_DIR,
            mode="val",
            seq_len=SEQ_LEN,
            stride=STRIDE,
            time_unit=TIME_UNIT,
            augment=False,
            force_triplets=False,
        )

        print(f"  Train samples: {len(train_ds)}")
        print(f"  Val samples: {len(val_ds)}")

    # Create data loaders with video-based batching
    if TWO_STAGE_TRAINING:
        # Create loaders for Stage 1 (ALL data)
        print("\n📊 Creating Stage 1 data loaders (ALL data)...")
        gen_train_s1 = torch.Generator()
        gen_train_s1.manual_seed(SEED)
        train_batch_sampler_s1 = VideoBatchSampler(
            train_ds_stage1,
            batch_size=BATCH_SIZE,
            drop_last=False,
            shuffle_videos=True,
            generator=gen_train_s1,
            batch_videos=False,
        )
        train_loader_stage1 = DataLoader(
            train_ds_stage1,
            batch_sampler=train_batch_sampler_s1,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        gen_val_s1 = torch.Generator()
        gen_val_s1.manual_seed(SEED)
        val_batch_sampler_s1 = VideoBatchSampler(
            val_ds_stage1,
            batch_size=BATCH_SIZE,
            drop_last=False,
            shuffle_videos=False,
            generator=gen_val_s1,
            batch_videos=False,
        )
        val_loader_stage1 = DataLoader(
            val_ds_stage1,
            batch_sampler=val_batch_sampler_s1,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Create loaders for Stage 2 (ONLY triplet-annotated data)
        print("📊 Creating Stage 2 data loaders (triplet-annotated only)...")
        gen_train_s2 = torch.Generator()
        gen_train_s2.manual_seed(SEED)
        train_batch_sampler_s2 = VideoBatchSampler(
            train_ds_stage2,
            batch_size=BATCH_SIZE,
            drop_last=False,
            shuffle_videos=True,
            generator=gen_train_s2,
            batch_videos=False,
        )
        train_loader_stage2 = DataLoader(
            train_ds_stage2,
            batch_sampler=train_batch_sampler_s2,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        gen_val_s2 = torch.Generator()
        gen_val_s2.manual_seed(SEED)
        val_batch_sampler_s2 = VideoBatchSampler(
            val_ds_stage2,
            batch_size=BATCH_SIZE,
            drop_last=False,
            shuffle_videos=False,
            generator=gen_val_s2,
            batch_videos=False,
        )
        val_loader_stage2 = DataLoader(
            val_ds_stage2,
            batch_sampler=val_batch_sampler_s2,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    else:
        # Single-stage: one set of loaders (ALL data)
        print("\n📊 Creating data loaders (ALL data)...")
        gen_train = torch.Generator()
        gen_train.manual_seed(SEED)
        train_batch_sampler = VideoBatchSampler(
            train_ds,
            batch_size=BATCH_SIZE,
            drop_last=False,
            shuffle_videos=True,
            generator=gen_train,
            batch_videos=False,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        gen_val = torch.Generator()
        gen_val.manual_seed(SEED)
        val_batch_sampler = VideoBatchSampler(
            val_ds,
            batch_size=BATCH_SIZE,
            drop_last=False,
            shuffle_videos=False,
            generator=gen_val,
            batch_videos=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_batch_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    # Create model
    print("\n🏗️  Building multi-task model...")
    model = MultiTaskSlowFastModel(
        sequence_length=SEQ_LEN,
        num_classes=NUM_CLASSES,
        time_horizon=TIME_HORIZON,
        backbone_fast="resnet18",
        backbone_slow="resnet18",
        pretrained_backbone_fast=True,
        pretrained_backbone_slow=True,
        freeze_backbone_fast=False,
        freeze_backbone_slow=False,
        hidden_channels=256,
        num_temporal_layers=5,
        dropout=DROPOUT_RATE,  # Use configurable dropout rate
        use_spatial_attention=True,
        attn_heads=8,
        num_verbs=NUM_VERBS,
        num_targets=NUM_TARGETS,
        triplet_hidden_dim=128,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize wandb if enabled
    wandb_run = None
    if USE_WANDB and DO_TRAIN:
        wandb_run = init_wandb()

    # Train model
    best_ckpt_path = CKPT_PATH  # Default for single-stage

    if DO_TRAIN:
        if TWO_STAGE_TRAINING:
            # ========== STAGE 1: Pre-train Phase/Anticipation ==========
            print("\n" + "=" * 80)
            print("STAGE 1: Pre-training Phase Recognition & Anticipation")
            print("  Using ALL data (with and without triplet annotations)")
            print("=" * 80)

            # Train Stage 1 with ALL data loaders
            stage1_ckpt = train(
                model,
                train_loader_stage1,
                val_loader_stage1,
                device,
                stage=1,
                wandb_run=wandb_run,
            )
            print(f"\n✅ Stage 1 completed! Best checkpoint: {stage1_ckpt}")

            # ========== STAGE 2: Fine-tune Triplets ==========
            print("\n" + "=" * 80)
            print("STAGE 2: Fine-tuning with Triplet Recognition")
            print("  Using ONLY triplet-annotated data")
            print("=" * 80)

            # Load stage 1 best checkpoint
            model.load_state_dict(torch.load(stage1_ckpt, map_location=device))
            print(f"✅ Loaded Stage 1 checkpoint: {stage1_ckpt}")

            # Train stage 2 with triplet-only data loaders (freezing will happen inside train function)
            best_ckpt_path = train(
                model,
                train_loader_stage2,
                val_loader_stage2,
                device,
                stage=2,
                wandb_run=wandb_run,
            )
            print(f"\n✅ Stage 2 completed! Best checkpoint: {best_ckpt_path}")

        else:
            # Single-stage training with ALL data (triplet loss masked dynamically)
            best_ckpt_path = train(
                model, train_loader, val_loader, device, stage=0, wandb_run=wandb_run
            )

    print("\n" + "=" * 80)
    print("🎉 Multi-task training completed!")
    print(f"Best model saved at: {best_ckpt_path}")
    print("=" * 80)

    # Finish wandb logging
    if USE_WANDB:
        try:
            import wandb

            wandb.finish()
        except:
            pass


if __name__ == "__main__":
    main()
