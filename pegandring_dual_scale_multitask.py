#!/usr/bin/env python3
"""
Dual-Scale Multi-Task Model for Peg & Ring Workflow

This model addresses temporal scale mismatch by using separate temporal encoders:
- Long-term encoder (16 frames) for phase recognition, anticipation, completion
- Short-term encoder (8 frames) for triplet recognition (actions last 2-4 seconds)

Key improvements over previous model:
1. Shared visual backbone (efficient, pretrained features)
2. Separate temporal encoders at appropriate scales (no compromise)
3. Deeper triplet classifiers with more capacity
4. Two-stage training: phase/anticipation first, then triplets
5. Better regularization and learning rate scheduling

Architecture Benefits:
- Phase tasks get long context they need (~16 frames)
- Triplet tasks get focused short-term context (~8 frames)
- No forced compromise between temporal scales
- Better gradient flow to each task group
- Memory efficient (shared visual features)

CRITICAL FIXES APPLIED (Oct 2025):
==================================
1. UNFROZE backbone + temporal encoder in Stage 2
   - Previous: Only 15% trainable (triplet heads only)
   - Problem: Features optimized for anticipation, not fine-grained actions
   - Fix: Allow full model adaptation to triplet task (~85% trainable)

2. INCREASED short-term sequence length: 4→8 frames
   - Previous: 4 frames (0.5-1 second context)
   - Problem: Insufficient temporal context for action discrimination
   - Fix: Match reference model's 8-frame context (2 seconds)

3. INCREASED Stage 2 LR: 5e-5 → 3e-4 (6x boost)
   - Previous: Fine-tuning LR with frozen features
   - Problem: Too slow for full model retraining
   - Fix: Match reference model's learning rate

4. INCREASED dropout: 0.15 → 0.3 (2x boost)
   - Previous: Light regularization
   - Problem: Potential overfitting with small triplet dataset
   - Fix: Match reference model's stronger regularization

5. REDUCED batch size: 48 → 32
   - Better generalization, match reference model
"""

import os
import sys
import random
import time
from pathlib import Path
from typing import Dict, Any, DefaultDict, Optional, List, Tuple, Set
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import average_precision_score
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from datasets.peg_and_ring_workflow import (
    PegAndRing,
    VideoBatchSampler,
    TRIPLET_VERBS,
    TRIPLET_TARGETS,
)

# Import model components from the original file
import sys

sys.path.insert(0, str(Path(__file__).parent))

# Backbone imports
try:
    from torchvision.models import (
        resnet18,
        resnet50,
        ResNet18_Weights,
        ResNet50_Weights,
    )

    _HAS_TORCHVISION_WEIGHTS_ENUM = True
except Exception:
    from torchvision.models import resnet18, resnet50

    _HAS_TORCHVISION_WEIGHTS_ENUM = False


# =============================================================================================
# Building Blocks (Reuse from original model)
# =============================================================================================


class TemporalConvBlock(nn.Module):
    """Temporal convolutional block with residual connection and causal padding."""

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
        self.pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
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
        residual = x
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))

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
    """Multi-scale temporal CNN with different dilation rates."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        dilations = [2**i for i in range(num_layers)]

        self.layers = nn.ModuleList()
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
        self.final_proj = nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.final_norm = nn.BatchNorm1d(hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        att = self.spatial_att(x)
        x = x * att
        x = self.global_pool(x).flatten(1)
        return x


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
    """Conformer convolution module."""

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
        x = self.ln(x)
        x = x.transpose(1, 2)

        x = self.pw1(x)
        a, b = x.chunk(2, dim=1)
        x = a * torch.sigmoid(b)

        pad = (self.kernel_size - 1) * self.dilation
        if pad > 0:
            x = F.pad(x, (pad, 0))
        x = self.dw(x)
        x = self.bn(x)
        x = F.gelu(x)
        x = self.pw2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
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
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TemporalConformerBlock(nn.Module):
    """FFN -> Causal Self-Attention -> Conv Module -> FFN"""

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
    """Stack of TemporalConformerBlock with increasing dilations."""

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
# Dual-Scale Multi-Task Model
# =============================================================================================


class DualScaleMultiTaskModel(nn.Module):
    """
    Dual-scale multi-task model with separate temporal encoders.

    Key Innovation: Different temporal scales for different task groups
    - Long-term (16 frames): Phase recognition, anticipation, completion
    - Short-term (8 frames): Triplet recognition (actions are 2-4 seconds)
    """

    def __init__(
        self,
        sequence_length_long: int = 16,
        sequence_length_short: int = 8,
        num_classes: int = 6,
        time_horizon: float = 2.0,
        *,
        backbone: str = "resnet18",
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        hidden_channels: int = 256,
        num_temporal_layers_long: int = 4,
        num_temporal_layers_short: int = 3,
        dropout: float = 0.15,
        use_spatial_attention: bool = True,
        attn_heads: int = 8,
        softmin_tau: float | None = None,
        sigmoid_scale: float = 1.0,
        floor_beta: float = 2.0,
        num_verbs: int = len(TRIPLET_VERBS),
        num_targets: int = len(TRIPLET_TARGETS),
        triplet_hidden_dim: int = 256,
    ):
        super().__init__()

        self.T_long = sequence_length_long
        self.T_short = sequence_length_short
        self.C = num_classes
        self.H = time_horizon
        self.hidden_channels = hidden_channels
        self.num_verbs = num_verbs
        self.num_targets = num_targets

        self.softmin_tau = softmin_tau if softmin_tau is not None else 0.02 * self.H
        self.sigmoid_scale = sigmoid_scale
        self.floor_beta = floor_beta

        # Shared Visual Backbone
        self.backbone_features, feat_dim = self._make_backbone(
            backbone, pretrained_backbone, freeze_backbone
        )

        if use_spatial_attention:
            self.spatial_pool = SpatialAttentionPool(feat_dim)
        else:
            self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        # Shared feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Long-term temporal encoder (for phase/anticipation/completion)
        self.temporal_cnn_long = MultiScaleTemporalCNN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_temporal_layers_long,
            dropout=dropout,
        )

        self.temporal_encoder_long = TemporalConformerEncoder(
            d_model=hidden_channels,
            num_layers=num_temporal_layers_long,
            num_heads=attn_heads,
            dropout=dropout,
        )

        # Short-term temporal encoder (for triplet recognition)
        self.temporal_cnn_short = MultiScaleTemporalCNN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_temporal_layers_short,
            dropout=dropout,
        )

        self.temporal_encoder_short = TemporalConformerEncoder(
            d_model=hidden_channels,
            num_layers=num_temporal_layers_short,
            num_heads=attn_heads,
            dropout=dropout,
        )

        # Phase recognition head
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

        # Anticipation head
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

        # Triplet prediction heads (deeper for better discrimination)
        self.left_arm_proj = nn.Sequential(
            nn.Linear(hidden_channels, triplet_hidden_dim),
            nn.LayerNorm(triplet_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(triplet_hidden_dim, triplet_hidden_dim),
            nn.LayerNorm(triplet_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.right_arm_proj = nn.Sequential(
            nn.Linear(hidden_channels, triplet_hidden_dim),
            nn.LayerNorm(triplet_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(triplet_hidden_dim, triplet_hidden_dim),
            nn.LayerNorm(triplet_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.left_verb_head = self._make_verb_classifier(triplet_hidden_dim, dropout)
        self.left_target_head = self._make_target_classifier(
            triplet_hidden_dim, dropout
        )
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
                bb = resnet18(pretrained=pretrained)
            feat_dim = 512
        elif backbone == "resnet50":
            if _HAS_TORCHVISION_WEIGHTS_ENUM:
                bb = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            else:
                bb = resnet50(pretrained=pretrained)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        features = nn.Sequential(*list(bb.children())[:-2])
        if freeze:
            for p in features.parameters():
                p.requires_grad = False
        return features, feat_dim

    def _make_verb_classifier(self, in_dim: int, dropout: float) -> nn.Module:
        """Create verb classifier with more capacity."""
        return nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, self.num_verbs),
        )

    def _make_target_classifier(self, in_dim: int, dropout: float) -> nn.Module:
        """Create target classifier with extra capacity for more classes."""
        return nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
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
            self.feature_proj,
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
        Forward pass with dual-scale temporal processing.

        Args:
            frames: [B, T, C, H, W] where T >= max(T_long, T_short)
            meta: metadata dict (optional)

        Returns:
            anticipation: [B, num_classes]
            phase_logits: [B, num_classes]
            completion: [B, 1]
            extras: dict with triplet predictions
        """
        B, T, C_in, H, W = frames.shape

        # ========================================================================
        # Shared Visual Feature Extraction
        # ========================================================================

        x = frames.view(B * T, C_in, H, W)
        with torch.set_grad_enabled(self.backbone_features.training):
            s = self.backbone_features(x)
        v = self._pool_spatial(self.spatial_pool, s)  # [B*T, feat_dim]

        # Project features (all timesteps at once for BatchNorm)
        v_proj = self.feature_proj(v)  # [B*T, hidden_channels]
        feats = v_proj.view(B, T, -1)  # [B, T, hidden_channels]

        # ========================================================================
        # Long-term Temporal Processing (Phase/Anticipation/Completion)
        # ========================================================================

        # Take last T_long frames
        if T >= self.T_long:
            feats_long = feats[:, -self.T_long :, :]
        else:
            pad_len = self.T_long - T
            feats_long = F.pad(feats, (0, 0, pad_len, 0))

        # Long-term temporal encoding
        feats_long_t = feats_long.transpose(1, 2)  # [B, hidden_channels, T_long]
        tf_long = self.temporal_cnn_long(feats_long_t)
        encoded_long = self.temporal_encoder_long(
            tf_long.transpose(1, 2)
        )  # [B, T_long, hidden_channels]

        # Current frame for phase tasks
        current_long = encoded_long[:, -1, :]

        # Phase classification
        phase_logits = self.phase_head(current_long)

        # Completion
        completion = self.completion_head(current_long)

        # Anticipation via attention pooling
        query = self.anticipation_query.expand(B, -1, -1)
        pooled, _ = self.temporal_attention_pool(
            query=query, key=encoded_long, value=encoded_long
        )
        pooled = pooled.squeeze(1)
        raw = self.anticipation_head(pooled)

        # Apply anticipation constraints
        if self.sigmoid_scale != 1.0:
            y = self.H * torch.sigmoid(self.sigmoid_scale * raw)
        else:
            y = self.H * torch.sigmoid(raw)

        m = self._softmin(y, dim=1, tau=self.softmin_tau)
        y = y - m
        anticipation = F.softplus(y, beta=self.floor_beta)

        # ========================================================================
        # Short-term Temporal Processing (Triplet Recognition)
        # ========================================================================

        # Take last T_short frames
        if T >= self.T_short:
            feats_short = feats[:, -self.T_short :, :]
        else:
            pad_len = self.T_short - T
            feats_short = F.pad(feats, (0, 0, pad_len, 0))

        # Short-term temporal encoding
        feats_short_t = feats_short.transpose(1, 2)  # [B, hidden_channels, T_short]
        tf_short = self.temporal_cnn_short(feats_short_t)
        encoded_short = self.temporal_encoder_short(
            tf_short.transpose(1, 2)
        )  # [B, T_short, hidden_channels]

        # Current frame for triplet tasks
        current_short = encoded_short[:, -1, :]

        # Arm-specific features
        left_features = self.left_arm_proj(current_short)
        right_features = self.right_arm_proj(current_short)

        # Triplet predictions
        left_verb_logits = self.left_verb_head(left_features)
        left_target_logits = self.left_target_head(left_features)
        right_verb_logits = self.right_verb_head(right_features)
        right_target_logits = self.right_target_head(right_features)

        extras = {
            "left_verb_logits": left_verb_logits,
            "left_target_logits": left_target_logits,
            "right_verb_logits": right_verb_logits,
            "right_target_logits": right_target_logits,
        }

        return anticipation, phase_logits, completion, extras


# =============================================================================================
# Configuration
# =============================================================================================

SEED = 42
ROOT_DIR = "data/peg_and_ring_workflow"
TIME_UNIT = "minutes"
NUM_CLASSES = 6
NUM_VERBS = len(TRIPLET_VERBS)
NUM_TARGETS = len(TRIPLET_TARGETS)

# Dual-scale sequence lengths
SEQ_LEN_LONG = 16  # For phase/anticipation tasks (need long context)
SEQ_LEN_SHORT = (
    8  # For triplet tasks (INCREASED: match reference model temporal context)
)
STRIDE = 1

# Use different batch sizes per training stage to avoid OOM in stage 2
# Stage 1: phase/anticipation (sparser sampling) - can use larger batch
# Stage 2: triplet fine-tuning (denser overlapping sampling) - reduce batch
BATCH_SIZE_STAGE1 = 16  # For Stage 1 training
BATCH_SIZE_STAGE2 = 8  # For Stage 2 training (reduced to avoid OOM)
NUM_WORKERS = 30

TIME_HORIZON = 2.0

# Two-stage training (RECOMMENDED for this architecture)
TWO_STAGE_TRAINING = True
STAGE1_EPOCHS = 100  # Phase/anticipation pre-training
STAGE2_EPOCHS = 200  # Triplet fine-tuning

# Stage 1: Learn phase/anticipation/completion with all data
STAGE1_LR = 1e-4
STAGE1_WEIGHT_DECAY = 2e-4
STAGE1_WEIGHT_ANTICIPATION = 1.0
STAGE1_WEIGHT_PHASE = 1.0
STAGE1_WEIGHT_COMPLETION = 0.5

# Stage 2: Add triplet learning (UNFROZEN: allow full model adaptation to triplets)
STAGE2_LR = (
    3e-4  # INCREASED: Match reference model LR (features need retraining for triplets)
)
STAGE2_WEIGHT_DECAY = 1e-4  # REDUCED: Match reference model (was 2e-4, too aggressive)
STAGE2_WEIGHT_ANTICIPATION = (
    0.0  # DISABLED: Focus purely on triplets (prevent task conflict)
)
STAGE2_WEIGHT_PHASE = 0.0  # DISABLED: Focus purely on triplets (prevent task conflict)
STAGE2_WEIGHT_COMPLETION = (
    0.0  # DISABLED: Focus purely on triplets (prevent task conflict)
)
STAGE2_WEIGHT_TRIPLET_VERB = 1.0
STAGE2_WEIGHT_TRIPLET_TARGET = 1.0
FREEZE_BACKBONE_STAGE2 = False  # UNFROZEN: Allow backbone to adapt to triplet task
FREEZE_LONG_TEMPORAL_STAGE2 = (
    False  # UNFROZEN: Features need retraining for fine-grained actions
)

# AP Metric Configuration
INCLUDE_NULL_IN_AP = (
    True  # Include NULL verb/target in AP calculations (part of the task)
)

# Regularization
LABEL_SMOOTHING = 0.05  # REDUCED: Match working reference model (was 0.1, too aggressive)
DROPOUT_RATE = (
    0.3  # INCREASED: Match reference model (better generalization, prevent overfitting)
)
GRADIENT_CLIP_VAL = 2.0  # INCREASED: Match reference model (allows more gradient flow)
USE_WARMUP = True
WARMUP_EPOCHS = 5

# Checkpoints
CKPT_PATH_STAGE1 = Path("dual_scale_multitask_stage1_best.pth")
CKPT_PATH_STAGE2 = Path("dual_scale_multitask_stage2_best.pth")
LAST_CKPT_PATH_STAGE1 = Path("dual_scale_multitask_stage1_last.pth")
LAST_CKPT_PATH_STAGE2 = Path("dual_scale_multitask_stage2_last.pth")

PRINT_EVERY = 40
DO_TRAIN = True
RESUME_IF_CRASH = True

# Logging
USE_WANDB = False  # Disabled - using local matplotlib plots instead

EVAL_ROOT = Path("results/eval_outputs_dual_scale")
EVAL_CLS_DIR = EVAL_ROOT / "classification"
EVAL_ANT_DIR = EVAL_ROOT / "anticipation"
EVAL_TRIPLET_DIR = EVAL_ROOT / "triplets"
PLOTS_DIR = EVAL_ROOT / "training_plots"  # Training plots in eval_outputs
EVAL_CLS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_ANT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_TRIPLET_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def init_wandb(stage: Optional[str] = None):
    """Initialize Weights & Biases logging - DISABLED."""
    return None


class TrainingPlotter:
    """Generate matplotlib plots for training metrics (replacement for W&B)."""

    def __init__(self, save_dir: Path, stage: int):
        self.save_dir = Path(save_dir)
        self.stage = stage
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Storage for metrics history
        self.history = defaultdict(list)
        self.epochs = []

        print(f"📊 Training plotter initialized: {self.save_dir}")

    def log(self, metrics: Dict[str, float], step: int):
        """Log metrics for the current epoch."""
        if step not in self.epochs:
            self.epochs.append(step)

        for key, value in metrics.items():
            self.history[key].append(value)

    def plot_all(self):
        """Generate all plots and save to disk."""
        if not self.epochs:
            print("⚠️  No data to plot yet")
            return

        try:
            stage_prefix = f"stage{self.stage}/"

            # Filter metrics by stage
            train_metrics = {
                k: v
                for k, v in self.history.items()
                if k.startswith(f"{stage_prefix}train/")
            }
            val_metrics = {
                k: v
                for k, v in self.history.items()
                if k.startswith(f"{stage_prefix}val/")
            }

            if self.stage == 1:
                # Stage 1: Phase/Anticipation metrics

                # Plot 1: Loss curves
                self._plot_metric_comparison(
                    train_key=f"{stage_prefix}train/loss",
                    val_key=f"{stage_prefix}val/mae",
                    title=f"Stage {self.stage} - Training Loss & Validation MAE",
                    ylabel_left="Train Loss",
                    ylabel_right="Val MAE",
                    filename=f"stage{self.stage}_loss_mae.png",
                )

                # Plot 2: Phase Accuracy
                self._plot_metric_comparison(
                    train_key=f"{stage_prefix}train/acc",
                    val_key=f"{stage_prefix}val/acc",
                    title=f"Stage {self.stage} - Phase Recognition Accuracy",
                    ylabel_left="Train Accuracy",
                    ylabel_right="Val Accuracy",
                    filename=f"stage{self.stage}_accuracy.png",
                )

                # Plot 3: MAE
                self._plot_metric_comparison(
                    train_key=f"{stage_prefix}train/mae",
                    val_key=f"{stage_prefix}val/mae",
                    title=f"Stage {self.stage} - Anticipation MAE",
                    ylabel_left="Train MAE",
                    ylabel_right="Val MAE",
                    filename=f"stage{self.stage}_mae.png",
                )

                # Plot 4: Completion MAE
                self._plot_metric_comparison(
                    train_key=f"{stage_prefix}train/compl_mae",
                    val_key=f"{stage_prefix}val/compl_mae",
                    title=f"Stage {self.stage} - Completion MAE",
                    ylabel_left="Train Completion MAE",
                    ylabel_right="Val Completion MAE",
                    filename=f"stage{self.stage}_completion.png",
                )

                # Plot 5: Detailed anticipation metrics
                self._plot_anticipation_metrics()

            elif self.stage == 2:
                # Stage 2: Triplet recognition metrics only

                # Plot 1: Triplet Loss
                self._plot_single_metric(
                    key=f"{stage_prefix}train/loss",
                    title=f"Stage {self.stage} - Triplet Training Loss",
                    ylabel="Training Loss",
                    filename=f"stage{self.stage}_triplet_loss.png",
                    color="tab:blue",
                )

                # Plot 2: Triplet Accuracy
                triplet_train_key = f"{stage_prefix}train/triplet_acc"
                triplet_val_key = f"{stage_prefix}val/triplet_acc"

                if (
                    triplet_train_key in self.history
                    and triplet_val_key in self.history
                ):
                    self._plot_metric_comparison(
                        train_key=triplet_train_key,
                        val_key=triplet_val_key,
                        title=f"Stage {self.stage} - Triplet Recognition Accuracy",
                        ylabel_left="Train Triplet Acc",
                        ylabel_right="Val Triplet Acc",
                        filename=f"stage{self.stage}_triplet_accuracy.png",
                    )

                # Plot 3: Detailed triplet component accuracies
                self._plot_triplet_breakdown()

                # Plot 4: mAP metrics for triplets
                self._plot_triplet_map()

            # Learning rate (both stages)
            lr_key = f"{stage_prefix}lr"
            if lr_key in self.history:
                self._plot_single_metric(
                    key=lr_key,
                    title=f"Stage {self.stage} - Learning Rate Schedule",
                    ylabel="Learning Rate",
                    filename=f"stage{self.stage}_learning_rate.png",
                    color="orange",
                )

            # Overview grid (stage-specific)
            self._plot_overview()

            print(f"✅ All plots saved to {self.save_dir}")

        except Exception as e:
            print(f"⚠️  Error generating plots: {e}")
            import traceback

            traceback.print_exc()

    def _plot_metric_comparison(
        self,
        train_key: str,
        val_key: str,
        title: str,
        ylabel_left: str,
        ylabel_right: str,
        filename: str,
    ):
        """Plot train vs validation metric."""
        if train_key not in self.history or val_key not in self.history:
            return

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel(ylabel_left, color="tab:blue")
        ax1.plot(self.epochs, self.history[train_key], "b-", label="Train", linewidth=2)
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel(ylabel_right, color="tab:orange")
        ax2.plot(
            self.epochs,
            self.history[val_key],
            "o-",
            color="tab:orange",
            label="Val",
            linewidth=2,
        )
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        plt.title(title, fontsize=14, fontweight="bold")

        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_single_metric(
        self, key: str, title: str, ylabel: str, filename: str, color: str = "tab:blue"
    ):
        """Plot a single metric."""
        if key not in self.history:
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.history[key], color=color, linewidth=2, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_triplet_breakdown(self):
        """Plot detailed triplet accuracy breakdown (stage 2 only)."""
        stage_prefix = f"stage{self.stage}/"

        metrics_to_plot = [
            (f"{stage_prefix}val/left_verb_acc", "Left Verb", "tab:blue"),
            (f"{stage_prefix}val/left_target_acc", "Left Target", "tab:cyan"),
            (f"{stage_prefix}val/right_verb_acc", "Right Verb", "tab:orange"),
            (f"{stage_prefix}val/right_target_acc", "Right Target", "tab:red"),
        ]

        plt.figure(figsize=(12, 6))

        for key, label, color in metrics_to_plot:
            if key in self.history:
                plt.plot(
                    self.epochs,
                    self.history[key],
                    label=label,
                    color=color,
                    linewidth=2,
                    marker="o",
                )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            f"Stage {self.stage} - Triplet Component Accuracies",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            self.save_dir / f"stage{self.stage}_triplet_breakdown.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_anticipation_metrics(self):
        """Plot detailed anticipation metrics (inMAE, oMAE, wMAE, eMAE) for stage 1."""
        stage_prefix = f"stage{self.stage}/"

        metrics_to_plot = [
            (f"{stage_prefix}val/inMAE", "in-horizon MAE", "tab:blue"),
            (f"{stage_prefix}val/oMAE", "out-of-horizon MAE", "tab:orange"),
            (f"{stage_prefix}val/wMAE", "weighted MAE", "tab:green"),
            (f"{stage_prefix}val/eMAE", "early MAE (< 10%)", "tab:red"),
        ]

        plt.figure(figsize=(12, 6))

        for key, label, color in metrics_to_plot:
            if key in self.history:
                plt.plot(
                    self.epochs,
                    self.history[key],
                    label=label,
                    color=color,
                    linewidth=2,
                    marker="o",
                )

        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title(
            f"Stage {self.stage} - Detailed Anticipation Metrics",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            self.save_dir / f"stage{self.stage}_anticipation_metrics.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_triplet_map(self):
        """Plot comprehensive AP metrics for triplets (stage 2 only)."""
        stage_prefix = f"stage{self.stage}/"

        # Plot comprehensive AP metrics (AP_i, AP_v, AP_t, AP_iv, AP_it, AP_ivt)
        metrics_to_plot = [
            (f"{stage_prefix}val/triplet_ap_i", "AP_i (Instrument)", "tab:blue"),
            (f"{stage_prefix}val/triplet_ap_v", "AP_v (Verb)", "tab:cyan"),
            (f"{stage_prefix}val/triplet_ap_t", "AP_t (Target)", "tab:orange"),
            (f"{stage_prefix}val/triplet_ap_iv", "AP_iv (Inst+Verb)", "tab:purple"),
            (f"{stage_prefix}val/triplet_ap_it", "AP_it (Inst+Target)", "tab:green"),
            (f"{stage_prefix}val/triplet_ap_ivt", "AP_ivt (Complete)", "tab:red"),
        ]

        plt.figure(figsize=(14, 7))

        for key, label, color in metrics_to_plot:
            if key in self.history:
                plt.plot(
                    self.epochs,
                    self.history[key],
                    label=label,
                    color=color,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                )

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Average Precision (AP)", fontsize=12)
        plt.title(
            f"Stage {self.stage} - Comprehensive Triplet AP Metrics",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.0)  # AP is between 0 and 1
        plt.tight_layout()
        plt.savefig(
            self.save_dir / f"stage{self.stage}_triplet_ap_comprehensive.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # Also plot legacy mAP metrics for backward compatibility
        legacy_metrics = [
            (f"{stage_prefix}val/triplet_verb_map", "Verb mAP", "tab:blue"),
            (f"{stage_prefix}val/triplet_target_map", "Target mAP", "tab:orange"),
            (f"{stage_prefix}val/triplet_overall_map", "Overall mAP", "tab:green"),
        ]

        plt.figure(figsize=(12, 6))

        for key, label, color in legacy_metrics:
            if key in self.history:
                plt.plot(
                    self.epochs,
                    self.history[key],
                    label=label,
                    color=color,
                    linewidth=2,
                    marker="o",
                )

        plt.xlabel("Epoch")
        plt.ylabel("mAP")
        plt.title(
            f"Stage {self.stage} - Triplet mAP Metrics (Legacy)",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(
            self.save_dir / f"stage{self.stage}_triplet_map.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_overview(self):
        """Plot grid overview of all metrics (stage-specific)."""
        stage_prefix = f"stage{self.stage}/"

        # Determine which metrics to plot based on stage
        if self.stage == 1:
            # Stage 1: Phase/Anticipation metrics
            plot_configs = [
                (f"{stage_prefix}train/loss", "Training Loss", "tab:blue"),
                (f"{stage_prefix}val/mae", "Validation MAE", "tab:orange"),
                (f"{stage_prefix}train/acc", "Train Phase Acc", "tab:green"),
                (f"{stage_prefix}val/acc", "Val Phase Acc", "tab:red"),
                (f"{stage_prefix}val/inMAE", "Val inMAE", "tab:purple"),
                (f"{stage_prefix}val/wMAE", "Val wMAE", "tab:cyan"),
            ]
        else:
            # Stage 2: Comprehensive AP metrics
            plot_configs = [
                (f"{stage_prefix}train/loss", "Training Loss", "tab:blue"),
                (f"{stage_prefix}train/triplet_acc", "Train Triplet Acc", "tab:green"),
                (f"{stage_prefix}val/triplet_acc", "Val Triplet Acc", "tab:orange"),
                (f"{stage_prefix}val/triplet_ap_i", "AP_i (Instrument)", "tab:blue"),
                (f"{stage_prefix}val/triplet_ap_t", "AP_t (Target)", "tab:orange"),
                (
                    f"{stage_prefix}val/triplet_ap_it",
                    "AP_it (Inst+Target)",
                    "tab:green",
                ),
                (f"{stage_prefix}val/triplet_ap_ivt", "AP_ivt (Complete)", "tab:red"),
                (f"{stage_prefix}val/triplet_overall_map", "Overall mAP", "tab:purple"),
            ]

        # Filter out missing metrics
        plot_configs = [(k, l, c) for k, l, c in plot_configs if k in self.history]

        if not plot_configs:
            return

        # Calculate grid size
        n_plots = len(plot_configs)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for idx, (key, label, color) in enumerate(plot_configs):
            ax = axes[idx]
            ax.plot(
                self.epochs,
                self.history[key],
                color=color,
                linewidth=2,
                marker="o",
                markersize=4,
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel(label)
            ax.set_title(label, fontweight="bold")
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(len(plot_configs), len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            f"Stage {self.stage} - Training Overview", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(
            self.save_dir / f"stage{self.stage}_overview.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


# =============================================================================================
# Continue with loss functions, training, evaluation from original file...
# (I'll import and reuse the existing ones)
# =============================================================================================


def compute_triplet_loss_with_masking(
    outputs: Dict[str, torch.Tensor],
    left_targets: torch.Tensor,
    right_targets: torch.Tensor,
    ce_criterion: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], int]:
    """Compute triplet loss with masking for partial annotations."""
    B = left_targets.shape[0]

    left_verb_gt = left_targets[:, 0]
    left_target_gt = left_targets[:, 2]
    right_verb_gt = right_targets[:, 0]
    right_target_gt = right_targets[:, 2]

    NULL_VERB_IDX = NUM_VERBS - 1
    NULL_TARGET_IDX = NUM_TARGETS - 1

    left_has_action = (left_verb_gt != NULL_VERB_IDX) | (
        left_target_gt != NULL_TARGET_IDX
    )
    right_has_action = (right_verb_gt != NULL_VERB_IDX) | (
        right_target_gt != NULL_TARGET_IDX
    )
    has_valid_triplet = left_has_action | right_has_action

    num_valid = has_valid_triplet.sum().item()

    if num_valid == 0:
        device = outputs["left_verb_logits"].device
        zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
        metrics = {
            "left_verb_acc": 0.0,
            "left_target_acc": 0.0,
            "right_verb_acc": 0.0,
            "right_target_acc": 0.0,
        }
        return zero_loss, zero_loss, metrics, 0

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

    left_loss = left_verb_loss + left_target_loss
    right_loss = right_verb_loss + right_target_loss

    with torch.no_grad():
        left_verb_pred = outputs["left_verb_logits"][valid_idx].argmax(dim=1)
        left_target_pred = outputs["left_target_logits"][valid_idx].argmax(dim=1)
        right_verb_pred = outputs["right_verb_logits"][valid_idx].argmax(dim=1)
        right_target_pred = outputs["right_target_logits"][valid_idx].argmax(dim=1)

        left_verb_acc = (left_verb_pred == left_verb_gt[valid_idx]).float().mean()
        left_target_acc = (left_target_pred == left_target_gt[valid_idx]).float().mean()
        right_verb_acc = (right_verb_pred == right_verb_gt[valid_idx]).float().mean()
        right_target_acc = (
            (right_target_pred == right_target_gt[valid_idx]).float().mean()
        )

        metrics = {
            "left_verb_acc": float(left_verb_acc),
            "left_target_acc": float(left_target_acc),
            "right_verb_acc": float(right_verb_acc),
            "right_target_acc": float(right_target_acc),
        }

    return left_loss, right_loss, metrics, num_valid


def calculate_comprehensive_triplet_metrics(
    verb_preds,
    subject_preds,
    dest_preds,
    verb_gts,
    subject_gts,
    dest_gts,
    verb_probs,
    subject_probs,
    dest_probs,
) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics for triplet classification."""

    # Individual component accuracies
    verb_acc = (verb_preds == verb_gts).mean()
    subject_acc = (subject_preds == subject_gts).mean()
    dest_acc = (dest_preds == dest_gts).mean()

    # Combination accuracies
    verb_subject_acc = (
        (verb_preds == verb_gts) & (subject_preds == subject_gts)
    ).mean()
    verb_dest_acc = ((verb_preds == verb_gts) & (dest_preds == dest_gts)).mean()
    subject_dest_acc = (
        (subject_preds == subject_gts) & (dest_preds == dest_gts)
    ).mean()
    complete_acc = (
        (verb_preds == verb_gts)
        & (subject_preds == subject_gts)
        & (dest_preds == dest_gts)
    ).mean()

    # Calculate mAP for each component (multi-class setting)
    def calculate_multiclass_map(y_true, y_prob, num_classes):
        """Calculate mAP for multi-class classification using one-vs-rest approach."""
        maps = []
        for class_idx in range(num_classes):
            # Convert to binary classification (class vs rest)
            binary_true = (y_true == class_idx).astype(int)
            class_probs = y_prob[:, class_idx]
            if len(np.unique(binary_true)) > 1:  # Check if both classes exist
                ap = average_precision_score(binary_true, class_probs)
                maps.append(ap)
        return np.mean(maps) if maps else 0.0

    verb_map = calculate_multiclass_map(verb_gts, verb_probs, NUM_VERBS)
    subject_map = calculate_multiclass_map(
        subject_gts, subject_probs, len(TRIPLET_VERBS)
    )  # Placeholder - will be removed
    dest_map = calculate_multiclass_map(dest_gts, dest_probs, NUM_TARGETS)

    # Overall mAP (average of verb and target mAPs - no subject in dual scale model)
    overall_map = (verb_map + dest_map) / 2.0

    return {
        # Individual accuracies
        "triplet_verb_acc": float(verb_acc),
        "triplet_destination_acc": float(dest_acc),
        # Combination accuracies
        "triplet_verb_destination_acc": float(verb_dest_acc),
        # mAP metrics
        "triplet_verb_map": float(verb_map),
        "triplet_destination_map": float(dest_map),
        "triplet_overall_map": float(overall_map),
        # Additional derived metrics
        "triplet_avg_component_acc": float((verb_acc + dest_acc) / 2.0),
    }


def calculate_anticipation_metrics(
    time_preds: np.ndarray, time_gts: np.ndarray, time_horizon: float
) -> Dict[str, float]:
    """
    Calculate detailed anticipation metrics: inMAE, oMAE, wMAE, eMAE.

    Args:
        time_preds: Predicted times to next phase
        time_gts: Ground truth times to next phase
        time_horizon: Maximum prediction horizon

    Returns:
        Dictionary with inMAE, oMAE, wMAE, eMAE metrics
    """
    abs_errors = np.abs(time_preds - time_gts)
    h = time_horizon

    # inMAE: in-horizon MAE (when ground truth < horizon)
    in_indices = time_gts < h
    in_errors = abs_errors[in_indices] if np.any(in_indices) else np.array([])
    inMAE = np.mean(in_errors) if in_errors.size > 0 else 0.0

    # oMAE: out-of-horizon MAE (when ground truth == horizon)
    out_indices = time_gts == h
    out_errors = (
        np.abs(time_preds[out_indices] - h) if np.any(out_indices) else np.array([])
    )
    oMAE = np.mean(out_errors) if out_errors.size > 0 else 0.0

    # wMAE: weighted MAE (average of inMAE and oMAE)
    wMAE = (inMAE + oMAE) / 2.0

    # eMAE: very-short-term MAE (when ground truth < 0.1 * horizon)
    e_indices = time_gts < (0.1 * h)
    e_errors = abs_errors[e_indices] if np.any(e_indices) else np.array([])
    eMAE = np.mean(e_errors) if e_errors.size > 0 else 0.0

    return {
        "inMAE": float(inMAE),
        "oMAE": float(oMAE),
        "wMAE": float(wMAE),
        "eMAE": float(eMAE),
    }


print("✅ Dual-Scale Multi-Task Model implementation loaded!")
print(f"📊 Long-term sequence: {SEQ_LEN_LONG} frames (phase/anticipation)")
print(f"📊 Short-term sequence: {SEQ_LEN_SHORT} frames (triplets)")
print(f"🎯 Two-stage training: {TWO_STAGE_TRAINING}")
print(
    f"🔓 Stage 2 unfrozen: backbone={not FREEZE_BACKBONE_STAGE2}, temporal={not FREEZE_LONG_TEMPORAL_STAGE2}"
)
print(f"📈 Stage 2 LR: {STAGE2_LR:.2e} (6x boost), Dropout: {DROPOUT_RATE} (2x boost)")
print(f"🎲 Batch sizes: stage1={BATCH_SIZE_STAGE1}, stage2={BATCH_SIZE_STAGE2} (stage2 reduced to avoid OOM)")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    time_horizon: float,
    compute_triplets: bool = True,
) -> Dict[str, Any]:
    """Evaluate model on all tasks with comprehensive metrics."""
    model.eval()

    total_mae = 0.0
    total_ce = 0.0
    total_acc = 0.0
    total_cmae = 0.0

    total_left_verb_acc = 0.0
    total_left_target_acc = 0.0
    total_right_verb_acc = 0.0
    total_right_target_acc = 0.0
    total_triplet_samples = 0

    total_samples = 0

    ce_loss = nn.CrossEntropyLoss()

    NULL_VERB_IDX = NUM_VERBS - 1
    NULL_TARGET_IDX = NUM_TARGETS - 1

    # Collect predictions for comprehensive metrics
    all_time_preds = []
    all_time_gts = []

    # Collect triplet predictions for mAP calculation
    all_left_verb_preds = []
    all_left_verb_gts = []
    all_left_verb_probs = []
    all_left_target_preds = []
    all_left_target_gts = []
    all_left_target_probs = []
    all_right_verb_preds = []
    all_right_verb_gts = []
    all_right_verb_probs = []
    all_right_target_preds = []
    all_right_target_gts = []
    all_right_target_probs = []

    for _, (frames, meta) in enumerate(loader):
        frames = frames.to(device)

        ttnp = meta["time_to_next_phase"].to(device).float()
        ttnp = torch.clamp(ttnp, min=0.0, max=time_horizon)
        labels = meta["phase_label"].to(device).long()
        completion_gt = meta["phase_completition"].to(device).float().unsqueeze(1)

        reg, logits, completion_pred, extras = model(frames, meta)

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

        # Collect anticipation predictions for detailed metrics
        all_time_preds.append(reg.cpu().numpy())
        all_time_gts.append(ttnp.cpu().numpy())

        if compute_triplets and "triplet_left_classification" in meta:
            left_targets = meta["triplet_left_classification"].to(device).long()
            right_targets = meta["triplet_right_classification"].to(device).long()

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
                valid_idx = has_valid_triplet
                num_valid = valid_idx.sum().item()

                # Get predictions and probabilities
                left_verb_logits = extras["left_verb_logits"][valid_idx]
                left_target_logits = extras["left_target_logits"][valid_idx]
                right_verb_logits = extras["right_verb_logits"][valid_idx]
                right_target_logits = extras["right_target_logits"][valid_idx]

                left_verb_pred = left_verb_logits.argmax(dim=1)
                left_target_pred = left_target_logits.argmax(dim=1)
                right_verb_pred = right_verb_logits.argmax(dim=1)
                right_target_pred = right_target_logits.argmax(dim=1)

                # Get probabilities for mAP
                left_verb_probs = torch.softmax(left_verb_logits, dim=1)
                left_target_probs = torch.softmax(left_target_logits, dim=1)
                right_verb_probs = torch.softmax(right_verb_logits, dim=1)
                right_target_probs = torch.softmax(right_target_logits, dim=1)

                left_verb_acc = (
                    (left_verb_pred == left_verb_gt[valid_idx]).float().mean()
                )
                left_target_acc = (
                    (left_target_pred == left_target_gt[valid_idx]).float().mean()
                )
                right_verb_acc = (
                    (right_verb_pred == right_verb_gt[valid_idx]).float().mean()
                )
                right_target_acc = (
                    (right_target_pred == right_target_gt[valid_idx]).float().mean()
                )

                total_left_verb_acc += float(left_verb_acc.item()) * num_valid
                total_left_target_acc += float(left_target_acc.item()) * num_valid
                total_right_verb_acc += float(right_verb_acc.item()) * num_valid
                total_right_target_acc += float(right_target_acc.item()) * num_valid
                total_triplet_samples += num_valid

                # Collect for comprehensive metrics
                all_left_verb_preds.append(left_verb_pred.cpu().numpy())
                all_left_verb_gts.append(left_verb_gt[valid_idx].cpu().numpy())
                all_left_verb_probs.append(left_verb_probs.cpu().numpy())
                all_left_target_preds.append(left_target_pred.cpu().numpy())
                all_left_target_gts.append(left_target_gt[valid_idx].cpu().numpy())
                all_left_target_probs.append(left_target_probs.cpu().numpy())
                all_right_verb_preds.append(right_verb_pred.cpu().numpy())
                all_right_verb_gts.append(right_verb_gt[valid_idx].cpu().numpy())
                all_right_verb_probs.append(right_verb_probs.cpu().numpy())
                all_right_target_preds.append(right_target_pred.cpu().numpy())
                all_right_target_gts.append(right_target_gt[valid_idx].cpu().numpy())
                all_right_target_probs.append(right_target_probs.cpu().numpy())

    N = max(1, total_samples)
    N_triplet = max(1, total_triplet_samples)

    results = {
        "mae": total_mae / N,
        "ce": total_ce / N,
        "acc": total_acc / N,
        "compl_mae": total_cmae / N,
        "samples": total_samples,
    }

    # Calculate detailed anticipation metrics (inMAE, oMAE, wMAE, eMAE)
    if len(all_time_preds) > 0:
        time_preds_np = np.concatenate(all_time_preds)
        time_gts_np = np.concatenate(all_time_gts)
        anticipation_metrics = calculate_anticipation_metrics(
            time_preds_np, time_gts_np, time_horizon
        )
        results.update(anticipation_metrics)

    if compute_triplets and total_triplet_samples > 0:
        results.update(
            {
                "left_verb_acc": total_left_verb_acc / N_triplet,
                "left_target_acc": total_left_target_acc / N_triplet,
                "right_verb_acc": total_right_verb_acc / N_triplet,
                "right_target_acc": total_right_target_acc / N_triplet,
                "mean_triplet_acc": (
                    total_left_verb_acc
                    + total_left_target_acc
                    + total_right_verb_acc
                    + total_right_target_acc
                )
                / (4.0 * N_triplet),
                "triplet_samples": total_triplet_samples,
            }
        )

        # Calculate comprehensive triplet AP metrics
        # AP_i (instrument/verb), AP_v (verb - kept for compatibility), AP_t (target)
        # AP_iv (instrument+verb - same as AP_i in dual-arm setting), AP_it (instrument+target), AP_ivt (all)
        # Always add AP fields (even if zero) so they get logged
        ap_i = 0.0  # Instrument (verb in our case)
        ap_v = 0.0  # Verb (same as instrument)
        ap_t = 0.0  # Target
        ap_iv = 0.0  # Instrument + Verb (same as instrument alone in our model)
        ap_it = 0.0  # Instrument + Target
        ap_ivt = 0.0  # Instrument + Verb + Target (complete triplet)

        if len(all_left_verb_preds) > 0:
            # Combine left and right predictions for overall metrics
            all_verb_preds = np.concatenate(all_left_verb_preds + all_right_verb_preds)
            all_verb_gts = np.concatenate(all_left_verb_gts + all_right_verb_gts)
            all_verb_probs = np.concatenate(all_left_verb_probs + all_right_verb_probs)
            all_target_preds = np.concatenate(
                all_left_target_preds + all_right_target_preds
            )
            all_target_gts = np.concatenate(all_left_target_gts + all_right_target_gts)
            all_target_probs = np.concatenate(
                all_left_target_probs + all_right_target_probs
            )

            # AP_i (Instrument/Verb): mAP across all verb classes
            verb_aps = []
            num_verb_classes = NUM_VERBS if INCLUDE_NULL_IN_AP else NUM_VERBS - 1
            for class_idx in range(num_verb_classes):
                binary_true = (all_verb_gts == class_idx).astype(int)
                class_probs = all_verb_probs[:, class_idx]
                # Only compute AP if this class exists in ground truth
                if np.sum(binary_true) > 0:
                    try:
                        ap = average_precision_score(binary_true, class_probs)
                        verb_aps.append(ap)
                    except:
                        pass  # Skip if AP calculation fails
            ap_i = np.mean(verb_aps) if len(verb_aps) > 0 else 0.0
            ap_v = ap_i  # In our model, verb = instrument

            # AP_t (Target): mAP across all target classes
            target_aps = []
            num_target_classes = NUM_TARGETS if INCLUDE_NULL_IN_AP else NUM_TARGETS - 1
            for class_idx in range(num_target_classes):
                binary_true = (all_target_gts == class_idx).astype(int)
                class_probs = all_target_probs[:, class_idx]
                # Only compute AP if this class exists in ground truth
                if np.sum(binary_true) > 0:
                    try:
                        ap = average_precision_score(binary_true, class_probs)
                        target_aps.append(ap)
                    except:
                        pass  # Skip if AP calculation fails
            ap_t = np.mean(target_aps) if len(target_aps) > 0 else 0.0

            # AP_iv (Instrument+Verb): Same as AP_i in our dual-arm model
            ap_iv = ap_i

            # AP_it (Instrument+Target): Binary classification for each (verb, target) pair
            it_aps = []
            for v_idx in range(num_verb_classes):
                for t_idx in range(num_target_classes):
                    # Binary: does this sample match (v_idx, t_idx)?
                    binary_true = (
                        (all_verb_gts == v_idx) & (all_target_gts == t_idx)
                    ).astype(int)
                    # Only compute AP if this combination exists in ground truth
                    if np.sum(binary_true) > 0:
                        try:
                            # Combine probabilities: P(verb=v_idx) * P(target=t_idx)
                            combined_probs = (
                                all_verb_probs[:, v_idx] * all_target_probs[:, t_idx]
                            )
                            ap = average_precision_score(binary_true, combined_probs)
                            it_aps.append(ap)
                        except:
                            pass  # Skip if AP calculation fails
            ap_it = np.mean(it_aps) if len(it_aps) > 0 else 0.0

            # AP_ivt (Complete triplet): Same as AP_it in our model (no separate subject)
            ap_ivt = ap_it

            # Debug: Print AP calculation details
            null_status = "including NULL" if INCLUDE_NULL_IN_AP else "excluding NULL"
            print(
                f"    AP Calculation ({null_status}): {len(verb_aps)} verb classes, {len(target_aps)} target classes, {len(it_aps)} verb-target pairs"
            )

        # Always add comprehensive AP metrics to results
        results.update(
            {
                # Individual component APs
                "triplet_ap_i": float(ap_i),  # AP_i (instrument/verb)
                "triplet_ap_v": float(ap_v),  # AP_v (verb, same as instrument)
                "triplet_ap_t": float(ap_t),  # AP_t (target)
                # Combination APs
                "triplet_ap_iv": float(ap_iv),  # AP_iv (instrument+verb)
                "triplet_ap_it": float(ap_it),  # AP_it (instrument+target)
                "triplet_ap_ivt": float(ap_ivt),  # AP_ivt (complete triplet)
                # Legacy metrics (keep for backward compatibility)
                "triplet_verb_map": float(ap_i),
                "triplet_target_map": float(ap_t),
                "triplet_overall_map": float(ap_ivt),
            }
        )
    elif compute_triplets:
        results.update(
            {
                "left_verb_acc": 0.0,
                "left_target_acc": 0.0,
                "right_verb_acc": 0.0,
                "right_target_acc": 0.0,
                "mean_triplet_acc": 0.0,
                "triplet_samples": 0,
                # Comprehensive AP metrics
                "triplet_ap_i": 0.0,
                "triplet_ap_v": 0.0,
                "triplet_ap_t": 0.0,
                "triplet_ap_iv": 0.0,
                "triplet_ap_it": 0.0,
                "triplet_ap_ivt": 0.0,
                # Legacy metrics
                "triplet_verb_map": 0.0,
                "triplet_target_map": 0.0,
                "triplet_overall_map": 0.0,
            }
        )

    return results


def freeze_model_components(
    model: nn.Module, freeze_backbone: bool = True, freeze_long_temporal: bool = True
):
    """Freeze specific components for stage 2 training."""
    if freeze_backbone:
        print("🔒 Freezing visual backbone...")
        for param in model.backbone_features.parameters():
            param.requires_grad = False
        for param in model.spatial_pool.parameters():
            param.requires_grad = False
        for param in model.feature_proj.parameters():
            param.requires_grad = False

    if freeze_long_temporal:
        print("🔒 Freezing long-term temporal encoder and phase heads...")
        for param in model.temporal_cnn_long.parameters():
            param.requires_grad = False
        for param in model.temporal_encoder_long.parameters():
            param.requires_grad = False
        for param in model.phase_head.parameters():
            param.requires_grad = False
        for param in model.anticipation_head.parameters():
            param.requires_grad = False
        model.anticipation_query.requires_grad = False
        for param in model.temporal_attention_pool.parameters():
            param.requires_grad = False
        for param in model.completion_head.parameters():
            param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(
        f"📊 Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)"
    )


class BestMetricsTracker:
    """Track best validation metrics across training."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.results_dir / "best_validation_metrics.json"

        # Initialize tracking dictionaries with default structure
        default_metrics = {
            "stage1": {
                "best_val_mae": float("inf"),
                "best_val_acc": 0.0,
                "best_inMAE": float("inf"),
                "best_oMAE": float("inf"),
                "best_wMAE": float("inf"),
                "best_eMAE": float("inf"),
            },
            "stage2": {
                # Comprehensive AP metrics
                "best_ap_i": 0.0,  # AP_i (instrument/verb)
                "best_ap_v": 0.0,  # AP_v (verb)
                "best_ap_t": 0.0,  # AP_t (target)
                "best_ap_iv": 0.0,  # AP_iv (instrument+verb)
                "best_ap_it": 0.0,  # AP_it (instrument+target)
                "best_ap_ivt": 0.0,  # AP_ivt (complete triplet)
                # Legacy metrics
                "best_triplet_acc": 0.0,
                "best_verb_map": 0.0,
                "best_target_map": 0.0,
                "best_overall_map": 0.0,
            },
        }

        # Load existing metrics if available and merge with defaults
        if self.filepath.exists():
            try:
                with open(self.filepath, "r") as f:
                    loaded_metrics = json.load(f)
                # Merge loaded metrics with defaults (keeps new keys if added)
                self.best_metrics = default_metrics.copy()
                for stage_key in loaded_metrics:
                    if stage_key in self.best_metrics:
                        self.best_metrics[stage_key].update(loaded_metrics[stage_key])
            except Exception as e:
                print(f"⚠️  Error loading metrics, using defaults: {e}")
                self.best_metrics = default_metrics
        else:
            self.best_metrics = default_metrics

    def update(self, stage: int, val_stats: Dict[str, Any]) -> Dict[str, bool]:
        """
        Update best metrics for the given stage.

        Returns:
            Dictionary indicating which metrics were improved
        """
        improved = {}
        stage_key = f"stage{stage}"

        if stage == 1:
            # Stage 1 metrics (lower is better for MAE, higher for accuracy)
            if val_stats["mae"] < self.best_metrics[stage_key]["best_val_mae"]:
                self.best_metrics[stage_key]["best_val_mae"] = float(val_stats["mae"])
                improved["mae"] = True

            if val_stats["acc"] > self.best_metrics[stage_key]["best_val_acc"]:
                self.best_metrics[stage_key]["best_val_acc"] = float(val_stats["acc"])
                improved["acc"] = True

            # Detailed anticipation metrics (if available)
            if "inMAE" in val_stats:
                if val_stats["inMAE"] < self.best_metrics[stage_key]["best_inMAE"]:
                    self.best_metrics[stage_key]["best_inMAE"] = float(
                        val_stats["inMAE"]
                    )
                    improved["inMAE"] = True

                if val_stats["oMAE"] < self.best_metrics[stage_key]["best_oMAE"]:
                    self.best_metrics[stage_key]["best_oMAE"] = float(val_stats["oMAE"])
                    improved["oMAE"] = True

                if val_stats["wMAE"] < self.best_metrics[stage_key]["best_wMAE"]:
                    self.best_metrics[stage_key]["best_wMAE"] = float(val_stats["wMAE"])
                    improved["wMAE"] = True

                if val_stats["eMAE"] < self.best_metrics[stage_key]["best_eMAE"]:
                    self.best_metrics[stage_key]["best_eMAE"] = float(val_stats["eMAE"])
                    improved["eMAE"] = True

        elif stage == 2:
            # Stage 2 metrics (higher is better for AP/mAP)
            # Comprehensive AP metrics
            if "triplet_ap_i" in val_stats:
                if (
                    val_stats["triplet_ap_i"]
                    > self.best_metrics[stage_key]["best_ap_i"]
                ):
                    self.best_metrics[stage_key]["best_ap_i"] = float(
                        val_stats["triplet_ap_i"]
                    )
                    improved["ap_i"] = True

            if "triplet_ap_v" in val_stats:
                if (
                    val_stats["triplet_ap_v"]
                    > self.best_metrics[stage_key]["best_ap_v"]
                ):
                    self.best_metrics[stage_key]["best_ap_v"] = float(
                        val_stats["triplet_ap_v"]
                    )
                    improved["ap_v"] = True

            if "triplet_ap_t" in val_stats:
                if (
                    val_stats["triplet_ap_t"]
                    > self.best_metrics[stage_key]["best_ap_t"]
                ):
                    self.best_metrics[stage_key]["best_ap_t"] = float(
                        val_stats["triplet_ap_t"]
                    )
                    improved["ap_t"] = True

            if "triplet_ap_iv" in val_stats:
                if (
                    val_stats["triplet_ap_iv"]
                    > self.best_metrics[stage_key]["best_ap_iv"]
                ):
                    self.best_metrics[stage_key]["best_ap_iv"] = float(
                        val_stats["triplet_ap_iv"]
                    )
                    improved["ap_iv"] = True

            if "triplet_ap_it" in val_stats:
                if (
                    val_stats["triplet_ap_it"]
                    > self.best_metrics[stage_key]["best_ap_it"]
                ):
                    self.best_metrics[stage_key]["best_ap_it"] = float(
                        val_stats["triplet_ap_it"]
                    )
                    improved["ap_it"] = True

            if "triplet_ap_ivt" in val_stats:
                if (
                    val_stats["triplet_ap_ivt"]
                    > self.best_metrics[stage_key]["best_ap_ivt"]
                ):
                    self.best_metrics[stage_key]["best_ap_ivt"] = float(
                        val_stats["triplet_ap_ivt"]
                    )
                    improved["ap_ivt"] = True

            # Legacy metrics (keep for backward compatibility)
            if "triplet_verb_map" in val_stats:
                if (
                    val_stats["triplet_verb_map"]
                    > self.best_metrics[stage_key]["best_verb_map"]
                ):
                    self.best_metrics[stage_key]["best_verb_map"] = float(
                        val_stats["triplet_verb_map"]
                    )
                    improved["verb_map"] = True

                if (
                    val_stats["triplet_target_map"]
                    > self.best_metrics[stage_key]["best_target_map"]
                ):
                    self.best_metrics[stage_key]["best_target_map"] = float(
                        val_stats["triplet_target_map"]
                    )
                    improved["target_map"] = True

                if (
                    val_stats["triplet_overall_map"]
                    > self.best_metrics[stage_key]["best_overall_map"]
                ):
                    self.best_metrics[stage_key]["best_overall_map"] = float(
                        val_stats["triplet_overall_map"]
                    )
                    improved["overall_map"] = True

        # Save to file
        if improved:
            self.save()

        return improved

    def save(self):
        """Save best metrics to JSON file."""
        with open(self.filepath, "w") as f:
            json.dump(self.best_metrics, f, indent=2)

    def print_summary(self):
        """Print a summary of best metrics."""
        print("\n" + "=" * 80)
        print("📊 BEST VALIDATION METRICS SUMMARY")
        print("=" * 80)

        print("\n🎯 Stage 1 (Phase Recognition & Anticipation):")
        s1 = self.best_metrics["stage1"]
        print(f"  Best Val MAE:        {s1['best_val_mae']:.4f}")
        print(f"  Best Val Accuracy:   {s1['best_val_acc']:.4f}")
        print(f"  Best inMAE:          {s1['best_inMAE']:.4f}")
        print(f"  Best oMAE:           {s1['best_oMAE']:.4f}")
        print(f"  Best wMAE:           {s1['best_wMAE']:.4f}")
        print(f"  Best eMAE:           {s1['best_eMAE']:.4f}")

        print("\n🎯 Stage 2 (Triplet Recognition):")
        s2 = self.best_metrics["stage2"]
        print(f"\n  📈 Comprehensive Average Precision Metrics:")
        print(f"  AP_i  (Instrument):           {s2['best_ap_i']:.4f}")
        print(f"  AP_v  (Verb):                 {s2['best_ap_v']:.4f}")
        print(f"  AP_t  (Target):               {s2['best_ap_t']:.4f}")
        print(f"  AP_iv (Instrument+Verb):      {s2['best_ap_iv']:.4f}")
        print(f"  AP_it (Instrument+Target):    {s2['best_ap_it']:.4f}")
        print(f"  AP_ivt (Complete Triplet):    {s2['best_ap_ivt']:.4f}")
        print(f"\n  📊 Legacy mAP Metrics:")
        print(f"  Best Verb mAP:                {s2['best_verb_map']:.4f}")
        print(f"  Best Target mAP:              {s2['best_target_map']:.4f}")
        print(f"  Best Overall mAP:             {s2['best_overall_map']:.4f}")

        print("=" * 80 + "\n")
        print(f"📁 Metrics saved to: {self.filepath}")
        print("=" * 80)

    def save_latex_table(self):
        """Save metrics in LaTeX table format."""
        s2 = self.best_metrics["stage2"]

        latex_path = self.results_dir / "best_metrics_latex.txt"
        with open(latex_path, "w") as f:
            f.write("% LaTeX table row for comprehensive AP metrics\n")
            f.write("% Format: AP_i & AP_v & AP_t & AP_iv & AP_it & AP_ivt\n\n")

            # Main comprehensive metrics
            f.write("% Comprehensive AP Metrics:\n")
            f.write(f"{s2['best_ap_i']:.4f} & ")
            f.write(f"{s2['best_ap_v']:.4f} & ")
            f.write(f"{s2['best_ap_t']:.4f} & ")
            f.write(f"{s2['best_ap_iv']:.4f} & ")
            f.write(f"{s2['best_ap_it']:.4f} & ")
            f.write(f"{s2['best_ap_ivt']:.4f}\n\n")

            # Also save in percentage format
            f.write("% Comprehensive AP Metrics (percentage format):\n")
            f.write(f"{s2['best_ap_i']*100:.2f} & ")
            f.write(f"{s2['best_ap_v']*100:.2f} & ")
            f.write(f"{s2['best_ap_t']*100:.2f} & ")
            f.write(f"{s2['best_ap_iv']*100:.2f} & ")
            f.write(f"{s2['best_ap_it']*100:.2f} & ")
            f.write(f"{s2['best_ap_ivt']*100:.2f}\n\n")

            # Individual metrics for reference
            f.write("% Individual metric values:\n")
            f.write(
                f"% AP_i (Instrument/Verb):    {s2['best_ap_i']:.4f} ({s2['best_ap_i']*100:.2f}%)\n"
            )
            f.write(
                f"% AP_v (Verb):                {s2['best_ap_v']:.4f} ({s2['best_ap_v']*100:.2f}%)\n"
            )
            f.write(
                f"% AP_t (Target):              {s2['best_ap_t']:.4f} ({s2['best_ap_t']*100:.2f}%)\n"
            )
            f.write(
                f"% AP_iv (Instrument+Verb):    {s2['best_ap_iv']:.4f} ({s2['best_ap_iv']*100:.2f}%)\n"
            )
            f.write(
                f"% AP_it (Instrument+Target):  {s2['best_ap_it']:.4f} ({s2['best_ap_it']*100:.2f}%)\n"
            )
            f.write(
                f"% AP_ivt (Complete Triplet):  {s2['best_ap_ivt']:.4f} ({s2['best_ap_ivt']*100:.2f}%)\n"
            )

        print(f"📄 LaTeX table saved to: {latex_path}")


def train_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    stage: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    task_weights: Dict[str, float],
    compute_triplets: bool,
    ckpt_path: Path,
    last_ckpt_path: Path,
    plotter: Optional[TrainingPlotter] = None,
    best_metrics_tracker: Optional[BestMetricsTracker] = None,
):
    """Train a single stage."""
    print(f"\n{'='*80}")
    print(f"Starting Stage {stage} Training")
    print(f"{'='*80}\n")

    # Loss functions
    reg_loss = nn.SmoothL1Loss(reduction="mean")
    ce_loss = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    ce_triplet = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    compl_loss = nn.SmoothL1Loss(beta=0.3)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),  # Higher beta2 for stability (match working reference model)
        eps=1e-8,
    )

    # Learning rate scheduler
    if USE_WARMUP:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS
        )
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - WARMUP_EPOCHS, eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[WARMUP_EPOCHS],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

    best_val_metric = 0.0 if stage == 2 else float("inf")
    print("Stage {0} training for {1} epochs - started.".format(stage, epochs))
    for epoch in range(1, epochs + 1):
        model.train()

        epoch_loss_total = 0.0
        epoch_loss_reg = 0.0
        epoch_loss_ce = 0.0
        epoch_loss_compl = 0.0
        epoch_loss_triplet = 0.0

        epoch_mae = 0.0
        epoch_acc = 0.0
        epoch_cmae = 0.0
        epoch_triplet_acc = 0.0
        epoch_triplet_samples = 0

        seen = 0
        t0 = time.time()

        for it, (frames, meta) in enumerate(train_loader, start=1):
            frames = frames.to(device)

            ttnp = meta["time_to_next_phase"].to(device).float()
            ttnp = torch.clamp(ttnp, min=0.0, max=TIME_HORIZON)
            labels = meta["phase_label"].to(device).long()
            completion_gt = meta["phase_completition"].to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            reg, logits, completion_pred, extras = model(frames, meta)

            # Triplet loss (if enabled)
            triplet_loss_value = 0.0
            has_triplet_loss = False

            if compute_triplets and "triplet_left_classification" in meta:
                left_targets = meta["triplet_left_classification"].to(device).long()
                right_targets = meta["triplet_right_classification"].to(device).long()

                left_loss, right_loss, triplet_metrics, num_valid = (
                    compute_triplet_loss_with_masking(
                        extras, left_targets, right_targets, ce_triplet
                    )
                )

                if num_valid > 0:
                    # Each arm loss contains both verb + target losses already combined
                    # Apply equal weighting to both arms (left + right)
                    loss_triplet_total = left_loss + right_loss
                    loss = loss_triplet_total  # Start with triplet loss
                    has_triplet_loss = True
                    triplet_loss_value = float(loss_triplet_total.item())

                    triplet_acc = (
                        triplet_metrics["left_verb_acc"]
                        + triplet_metrics["left_target_acc"]
                        + triplet_metrics["right_verb_acc"]
                        + triplet_metrics["right_target_acc"]
                    ) / 4.0
                    epoch_triplet_acc += triplet_acc * num_valid
                    epoch_triplet_samples += num_valid

            # Add phase losses (only if needed for regularization or if no triplet loss)
            if (
                not has_triplet_loss
                or task_weights.get("anticipation", 0) > 0
                or task_weights.get("phase", 0) > 0
            ):
                loss_reg = reg_loss(reg, ttnp)
                loss_ce = ce_loss(logits, labels)
                loss_compl = compl_loss(completion_pred, completion_gt)

                phase_loss = (
                    task_weights["anticipation"] * loss_reg
                    + task_weights["phase"] * loss_ce
                    + task_weights["completion"] * loss_compl
                )

                if has_triplet_loss:
                    loss = loss + phase_loss  # Add phase regularization
                else:
                    loss = phase_loss  # Only phase loss (skip this batch if stage 2)

            # Skip backward if no valid loss in stage 2
            if compute_triplets and not has_triplet_loss:
                # No triplet annotations in this batch during stage 2
                # Skip this batch since we're only training triplet heads
                continue

            loss.backward()

            if GRADIENT_CLIP_VAL > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)

            optimizer.step()

            bs = frames.size(0)
            epoch_loss_total += float(loss.item()) * bs
            epoch_loss_triplet += triplet_loss_value * bs

            # Only accumulate phase metrics if we computed them
            if (
                not has_triplet_loss
                or task_weights.get("anticipation", 0) > 0
                or task_weights.get("phase", 0) > 0
            ):
                epoch_loss_reg += float(loss_reg.item()) * bs
                epoch_loss_ce += float(loss_ce.item()) * bs
                epoch_loss_compl += float(loss_compl.item()) * bs

                with torch.no_grad():
                    preds = logits.argmax(dim=1)
                    acc = (preds == labels).float().mean()
                    mae = torch.abs(reg - ttnp).mean()
                    cmae = torch.abs(completion_pred - completion_gt).mean()

                    epoch_mae += float(mae.item()) * bs
                    epoch_acc += float(acc.item()) * bs
                    epoch_cmae += float(cmae.item()) * bs

            seen += bs

            if it % PRINT_EVERY == 0:
                # Only print phase metrics if they were computed
                if (
                    has_triplet_loss
                    and task_weights.get("anticipation", 0) == 0
                    and task_weights.get("phase", 0) == 0
                ):
                    # Stage 2 with pure triplet loss
                    print(
                        f"[Ep {epoch}/{epochs} it {it}] "
                        f"loss={loss.item():.4f} triplet_only"
                    )
                else:
                    # Stage 1 or Stage 2 with phase losses
                    print(
                        f"[Ep {epoch}/{epochs} it {it}] "
                        f"loss={loss.item():.4f} mae={mae.item():.4f} acc={acc.item():.3f}"
                    )

        scheduler.step()

        # Epoch summary
        N = max(1, seen)
        N_triplet = max(1, epoch_triplet_samples)

        summary = (
            f"\nEpoch {epoch:02d} [{time.time()-t0:.1f}s] LR={optimizer.param_groups[0]['lr']:.2e} "
            f"| loss={epoch_loss_total/N:.4f} mae={epoch_mae/N:.4f} "
            f"acc={epoch_acc/N:.4f} compl_mae={epoch_cmae/N:.4f}"
        )
        if compute_triplets and epoch_triplet_samples > 0:
            summary += f" | triplet_acc={epoch_triplet_acc/N_triplet:.3f}"
        print(summary)

        # Validation
        val_stats = evaluate(
            model, val_loader, device, TIME_HORIZON, compute_triplets=compute_triplets
        )

        # Stage 2: Only show triplet metrics (val_mae/val_acc not significant)
        if stage == 2 and compute_triplets and val_stats.get("triplet_samples", 0) > 0:
            val_summary = (
                f"           val_triplet_acc={val_stats['mean_triplet_acc']:.3f}"
            )

            # Print comprehensive AP metrics for stage 2
            if "triplet_ap_i" in val_stats:
                print(val_summary)
                print(
                    f"           AP Metrics: AP_i={val_stats['triplet_ap_i']:.4f} AP_v={val_stats['triplet_ap_v']:.4f} "
                    f"AP_t={val_stats['triplet_ap_t']:.4f}"
                )
                print(
                    f"                       AP_iv={val_stats['triplet_ap_iv']:.4f} AP_it={val_stats['triplet_ap_it']:.4f} "
                    f"AP_ivt={val_stats['triplet_ap_ivt']:.4f}"
                )
            else:
                print(val_summary)
        else:
            # Stage 1: Show all metrics
            val_summary = (
                f"           val_mae={val_stats['mae']:.4f} val_acc={val_stats['acc']:.4f} "
                f"val_compl_mae={val_stats['compl_mae']:.4f}"
            )
            if compute_triplets and val_stats.get("triplet_samples", 0) > 0:
                val_summary += f" | val_triplet_acc={val_stats['mean_triplet_acc']:.3f}"
            print(val_summary)

        # Log metrics to plotter
        if plotter is not None:
            # Training metrics
            log_dict = {
                f"stage{stage}/epoch": epoch,
                f"stage{stage}/train/loss": epoch_loss_total / N,
                f"stage{stage}/train/mae": epoch_mae / N,
                f"stage{stage}/train/acc": epoch_acc / N,
                f"stage{stage}/train/compl_mae": epoch_cmae / N,
                f"stage{stage}/lr": optimizer.param_groups[0]["lr"],
            }

            # Validation metrics
            log_dict.update(
                {
                    f"stage{stage}/val/mae": val_stats["mae"],
                    f"stage{stage}/val/acc": val_stats["acc"],
                    f"stage{stage}/val/compl_mae": val_stats["compl_mae"],
                }
            )

            # Stage 1: Add detailed anticipation metrics
            if stage == 1 and "inMAE" in val_stats:
                log_dict.update(
                    {
                        f"stage{stage}/val/inMAE": val_stats["inMAE"],
                        f"stage{stage}/val/oMAE": val_stats["oMAE"],
                        f"stage{stage}/val/wMAE": val_stats["wMAE"],
                        f"stage{stage}/val/eMAE": val_stats["eMAE"],
                    }
                )

            # Triplet metrics (if applicable)
            if compute_triplets and epoch_triplet_samples > 0:
                log_dict[f"stage{stage}/train/triplet_acc"] = (
                    epoch_triplet_acc / N_triplet
                )

            if compute_triplets and val_stats.get("triplet_samples", 0) > 0:
                log_dict.update(
                    {
                        f"stage{stage}/val/triplet_acc": val_stats["mean_triplet_acc"],
                        f"stage{stage}/val/left_verb_acc": val_stats["left_verb_acc"],
                        f"stage{stage}/val/left_target_acc": val_stats[
                            "left_target_acc"
                        ],
                        f"stage{stage}/val/right_verb_acc": val_stats["right_verb_acc"],
                        f"stage{stage}/val/right_target_acc": val_stats[
                            "right_target_acc"
                        ],
                    }
                )

                # Stage 2: Add comprehensive AP metrics
                if stage == 2:
                    if "triplet_ap_i" in val_stats:
                        log_dict.update(
                            {
                                f"stage{stage}/val/triplet_ap_i": val_stats[
                                    "triplet_ap_i"
                                ],
                                f"stage{stage}/val/triplet_ap_v": val_stats[
                                    "triplet_ap_v"
                                ],
                                f"stage{stage}/val/triplet_ap_t": val_stats[
                                    "triplet_ap_t"
                                ],
                                f"stage{stage}/val/triplet_ap_iv": val_stats[
                                    "triplet_ap_iv"
                                ],
                                f"stage{stage}/val/triplet_ap_it": val_stats[
                                    "triplet_ap_it"
                                ],
                                f"stage{stage}/val/triplet_ap_ivt": val_stats[
                                    "triplet_ap_ivt"
                                ],
                            }
                        )
                    # Legacy mAP metrics
                    if "triplet_verb_map" in val_stats:
                        log_dict.update(
                            {
                                f"stage{stage}/val/triplet_verb_map": val_stats[
                                    "triplet_verb_map"
                                ],
                                f"stage{stage}/val/triplet_target_map": val_stats[
                                    "triplet_target_map"
                                ],
                                f"stage{stage}/val/triplet_overall_map": val_stats[
                                    "triplet_overall_map"
                                ],
                            }
                        )

            plotter.log(log_dict, step=epoch)

        # Update best metrics tracker
        if best_metrics_tracker is not None:
            improved = best_metrics_tracker.update(stage, val_stats)
            if improved:
                improved_str = ", ".join(improved.keys())
                print(f"🌟 New best: {improved_str}")

        # Save best model
        is_best = False
        if stage == 2:
            # Stage 2: maximize triplet accuracy
            if val_stats.get("mean_triplet_acc", 0.0) > best_val_metric:
                best_val_metric = val_stats["mean_triplet_acc"]
                is_best = True
        else:
            # Stage 1: minimize anticipation MAE
            if val_stats["mae"] < best_val_metric:
                best_val_metric = val_stats["mae"]
                is_best = True

        if is_best:
            torch.save(model.state_dict(), ckpt_path)
            metric_name = "triplet_acc" if stage == 2 else "MAE"
            print(
                f"💾 Saved best model (val_{metric_name}={best_val_metric:.4f}) -> {ckpt_path}"
            )

        # Save last checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_metric": best_val_metric,
        }
        torch.save(ckpt, last_ckpt_path)

        # Generate plots and log best metrics every 50 epochs
        if epoch % 50 == 0 and epoch > 0:
            if plotter is not None:
                print(f"📊 Generating intermediate plots (epoch {epoch})...")
                plotter.plot_all()

            if best_metrics_tracker is not None:
                try:
                    print(f"\n📋 Best Metrics Summary (Epoch {epoch}):")
                    if stage == 1:
                        s1 = best_metrics_tracker.best_metrics.get("stage1", {})
                        print(
                            f"  Best Val MAE:  {s1.get('best_val_mae', float('inf')):.4f}"
                        )
                        print(f"  Best Val Acc:  {s1.get('best_val_acc', 0.0):.4f}")
                        print(
                            f"  Best inMAE:    {s1.get('best_inMAE', float('inf')):.4f}"
                        )
                        print(
                            f"  Best wMAE:     {s1.get('best_wMAE', float('inf')):.4f}"
                        )
                    elif stage == 2:
                        s2 = best_metrics_tracker.best_metrics.get("stage2", {})
                        print(f"  Best AP_i:     {s2.get('best_ap_i', 0.0):.4f}")
                        print(f"  Best AP_v:     {s2.get('best_ap_v', 0.0):.4f}")
                        print(f"  Best AP_t:     {s2.get('best_ap_t', 0.0):.4f}")
                        print(f"  Best AP_iv:    {s2.get('best_ap_iv', 0.0):.4f}")
                        print(f"  Best AP_it:    {s2.get('best_ap_it', 0.0):.4f}")
                        print(f"  Best AP_ivt:   {s2.get('best_ap_ivt', 0.0):.4f}")
                except Exception as e:
                    print(f"⚠️  Error displaying best metrics: {e}")

    print(f"\n🎉 Stage {stage} training completed!")

    # Generate final plots at the end of training
    if plotter is not None:
        print(f"📊 Generating final training plots...")
        plotter.plot_all()

    return ckpt_path


def main():
    print("🚀 Starting Dual-Scale Multi-Task Training")
    print(f"Long-term sequence: {SEQ_LEN_LONG} frames (phase/anticipation)")
    print(f"Short-term sequence: {SEQ_LEN_SHORT} frames (triplets) - INCREASED from 4")
    print(f"Two-stage training: {TWO_STAGE_TRAINING}")
    print(f"Batch sizes: stage1={BATCH_SIZE_STAGE1}, stage2={BATCH_SIZE_STAGE2} (stage2 reduced to avoid OOM)")
    print(f"Dropout: {DROPOUT_RATE} (increased from 0.15)")
    print(f"Stage 2 LR: {STAGE2_LR:.2e} (increased from 5e-5)")
    print(
        f"Stage 2 unfrozen: backbone={not FREEZE_BACKBONE_STAGE2}, temporal={not FREEZE_LONG_TEMPORAL_STAGE2}"
    )
    print("")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Create datasets
    print("📁 Loading datasets...")

    # Stage 1 datasets (ALL data)
    train_ds_stage1 = PegAndRing(
        ROOT_DIR,
        mode="train",
        seq_len=SEQ_LEN_LONG,  # Use long sequence for all data
        stride=STRIDE,
        time_unit=TIME_UNIT,
        augment=False,
        force_triplets=False,
    )

    val_ds_stage1 = PegAndRing(
        ROOT_DIR,
        mode="val",
        seq_len=SEQ_LEN_LONG,
        stride=STRIDE,
        time_unit=TIME_UNIT,
        augment=False,
        force_triplets=False,
    )

    print(f"Stage 1 - Train: {len(train_ds_stage1)}, Val: {len(val_ds_stage1)}")

    # Stage 2 datasets (triplet-annotated only)
    train_ds_stage2 = PegAndRing(
        ROOT_DIR,
        mode="train",
        seq_len=SEQ_LEN_LONG,  # Model handles both scales internally
        stride=STRIDE,
        time_unit=TIME_UNIT,
        augment=False,
        force_triplets=True,
    )

    val_ds_stage2 = PegAndRing(
        ROOT_DIR,
        mode="val",
        seq_len=SEQ_LEN_LONG,
        stride=STRIDE,
        time_unit=TIME_UNIT,
        augment=False,
        force_triplets=True,
    )

    print(f"Stage 2 - Train: {len(train_ds_stage2)}, Val: {len(val_ds_stage2)}\n")

    # Create data loaders
    print("📊 Creating data loaders...")

    gen_train_s1 = torch.Generator()
    gen_train_s1.manual_seed(SEED)
    train_batch_sampler_s1 = VideoBatchSampler(
        train_ds_stage1,
        batch_size=BATCH_SIZE_STAGE1,
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
        batch_size=BATCH_SIZE_STAGE1,
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

    gen_train_s2 = torch.Generator()
    gen_train_s2.manual_seed(SEED)
    train_batch_sampler_s2 = VideoBatchSampler(
        train_ds_stage2,
        batch_size=BATCH_SIZE_STAGE2,
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
        batch_size=BATCH_SIZE_STAGE2,
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

    # Create model
    print("\n🏗️  Building dual-scale multi-task model...")
    model = DualScaleMultiTaskModel(
        sequence_length_long=SEQ_LEN_LONG,
        sequence_length_short=SEQ_LEN_SHORT,
        num_classes=NUM_CLASSES,
        time_horizon=TIME_HORIZON,
        backbone="resnet50",  # Use ResNet18 to prevent OOM (ResNet50 too large for batch_size=16+seq_len=16)
        pretrained_backbone=True,
        freeze_backbone=False,
        hidden_channels=256,
        num_temporal_layers_long=4,
        num_temporal_layers_short=3,
        dropout=DROPOUT_RATE,
        use_spatial_attention=True,
        attn_heads=8,
        num_verbs=NUM_VERBS,
        num_targets=NUM_TARGETS,
        triplet_hidden_dim=256,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    if not DO_TRAIN:
        print("Training disabled (DO_TRAIN=False)")
        return

    # Create plotters for each stage
    plotter_stage1 = TrainingPlotter(PLOTS_DIR / "stage1", stage=1)
    plotter_stage2 = TrainingPlotter(PLOTS_DIR / "stage2", stage=2)

    # Create best metrics tracker
    best_metrics_tracker = BestMetricsTracker(EVAL_ROOT)
    print(f"📊 Best metrics tracker initialized: {best_metrics_tracker.filepath}")

    # Check if Stage 1 checkpoint exists
    stage1_exists = CKPT_PATH_STAGE1.exists()

    if stage1_exists:
        print("\n" + "=" * 80)
        print("✅ Stage 1 checkpoint found - skipping Stage 1 training")
        print(f"📁 Loading from: {CKPT_PATH_STAGE1}")
        print("=" * 80)
    else:
        # STAGE 1: Phase/Anticipation Pre-training
        print("\n" + "=" * 80)
        print("STAGE 1: Phase Recognition & Anticipation Pre-training")
        print("=" * 80)

        stage1_weights = {
            "anticipation": STAGE1_WEIGHT_ANTICIPATION,
            "phase": STAGE1_WEIGHT_PHASE,
            "completion": STAGE1_WEIGHT_COMPLETION,
        }

        train_stage(
            model=model,
            train_loader=train_loader_stage1,
            val_loader=val_loader_stage1,
            device=device,
            stage=1,
            epochs=STAGE1_EPOCHS,
            lr=STAGE1_LR,
            weight_decay=STAGE1_WEIGHT_DECAY,
            task_weights=stage1_weights,
            compute_triplets=False,
            ckpt_path=CKPT_PATH_STAGE1,
            last_ckpt_path=LAST_CKPT_PATH_STAGE1,
            plotter=plotter_stage1,
            best_metrics_tracker=best_metrics_tracker,
        )

    # STAGE 2: Triplet Fine-tuning
    print("\n" + "=" * 80)
    print("STAGE 2: Triplet Recognition Fine-tuning")
    print("=" * 80)

    # Load best stage 1 model
    print(f"📁 Loading best Stage 1 model from {CKPT_PATH_STAGE1}")
    model.load_state_dict(torch.load(CKPT_PATH_STAGE1, map_location=device))

    # Freeze components
    freeze_model_components(model, FREEZE_BACKBONE_STAGE2, FREEZE_LONG_TEMPORAL_STAGE2)

    stage2_weights = {
        "anticipation": STAGE2_WEIGHT_ANTICIPATION,
        "phase": STAGE2_WEIGHT_PHASE,
        "completion": STAGE2_WEIGHT_COMPLETION,
        "triplet_verb": STAGE2_WEIGHT_TRIPLET_VERB,
        "triplet_target": STAGE2_WEIGHT_TRIPLET_TARGET,
    }

    train_stage(
        model=model,
        train_loader=train_loader_stage2,
        val_loader=val_loader_stage2,
        device=device,
        stage=2,
        epochs=STAGE2_EPOCHS,
        lr=STAGE2_LR,
        weight_decay=STAGE2_WEIGHT_DECAY,
        task_weights=stage2_weights,
        compute_triplets=True,
        ckpt_path=CKPT_PATH_STAGE2,
        last_ckpt_path=LAST_CKPT_PATH_STAGE2,
        plotter=plotter_stage2,
        best_metrics_tracker=best_metrics_tracker,
    )

    print("\n" + "=" * 80)
    print("🎉 All training completed!")
    print(f"Best Stage 1 model: {CKPT_PATH_STAGE1}")
    print(f"Best Stage 2 model: {CKPT_PATH_STAGE2}")
    print(f"📊 Training plots saved to: {PLOTS_DIR}")

    # Print best metrics summary and save LaTeX table
    if best_metrics_tracker is not None:
        best_metrics_tracker.print_summary()
        best_metrics_tracker.save_latex_table()

    print("=" * 80)


def fps_test_run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Running FPS test on device: {device}\n")
    model = DualScaleMultiTaskModel(
        sequence_length_long=SEQ_LEN_LONG,
        sequence_length_short=SEQ_LEN_SHORT,
        num_classes=NUM_CLASSES,
        time_horizon=TIME_HORIZON,
        backbone="resnet50",
        pretrained_backbone=True,
        freeze_backbone=False,
        hidden_channels=256,
        num_temporal_layers_long=4,
        num_temporal_layers_short=3,
        dropout=DROPOUT_RATE,
        use_spatial_attention=True,
        attn_heads=8,
        num_verbs=NUM_VERBS,
        num_targets=NUM_TARGETS,
        triplet_hidden_dim=256,
    ).to(device)

    warmup = 10
    iterations = 100
    batch_size = 1
    for _ in range(warmup):
        inputs = torch.randn(batch_size, SEQ_LEN_LONG, 3, 224, 224).to(device)
        _ = model(inputs, meta={})

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iterations):
        inputs = torch.randn(batch_size, SEQ_LEN_LONG, 3, 224, 224).to(device)
        _ = model(inputs, meta={})

    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = iterations * batch_size / elapsed_time
    print(
        f"⏱️ FPS: {fps:.2f} - Latency (ms) per batch: {1000 * elapsed_time / (iterations * batch_size):.2f} ms\n"
    )


if __name__ == "__main__":
    fps_test_run()
    main()
