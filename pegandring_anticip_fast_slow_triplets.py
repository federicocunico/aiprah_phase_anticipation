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
import wandb
from typing import NamedTuple

from datasets.peg_and_ring_workflow import PegAndRing, VideoBatchSampler, TRIPLET_VERBS, TRIPLET_SUBJECTS, TRIPLET_OBJECTS
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


class MultiTaskLossWeighter(nn.Module):
    """
    Uncertainty-based multi-task loss weighting from:
    'Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics'
    """
    def __init__(self, num_tasks: int):
        super().__init__()
        # Log-variance parameters for each task
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        losses: [num_tasks] tensor of individual task losses
        Returns weighted total loss
        """
        # Uncertainty weighting: loss / (2 * sigma^2) + log(sigma)
        # where sigma^2 = exp(log_var)
        weights = torch.exp(-self.log_vars)
        weighted_losses = weights * losses + 0.5 * self.log_vars
        return weighted_losses.sum()
    
    def get_weights(self) -> torch.Tensor:
        """Return current task weights for logging"""
        return torch.exp(-self.log_vars)


class TaskOutputs(NamedTuple):
    """
    Structured outputs for multi-task model
    """
    anticipation: torch.Tensor
    phase_logits: torch.Tensor 
    completion: torch.Tensor
    triplet_verb_logits: torch.Tensor
    triplet_subject_logits: torch.Tensor
    triplet_destination_logits: torch.Tensor


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

        # ---- Separate Task-Specific Encoders ----
        # Phase anticipation encoder (main task)
        self.phase_temporal_encoder = TemporalConformerEncoder(
            d_model=hidden_channels,
            num_layers=num_temporal_layers,
            num_heads=attn_heads,
            dropout=dropout,
        )
        
        # Triplet classification encoder (auxiliary task) - more capacity for complex task
        self.triplet_temporal_encoder = TemporalConformerEncoder(
            d_model=hidden_channels,
            num_layers=num_temporal_layers + 1,  # More layers for complex triplet task
            num_heads=attn_heads,
            dropout=dropout * 0.5,  # Less dropout for harder task
        )
        
        # Cross-task attention for knowledge transfer
        self.cross_task_attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=attn_heads // 2,
            dropout=dropout,
            batch_first=True,
        )
        
        # Triplet-specific attention pooling (look at entire sequence)
        self.triplet_attention_pool = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.triplet_query = nn.Parameter(
            torch.randn(1, 1, hidden_channels) * 0.02
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

        # ---- Enhanced Triplet Classification Heads ----
        # Hierarchical triplet processing with shared and specific components
        
        # Shared triplet feature processor
        self.triplet_shared_processor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        
        # Verb head (4 classes - relatively simple)
        self.triplet_verb_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels // 2, NUM_VERBS),
        )
        
        # Subject head (3 classes - simple)
        self.triplet_subject_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels // 2, NUM_SUBJECTS),
        )
        
        # Destination head (14 classes - complex, needs much more capacity)
        self.triplet_destination_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),  # Double capacity
            nn.LayerNorm(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),  # Less dropout
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(hidden_channels // 2, NUM_OBJECTS),
        )
        
        # Triplet component interaction modeling
        self.triplet_interaction_attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=6,
            dropout=dropout * 0.5,
            batch_first=True,
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
            self.triplet_verb_head,
            self.triplet_subject_head,
            self.triplet_destination_head,
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
    ) -> TaskOutputs:
        B, T, C_in, H, W = frames.shape
        assert T == self.T, f"Expected sequence length {self.T}, got {T}"

        # ---- Shared Feature Extraction ----
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

        # Align + fuse shared features
        ts_up = F.interpolate(ts, size=T, mode="linear", align_corners=False)
        shared_features = self.fusion_gate_fast(tf, ts_up)  # [B, C, T]
        shared_seq = shared_features.transpose(1, 2)  # [B, T, C]

        # ---- Task-Specific Processing ----
        # Phase anticipation branch
        phase_encoded = self.phase_temporal_encoder(shared_seq)
        phase_current = phase_encoded[:, -1, :]  # Current timestep for classification
        
        # Triplet classification branch (separate encoder)
        triplet_encoded = self.triplet_temporal_encoder(shared_seq)
        
        # Cross-task knowledge transfer (optional)
        # Let triplet branch learn from phase branch
        triplet_enhanced, _ = self.cross_task_attention(
            query=triplet_encoded,
            key=phase_encoded,
            value=phase_encoded
        )
        triplet_encoded = triplet_encoded + 0.1 * triplet_enhanced  # Residual connection
        
        # ---- Phase Task Outputs ----
        phase_logits = self.phase_head(phase_current)
        completion = self.completion_head(phase_current)

        # Phase anticipation via attention pooling
        query = self.anticipation_query.expand(B, -1, -1)
        pooled, _ = self.temporal_attention_pool(query=query, key=phase_encoded, value=phase_encoded)
        pooled = pooled.squeeze(1)
        raw = self.anticipation_head(pooled)

        if self.sigmoid_scale != 1.0:
            y = self.H * torch.sigmoid(self.sigmoid_scale * raw)
        else:
            y = self.H * torch.sigmoid(raw)

        m = self._softmin(y, dim=1, tau=self.softmin_tau)
        y = y - m
        anticipation = F.softplus(y, beta=self.floor_beta)

        # ---- Triplet Task Outputs ----
        # Global context + current state for triplet reasoning
        triplet_query = self.triplet_query.expand(B, -1, -1)
        triplet_global, _ = self.triplet_attention_pool(
            query=triplet_query, key=triplet_encoded, value=triplet_encoded
        )
        triplet_global = triplet_global.squeeze(1)  # [B, C]
        triplet_current = triplet_encoded[:, -1, :]   # [B, C]
        
        # Multi-scale triplet reasoning
        triplet_contexts = torch.stack([triplet_global, triplet_current], dim=1)  # [B, 2, C]
        triplet_refined, _ = self.triplet_interaction_attention(
            query=triplet_contexts, key=triplet_contexts, value=triplet_contexts
        )
        
        # Combine global and local triplet understanding
        triplet_features = triplet_refined.mean(dim=1)  # [B, C]
        triplet_features = self.triplet_shared_processor(triplet_features)
        
        # Individual triplet component predictions
        triplet_verb_logits = self.triplet_verb_head(triplet_features)
        triplet_subject_logits = self.triplet_subject_head(triplet_features) 
        triplet_destination_logits = self.triplet_destination_head(triplet_features)

        return TaskOutputs(
            anticipation=anticipation,
            phase_logits=phase_logits,
            completion=completion,
            triplet_verb_logits=triplet_verb_logits,
            triplet_subject_logits=triplet_subject_logits,
            triplet_destination_logits=triplet_destination_logits,
        )


# ---------------------------
# Fixed training hyperparams
# ---------------------------
SEED = 42
ROOT_DIR = "data/peg_and_ring_workflow"
TIME_UNIT = "minutes"
NUM_CLASSES = 6

# Triplet classification dimensions - dynamically imported from dataset
NUM_VERBS = len(TRIPLET_VERBS)      # dynamically: reach, grasp, release, null
NUM_SUBJECTS = len(TRIPLET_SUBJECTS)   # dynamically: left_arm, right_arm, null  
NUM_OBJECTS = len(TRIPLET_OBJECTS)   # dynamically: 5 pegs + 4 rings + center + outside + 2 arms + null

SEQ_LEN = 16
STRIDE = 1

BATCH_SIZE = 12  # Reduced physical batch size
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 12 * 2 = 24
USE_GRADIENT_ACCUMULATION = False  # Enable/disable gradient accumulation
NUM_WORKERS = 30
EPOCHS = 200

LR = 1e-4
WEIGHT_DECAY = 2e-4

TIME_HORIZON = 2.0

PRINT_EVERY = 40
CKPT_PATH = Path("peg_and_ring_slowfast_transformer_triplets.pth")
LAST_CKPT_PATH = Path("peg_and_ring_slowfast_transformer_triplets_last.pth")  # resume point

# control flags
DO_TRAIN = True
RESUME_IF_CRASH = True
USE_WANDB = True  # Enable Weights & Biases logging
WANDB_PROJECT = "peg_ring_slowfast_triplets_v2"  # wandb project name
WANDB_RUN_NAME = None  # Auto-generated if None

# Multi-task learning configuration
USE_DYNAMIC_LOSS_WEIGHTING = True  # Use uncertainty-based loss weighting
USE_SEPARATE_TASK_ENCODERS = True  # Use separate encoders for different tasks
TRIPLET_ENCODER_EXTRA_LAYERS = 1   # Extra layers for triplet encoder
CROSS_TASK_ATTENTION_WEIGHT = 0.1  # Weight for cross-task knowledge transfer

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

    # Phase anticipation and classification metrics
    total_mae = 0.0
    total_ce = 0.0
    total_acc = 0.0
    total_cmae = 0.0
    
    # Triplet classification metrics  
    total_triplet_verb_acc = 0.0
    total_triplet_subject_acc = 0.0
    total_triplet_destination_acc = 0.0
    total_triplet_complete_acc = 0.0  # All three components correct
    total_triplet_ce = 0.0
    
    total_samples = 0

    ce_loss = nn.CrossEntropyLoss()

    for _, (frames, meta) in enumerate(loader):
        frames = frames.to(device)
        ttnp = meta["time_to_next_phase"].to(device).float()
        ttnp = torch.clamp(ttnp, min=0.0, max=time_horizon)
        labels = meta["phase_label"].to(device).long()
        completion_gt = meta["phase_completition"].to(device).float().unsqueeze(1)
        
        # Triplet ground truth labels
        triplet_gt = meta["triplet_classification"].to(device).long()  # [B, 3]
        triplet_verb_gt = triplet_gt[:, 0]     # [B]
        triplet_subject_gt = triplet_gt[:, 1]  # [B]
        triplet_dest_gt = triplet_gt[:, 2]     # [B]

        outputs = model(frames, meta)

        # Phase metrics
        preds_cls = outputs.phase_logits.argmax(dim=1)
        acc = (preds_cls == labels).float().mean()
        mae = torch.mean(torch.abs(outputs.anticipation - ttnp))
        ce = ce_loss(outputs.phase_logits, labels)
        cmae = torch.mean(torch.abs(outputs.completion - completion_gt))

        # Triplet metrics
        triplet_verb_logits = outputs.triplet_verb_logits
        triplet_subject_logits = outputs.triplet_subject_logits
        triplet_dest_logits = outputs.triplet_destination_logits
        
        triplet_verb_pred = triplet_verb_logits.argmax(dim=1)
        triplet_subject_pred = triplet_subject_logits.argmax(dim=1)
        triplet_dest_pred = triplet_dest_logits.argmax(dim=1)
        
        # Individual component accuracies
        verb_acc = (triplet_verb_pred == triplet_verb_gt).float().mean()
        subject_acc = (triplet_subject_pred == triplet_subject_gt).float().mean()
        dest_acc = (triplet_dest_pred == triplet_dest_gt).float().mean()
        
        # Complete triplet accuracy (all three components correct)
        complete_correct = (
            (triplet_verb_pred == triplet_verb_gt) &
            (triplet_subject_pred == triplet_subject_gt) &
            (triplet_dest_pred == triplet_dest_gt)
        ).float().mean()
        
        # Triplet cross-entropy losses
        triplet_verb_ce = ce_loss(triplet_verb_logits, triplet_verb_gt)
        triplet_subject_ce = ce_loss(triplet_subject_logits, triplet_subject_gt)
        triplet_dest_ce = ce_loss(triplet_dest_logits, triplet_dest_gt)
        triplet_total_ce = (triplet_verb_ce + triplet_subject_ce + triplet_dest_ce) / 3.0

        bs = frames.size(0)
        total_mae += float(mae.item()) * bs
        total_ce += float(ce.item()) * bs
        total_acc += float(acc.item()) * bs
        total_cmae += float(cmae.item()) * bs
        
        total_triplet_verb_acc += float(verb_acc.item()) * bs
        total_triplet_subject_acc += float(subject_acc.item()) * bs
        total_triplet_destination_acc += float(dest_acc.item()) * bs
        total_triplet_complete_acc += float(complete_correct.item()) * bs
        total_triplet_ce += float(triplet_total_ce.item()) * bs
        
        total_samples += bs

    return {
        # Phase anticipation and classification
        "mae": total_mae / max(1, total_samples),
        "ce": total_ce / max(1, total_samples),
        "acc": total_acc / max(1, total_samples),
        "compl_mae": total_cmae / max(1, total_samples),
        
        # Triplet classification metrics
        "triplet_verb_acc": total_triplet_verb_acc / max(1, total_samples),
        "triplet_subject_acc": total_triplet_subject_acc / max(1, total_samples),
        "triplet_destination_acc": total_triplet_destination_acc / max(1, total_samples),
        "triplet_complete_acc": total_triplet_complete_acc / max(1, total_samples),
        "triplet_ce": total_triplet_ce / max(1, total_samples),
        
        "samples": total_samples,
    }


def log_to_wandb(metrics: Dict[str, Any], step: Optional[int] = None):
    """Log metrics to wandb with optional step."""
    if not USE_WANDB or not wandb.run:
        print(f"[WARN] Wandb not available: USE_WANDB={USE_WANDB}, wandb.run={wandb.run}")
        return
    
    log_dict = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            log_dict[key] = float(value)
    
    if log_dict:
        if step is not None:
            wandb.log(log_dict, step=step)
            # ] Logged {len(log_dict)} metrics to wandb at step {step}: {list(log_dict.keys())}")
        else:
            wandb.log(log_dict)
            # print(f"[DEBUG] Logged {len(log_dict)} metrics to wandb: {list(log_dict.keys())}")
    else:
        print(f"[WARN] No valid metrics to log: {metrics}")


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
        root_dir=root_dir, mode=split, seq_len=SEQ_LEN, stride=1, time_unit=time_unit, force_triplets=True
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
        outputs = model(frames, meta)

        pred_phase = outputs.phase_logits.argmax(dim=1).detach().cpu().numpy()
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
        root_dir=root_dir, mode=split, seq_len=SEQ_LEN, stride=1, time_unit=time_unit, force_triplets=True
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
        model_outputs = model(frames, meta)

        pred = torch.clamp(model_outputs.anticipation, min=0.0, max=time_horizon).detach().cpu().numpy()
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
    
    # Losses
    reg_loss = nn.SmoothL1Loss(reduction="mean")
    ce_loss = nn.CrossEntropyLoss()
    compl_loss = nn.SmoothL1Loss(beta=0.3)
    
    # Multi-task loss weighter
    loss_weighter = MultiTaskLossWeighter(num_tasks=6).to(device)  # reg, cls, compl, verb, subj, dest
    
    # Separate optimizers for main model and loss weighting
    model_optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    weighter_optimizer = optim.AdamW(loss_weighter.parameters(), lr=LR * 10, weight_decay=0)  # Higher LR for weighting
    
    # Separate schedulers
    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model_optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    weighter_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        weighter_optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    # Resume if requested
    start_epoch, best_val_mae = _try_resume(model, model_optimizer, model_scheduler, device)

    # Run initial validation before training starts
    if start_epoch == 1:
        print("Running initial validation before training...")
        val_stats_initial = evaluate(model, val_loader, device, TIME_HORIZON)
        print(f"Initial validation: mae={val_stats_initial['mae']:.4f} acc={val_stats_initial['acc']:.4f}")
        log_to_wandb({
            "val_mae_initial": val_stats_initial['mae'],
            "val_accuracy_initial": val_stats_initial['acc'],
            "val_triplet_complete_accuracy_initial": val_stats_initial['triplet_complete_acc'],
        }, step=0)

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_mae = epoch_ce = epoch_loss = epoch_acc = epoch_cmae = 0.0
        epoch_triplet_ce = epoch_triplet_verb_acc = epoch_triplet_subject_acc = 0.0
        epoch_triplet_dest_acc = epoch_triplet_complete_acc = 0.0
        seen = 0
        t0 = time.time()

        for it, (frames, meta) in enumerate(train_loader, start=1):
            frames = frames.to(device)
            labels = meta["phase_label"].to(device).long()
            complets_gt = meta["phase_completition"].to(device).float().unsqueeze(1)
            ttnp = torch.clamp(
                meta["time_to_next_phase"].to(device).float(), 0.0, TIME_HORIZON
            )
            
            # Triplet ground truth labels
            triplet_gt = meta["triplet_classification"].to(device).long()  # [B, 3]
            triplet_verb_gt = triplet_gt[:, 0]
            triplet_subject_gt = triplet_gt[:, 1]
            triplet_dest_gt = triplet_gt[:, 2]

            outputs = model(frames, meta)

            # Individual task losses
            loss_reg = reg_loss(outputs.anticipation, ttnp)
            loss_cls = ce_loss(outputs.phase_logits, labels)
            loss_compl = compl_loss(outputs.completion, complets_gt)
            loss_triplet_verb = ce_loss(outputs.triplet_verb_logits, triplet_verb_gt)
            loss_triplet_subject = ce_loss(outputs.triplet_subject_logits, triplet_subject_gt)
            loss_triplet_dest = ce_loss(outputs.triplet_destination_logits, triplet_dest_gt)
            
            # Create loss tensor for dynamic weighting
            task_losses = torch.stack([
                loss_reg, loss_cls, loss_compl, 
                loss_triplet_verb, loss_triplet_subject, loss_triplet_dest
            ])
            
            # Dynamic multi-task loss weighting
            loss_total = loss_weighter(task_losses)
            
            # Scale loss for gradient accumulation if enabled
            if USE_GRADIENT_ACCUMULATION:
                loss_total = loss_total / GRADIENT_ACCUMULATION_STEPS

            loss_total.backward()
            
            # Gradient accumulation: only step optimizer every N accumulation steps if enabled
            if USE_GRADIENT_ACCUMULATION:
                if it % GRADIENT_ACCUMULATION_STEPS == 0:
                    model_optimizer.step()
                    weighter_optimizer.step()
                    model_optimizer.zero_grad(set_to_none=True)
                    weighter_optimizer.zero_grad(set_to_none=True)
            else:
                # Normal training: step optimizer every iteration
                model_optimizer.step()
                weighter_optimizer.step()
                model_optimizer.zero_grad(set_to_none=True)
                weighter_optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                # Phase metrics
                pred_cls = outputs.phase_logits.argmax(dim=1)
                train_mae = torch.mean(torch.abs(outputs.anticipation - ttnp))
                train_acc = (pred_cls == labels).float().mean()
                train_cmae = torch.mean(torch.abs(outputs.completion - complets_gt))
                
                # Triplet metrics
                triplet_verb_pred = outputs.triplet_verb_logits.argmax(dim=1)
                triplet_subject_pred = outputs.triplet_subject_logits.argmax(dim=1)
                triplet_dest_pred = outputs.triplet_destination_logits.argmax(dim=1)
                
                # Combined triplet loss for logging
                loss_triplet = (loss_triplet_verb + loss_triplet_subject + loss_triplet_dest) / 3.0
                
                triplet_verb_acc = (triplet_verb_pred == triplet_verb_gt).float().mean()
                triplet_subject_acc = (triplet_subject_pred == triplet_subject_gt).float().mean()
                triplet_dest_acc = (triplet_dest_pred == triplet_dest_gt).float().mean()
                triplet_complete_acc = (
                    (triplet_verb_pred == triplet_verb_gt) &
                    (triplet_subject_pred == triplet_subject_gt) &
                    (triplet_dest_pred == triplet_dest_gt)
                ).float().mean()

            bs = frames.size(0)
            epoch_mae += float(train_mae.item()) * bs
            epoch_ce += float(loss_cls.item()) * bs
            # Unscale loss for reporting if gradient accumulation is used
            reported_loss = loss_total.item() * (GRADIENT_ACCUMULATION_STEPS if USE_GRADIENT_ACCUMULATION else 1)
            epoch_loss += float(reported_loss) * bs
            epoch_acc += float(train_acc.item()) * bs
            epoch_cmae += float(train_cmae.item()) * bs
            epoch_triplet_ce += float(loss_triplet.item()) * bs
            epoch_triplet_verb_acc += float(triplet_verb_acc.item()) * bs
            epoch_triplet_subject_acc += float(triplet_subject_acc.item()) * bs
            epoch_triplet_dest_acc += float(triplet_dest_acc.item()) * bs
            epoch_triplet_complete_acc += float(triplet_complete_acc.item()) * bs
            seen += bs

            if it % PRINT_EVERY == 0:
                cur_lr = model_scheduler.get_last_lr()[0]
                task_weights = loss_weighter.get_weights().detach().cpu().numpy()
                # Show unscaled loss for better interpretability
                unscaled_loss = loss_total.item() * (GRADIENT_ACCUMULATION_STEPS if USE_GRADIENT_ACCUMULATION else 1)
                print(
                    f"[Epoch {epoch:02d} | it {it:04d}] "
                    f"loss={unscaled_loss:.4f} | mae={train_mae.item():.4f} | "
                    f"acc={train_acc.item():.4f} | reg={loss_reg.item():.4f} | "
                    f"cls={loss_cls.item():.4f} | compl={loss_compl.item():.4f} | "
                    f"trip={loss_triplet.item():.4f} | trip_acc={triplet_complete_acc.item():.4f} | lr={cur_lr:.2e}"
                )
                print(f"           Task weights: reg={task_weights[0]:.3f}, cls={task_weights[1]:.3f}, compl={task_weights[2]:.3f}, "
                      f"verb={task_weights[3]:.3f}, subj={task_weights[4]:.3f}, dest={task_weights[5]:.3f}")

        if epoch % 5 == 0:
            gpu_mem_allocated = torch.cuda.memory_allocated() / 1e9
            gpu_mem_cached = torch.cuda.memory_reserved() / 1e9
            print(f"GPU mem: {gpu_mem_allocated:.4f}GB")
            print(f"GPU cached: {gpu_mem_cached:.4f}GB")
            
            # Log GPU memory usage to wandb
            log_to_wandb({
                "system_gpu_memory_allocated_gb": gpu_mem_allocated,
                "system_gpu_memory_cached_gb": gpu_mem_cached,
            }, step=epoch)

            # collect
            gc.collect()
            torch.cuda.empty_cache()

        # Final gradient step if there are remaining accumulated gradients and gradient accumulation is enabled
        if USE_GRADIENT_ACCUMULATION and len(train_loader) % GRADIENT_ACCUMULATION_STEPS != 0:
            model_optimizer.step()
            weighter_optimizer.step()
            model_optimizer.zero_grad(set_to_none=True)
            weighter_optimizer.zero_grad(set_to_none=True)
        
        # step schedulers once per epoch
        model_scheduler.step()
        weighter_scheduler.step()

        train_loss_avg = epoch_loss / max(1, seen)
        train_mae_avg = epoch_mae / max(1, seen)
        train_ce_avg = epoch_ce / max(1, seen)
        train_acc_avg = epoch_acc / max(1, seen)
        train_cmae_avg = epoch_cmae / max(1, seen)
        train_triplet_ce_avg = epoch_triplet_ce / max(1, seen)
        train_triplet_verb_acc_avg = epoch_triplet_verb_acc / max(1, seen)
        train_triplet_subject_acc_avg = epoch_triplet_subject_acc / max(1, seen)
        train_triplet_dest_acc_avg = epoch_triplet_dest_acc / max(1, seen)
        train_triplet_complete_acc_avg = epoch_triplet_complete_acc / max(1, seen)
        
        print(
            f"\nEpoch {epoch:02d} [{time.time()-t0:.1f}s] "
            f"| train_loss={train_loss_avg:.4f} train_mae={train_mae_avg:.4f} "
            f"train_ce={train_ce_avg:.4f} train_acc={train_acc_avg:.4f} "
            f"train_compl_mae={train_cmae_avg:.4f}"
        )
        print(
            f"           triplet: ce={train_triplet_ce_avg:.4f} "
            f"v_acc={train_triplet_verb_acc_avg:.4f} s_acc={train_triplet_subject_acc_avg:.4f} "
            f"d_acc={train_triplet_dest_acc_avg:.4f} complete_acc={train_triplet_complete_acc_avg:.4f}"
        )
        
        # Log epoch-level training metrics to wandb
        epoch_duration = time.time() - t0
        log_to_wandb({
            "epoch": epoch,
            "train_total_loss": train_loss_avg,
            "train_mae": train_mae_avg,
            "train_cross_entropy": train_ce_avg,
            "train_accuracy": train_acc_avg,
            "train_completion_mae": train_cmae_avg,
            "train_triplet_cross_entropy": train_triplet_ce_avg,
            "train_triplet_verb_accuracy": train_triplet_verb_acc_avg,
            "train_triplet_subject_accuracy": train_triplet_subject_acc_avg,
            "train_triplet_destination_accuracy": train_triplet_dest_acc_avg,
            "train_triplet_complete_accuracy": train_triplet_complete_acc_avg,
            "train_epoch_duration_seconds": epoch_duration,
            "train_learning_rate": model_scheduler.get_last_lr()[0],
            "train_samples": seen,
        }, step=epoch)

        # Validation
        val_stats = evaluate(model, val_loader, device, TIME_HORIZON)
        print(
            f"           val_mae={val_stats['mae']:.4f} val_ce={val_stats['ce']:.4f} "
            f"val_acc={val_stats['acc']:.4f} val_compl_mae={val_stats['compl_mae']:.4f} "
            f"| samples={val_stats['samples']}"
        )
        print(
            f"           val_triplet: ce={val_stats['triplet_ce']:.4f} "
            f"v_acc={val_stats['triplet_verb_acc']:.4f} s_acc={val_stats['triplet_subject_acc']:.4f} "
            f"d_acc={val_stats['triplet_destination_acc']:.4f} complete_acc={val_stats['triplet_complete_acc']:.4f}"
        )
        
        # Log validation metrics to wandb
        log_to_wandb({
            "val_mae": val_stats['mae'],
            "val_cross_entropy": val_stats['ce'],
            "val_accuracy": val_stats['acc'],
            "val_completion_mae": val_stats['compl_mae'],
            "val_triplet_cross_entropy": val_stats['triplet_ce'],
            "val_triplet_verb_accuracy": val_stats['triplet_verb_acc'],
            "val_triplet_subject_accuracy": val_stats['triplet_subject_acc'],
            "val_triplet_destination_accuracy": val_stats['triplet_destination_acc'],
            "val_triplet_complete_accuracy": val_stats['triplet_complete_acc'],
            "val_samples": val_stats['samples'],
        }, step=epoch)

        # Save best (unchanged)
        is_best = val_stats["mae"] < best_val_mae
        if is_best:
            best_val_mae = val_stats["mae"]
            torch.save(model.state_dict(), CKPT_PATH)
            print(
                f"✅  New best val_mae={best_val_mae:.4f} val_acc={val_stats['acc']:.4f} — saved to: {CKPT_PATH}"
            )
            
            # Log best model metrics to wandb
            log_to_wandb({
                "best_val_mae": best_val_mae,
                "best_val_accuracy": val_stats['acc'],
                "best_val_triplet_complete_accuracy": val_stats['triplet_complete_acc'],
                "best_epoch": epoch,
            }, step=epoch)

        # Always save last for resume (even if not best)
        _save_last_checkpoint(model, model_optimizer, model_scheduler, epoch, best_val_mae, LAST_CKPT_PATH)
        
        # Log task weights to wandb
        current_weights = loss_weighter.get_weights().detach().cpu().numpy()
        log_to_wandb({
            "task_weight_regression": current_weights[0],
            "task_weight_classification": current_weights[1],
            "task_weight_completion": current_weights[2],
            "task_weight_triplet_verb": current_weights[3],
            "task_weight_triplet_subject": current_weights[4],
            "task_weight_triplet_destination": current_weights[5],
        }, step=epoch)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    # Initialize wandb if enabled (before training starts)
    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "seed": SEED,
                "batch_size": BATCH_SIZE,
                "use_gradient_accumulation": USE_GRADIENT_ACCUMULATION,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS if USE_GRADIENT_ACCUMULATION else 1,
                "effective_batch_size": BATCH_SIZE * (GRADIENT_ACCUMULATION_STEPS if USE_GRADIENT_ACCUMULATION else 1),
                "num_workers": NUM_WORKERS,
                "epochs": EPOCHS,
                "learning_rate": LR,
                "weight_decay": WEIGHT_DECAY,
                "time_horizon": TIME_HORIZON,
                "seq_len": SEQ_LEN,
                "stride": STRIDE,
                "num_classes": NUM_CLASSES,
                "num_verbs": NUM_VERBS,
                "num_subjects": NUM_SUBJECTS,
                "num_objects": NUM_OBJECTS,
                "time_unit": TIME_UNIT,
                "model_hidden_channels": 384,
                "model_num_temporal_layers": 6,
                "model_dropout": 0.1,
                "model_attn_heads": 8,
                "backbone_fast": "resnet50",
                "backbone_slow": "resnet50",
                "architecture": "SlowFastTemporalAnticipation_v2",
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                # Multi-task learning config
                "use_dynamic_loss_weighting": USE_DYNAMIC_LOSS_WEIGHTING,
                "use_separate_task_encoders": USE_SEPARATE_TASK_ENCODERS,
                "triplet_encoder_extra_layers": TRIPLET_ENCODER_EXTRA_LAYERS,
                "cross_task_attention_weight": CROSS_TASK_ATTENTION_WEIGHT,
                "multitask_strategy": "uncertainty_weighting_separate_encoders",
            },
            tags=["slowfast", "triplets", "multi-task-v2", "phase-anticipation", "uncertainty-weighting", "separate-encoders"]
        )

    # Datasets - Use triplet-based splitting for multi-task learning
    train_ds = PegAndRing(
        ROOT_DIR,
        mode="train",
        seq_len=SEQ_LEN,
        stride=STRIDE,
        time_unit=TIME_UNIT,
        augment=True,
        force_triplets=True,  # Only videos with triplet annotations
    )
    val_ds = PegAndRing(
        ROOT_DIR,
        mode="val",
        seq_len=SEQ_LEN,
        stride=STRIDE,
        time_unit=TIME_UNIT,
        augment=False,
        force_triplets=True,  # Only videos with triplet annotations
    )
    test_ds = PegAndRing(
        ROOT_DIR,
        mode="val",
        seq_len=SEQ_LEN,
        stride=STRIDE,
        time_unit=TIME_UNIT,
        augment=False,
        force_triplets=True,  # Only videos with triplet annotations
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
    # Adjust print frequency for gradient accumulation if enabled
    if USE_GRADIENT_ACCUMULATION:
        PRINT_EVERY = (len(train_loader) // 5) * GRADIENT_ACCUMULATION_STEPS
    else:
        PRINT_EVERY = len(train_loader) // 5  # Print 5 times per epoch

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
    
    # Log model architecture to wandb if enabled
    if USE_WANDB and wandb.run:
        wandb.watch(model, log="all", log_freq=100)

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
    print(
        f"       triplet: ce={test_stats['triplet_ce']:.4f} "
        f"v_acc={test_stats['triplet_verb_acc']:.4f} s_acc={test_stats['triplet_subject_acc']:.4f} "
        f"d_acc={test_stats['triplet_destination_acc']:.4f} complete_acc={test_stats['triplet_complete_acc']:.4f}"
    )
    
    # Log final test results to wandb
    log_to_wandb({
        "test_mae": test_stats['mae'],
        "test_cross_entropy": test_stats['ce'],
        "test_accuracy": test_stats['acc'],
        "test_completion_mae": test_stats['compl_mae'],
        "test_triplet_cross_entropy": test_stats['triplet_ce'],
        "test_triplet_verb_accuracy": test_stats['triplet_verb_acc'],
        "test_triplet_subject_accuracy": test_stats['triplet_subject_acc'],
        "test_triplet_destination_accuracy": test_stats['triplet_destination_acc'],
        "test_triplet_complete_accuracy": test_stats['triplet_complete_acc'],
        "test_samples": test_stats['samples'],
    })
    
    # Create summary table for wandb
    if USE_WANDB and wandb.run:
        summary_table = wandb.Table(
            columns=["Metric", "Value"],
            data=[
                ["Test MAE", f"{test_stats['mae']:.4f}"],
                ["Test Accuracy", f"{test_stats['acc']:.4f}"],
                ["Test Triplet Complete Accuracy", f"{test_stats['triplet_complete_acc']:.4f}"],
                ["Test Triplet Verb Accuracy", f"{test_stats['triplet_verb_acc']:.4f}"],
                ["Test Triplet Subject Accuracy", f"{test_stats['triplet_subject_acc']:.4f}"],
                ["Test Triplet Destination Accuracy", f"{test_stats['triplet_destination_acc']:.4f}"],
                ["Model Parameters", f"{sum(p.numel() for p in model.parameters()):,}"],
                ["Trainable Parameters", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"],
            ]
        )
        wandb.log({"test_summary_table": summary_table})

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
    
    # Finish wandb run
    if USE_WANDB and wandb.run:
        wandb.finish()
        print("\n📊 Wandb run completed and logged.")


if __name__ == "__main__":
    main()
