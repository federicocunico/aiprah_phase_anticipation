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
from sklearn.metrics import average_precision_score, classification_report
import json
import time

from datasets.peg_and_ring_workflow import PegAndRing, VideoBatchSampler, TRIPLET_VERBS, TRIPLET_SUBJECTS, TRIPLET_TARGETS
import torch.nn.functional as F
import numpy as np

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


# MultiScaleTemporalCNN removed - replaced by EnhancedTemporalCNN


# SpatialAttentionPool removed - replaced by MultiScaleSpatialAttention


# GatedFusion removed - replaced by CrossModalAttentionFusion


class MultiScaleSpatialAttention(nn.Module):
    """Multi-scale spatial attention for richer spatial features"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.scales = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
        ])
        self.attention = nn.Conv2d(out_channels * 3, 1, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.out_proj = nn.Linear(out_channels * 3, out_channels * 4)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale = torch.cat([scale(x) for scale in self.scales], dim=1)
        attention = torch.sigmoid(self.attention(multi_scale))
        attended = multi_scale * attention
        pooled = self.global_pool(attended).flatten(1)
        return self.out_proj(pooled)


class CrossModalAttentionFusion(nn.Module):
    """Cross-modal attention between fast and slow pathways"""
    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=False
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels * 4, channels),
        )
        
    def forward(self, fast_features: torch.Tensor, slow_features: torch.Tensor) -> torch.Tensor:
        # fast_features, slow_features: [B, C, T]
        B, C, T = fast_features.shape
        
        # Transpose for attention: [B, C, T] -> [T, B, C]
        fast_seq = fast_features.permute(2, 0, 1)  # [T, B, C]
        slow_seq = slow_features.permute(2, 0, 1)  # [T, B, C]
        
        # Cross attention: fast attends to slow
        attended, _ = self.cross_attention(fast_seq, slow_seq, slow_seq)
        attended = self.norm1(attended + fast_seq)
        
        # FFN
        enhanced = self.ffn(attended)
        enhanced = self.norm2(enhanced + attended)
        
        # Transpose back: [T, B, C] -> [B, C, T]
        return enhanced.permute(1, 2, 0)


class EnhancedTemporalCNN(nn.Module):
    """Enhanced temporal CNN optimized for robotic action sequences"""
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, dropout: float):
        super().__init__()
        
        self.layers = nn.ModuleList()
        # Optimized dilations for 8-frame sequences and 2-4 second action changes
        # Use smaller dilations to capture fine-grained action boundaries
        dilations = [1, 2, 3, 4, 6, 8, 12, 16][:num_layers]  # Max dilation 16 for 8-frame context
        
        for i, dilation in enumerate(dilations):
            # Ensure dilation doesn't exceed sequence length
            effective_dilation = min(dilation, 8)  # Cap at sequence length
            
            layer = nn.Sequential(
                nn.Conv1d(
                    in_channels if i == 0 else hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    dilation=effective_dilation,
                    padding=effective_dilation,
                ),
                nn.BatchNorm1d(hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
                nn.BatchNorm1d(hidden_channels),
            )
            self.layers.append(layer)
            
        self.residual_projections = nn.ModuleList([
            nn.Conv1d(in_channels, hidden_channels, 1) if in_channels != hidden_channels else nn.Identity()
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (layer, proj) in enumerate(zip(self.layers, self.residual_projections)):
            residual = proj(x) if i == 0 else x
            x = F.gelu(layer(x) + residual)
        return x


class TransformerEncoder(nn.Module):
    """Enhanced Transformer encoder with action-aware temporal modeling"""
    def __init__(self, d_model: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,  # Pre-norm for better training
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
        # Action-aware positional encoding (emphasizes boundaries at 2-4 second intervals)
        self.action_pos_encoding = nn.Parameter(
            self._generate_action_aware_positions(d_model, max_len=8)
        )
        
    def _generate_action_aware_positions(self, d_model: int, max_len: int) -> torch.Tensor:
        """Generate positional encodings that emphasize action boundaries"""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        
        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        # Emphasize positions corresponding to typical action boundaries (every 2-4 frames)
        boundary_frames = [2, 4, 6]  # Typical action change points in 8-frame sequence
        for bf in boundary_frames:
            if bf < max_len:
                pos_encoding[bf] *= 1.5  # Amplify boundary positions
        
        return pos_encoding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add action-aware positional encoding
        B, T, D = x.shape
        if hasattr(self, 'action_pos_encoding'):
            pos_enc = self.action_pos_encoding[:T, :D].unsqueeze(0).expand(B, -1, -1)
            # Only add if dimensions match
            if pos_enc.shape == x.shape:
                x = x + pos_enc
            
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing for better generalization on small datasets"""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_target = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        return F.kl_div(F.log_softmax(pred, dim=-1), smooth_target, reduction='batchmean')


class MotionPriorExtractor(nn.Module):
    """Extract motion features as priors for object manipulation"""
    def __init__(self, channels: int):
        super().__init__()
        # Simple frame difference for motion estimation
        self.motion_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # RGB frame differences
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Motion-based attention
        self.motion_attention = nn.Sequential(
            nn.Linear(64, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )
        
    def compute_frame_differences(self, frames: torch.Tensor) -> torch.Tensor:
        """Compute frame differences optimized for action boundary detection"""
        B, T, C, H, W = frames.shape
        differences = []
        
        for t in range(T - 1):
            # Compute absolute difference between consecutive frames
            diff = torch.abs(frames[:, t + 1] - frames[:, t])  # [B, C, H, W]
            
            # Enhance action boundaries with gradient magnitude
            diff_magnitude = torch.norm(diff, dim=1, keepdim=True)  # [B, 1, H, W]
            enhanced_diff = diff * (1 + 0.5 * torch.sigmoid(diff_magnitude - 0.1))
            
            differences.append(enhanced_diff)
        
        return torch.stack(differences, dim=1)  # [B, T-1, C, H, W]
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract motion priors from frame sequence"""
        B, T, C, H, W = frames.shape
        
        # Compute frame differences as motion proxy
        differences = self.compute_frame_differences(frames)  # [B, T-1, C, H, W]
        
        # Process differences through CNN
        motion_features = []
        for t in range(T - 1):
            diff_t = differences[:, t]  # [B, C, H, W]
            motion_feat = self.motion_conv(diff_t)  # [B, 64, 1, 1]
            motion_features.append(motion_feat.flatten(1))  # [B, 64]
        
        # Average motion features across time
        motion_feat_avg = torch.stack(motion_features, dim=1).mean(dim=1)  # [B, 64]
        
        # Generate attention weights
        motion_attention = self.motion_attention(motion_feat_avg)  # [B, channels]
        
        return motion_attention


# Conformer-style classes removed - using simpler TransformerEncoder


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
class SlowFastTripletPredictorV2(nn.Module):
    """
    Enhanced Slow-Fast model designed for small datasets and complex triplet prediction.
    
    Key improvements:
    1. Larger model capacity with better regularization
    2. Motion-based priors for object manipulation
    3. Hierarchical triplet prediction (verb -> subject -> destination)
    4. Strong regularization techniques for small datasets
    5. Cross-modal attention fusion
    """

    def __init__(
        self,
        sequence_length: int,
        *,
        backbone_fast: str = "resnet50",
        backbone_slow: str = "resnet50",
        pretrained_backbone_fast: bool = True,
        pretrained_backbone_slow: bool = True,
        freeze_backbone_fast: bool = False,
        freeze_backbone_slow: bool = False,
        hidden_channels: int = 768,  # Reduced from 1024 (shorter sequences need less capacity)
        num_temporal_layers: int = 8,   # Reduced from 12 (8 frames = less temporal complexity)
        dropout: float = 0.25,  # Slightly reduced for shorter sequences
        use_spatial_attention: bool = True,
        attn_heads: int = 12,  # 12 heads × 64 = 768 (divisible)
        use_hierarchical_prediction: bool = True,
        use_motion_priors: bool = True,
        use_label_smoothing: bool = True,
        softmin_tau: float | None = None,
        sigmoid_scale: float = 1.0,
        floor_beta: float = 2.0,
    ):
        super().__init__()

        self.T = sequence_length
        self.hidden_channels = hidden_channels
        self.use_hierarchical_prediction = use_hierarchical_prediction
        self.use_motion_priors = use_motion_priors
    #
        self.use_label_smoothing = use_label_smoothing

        # ---- Enhanced Visual Backbones ----
        self.backbone_fast_features, feat_dim_fast = self._make_backbone(
            backbone_fast, pretrained_backbone_fast, freeze_backbone_fast
        )
        self.backbone_slow_features, feat_dim_slow = self._make_backbone(
            backbone_slow, pretrained_backbone_slow, freeze_backbone_slow
        )

        # ---- Enhanced Spatial Processing ----
        if use_spatial_attention:
            self.spatial_pool_fast = MultiScaleSpatialAttention(feat_dim_fast, hidden_channels // 4)
            self.spatial_pool_slow = MultiScaleSpatialAttention(feat_dim_slow, hidden_channels // 4)
        else:
            self.spatial_pool_fast = nn.AdaptiveAvgPool2d(1)
            self.spatial_pool_slow = nn.AdaptiveAvgPool2d(1)
            
        # ---- Motion Prior Extractor ----
        if use_motion_priors:
            self.motion_prior = MotionPriorExtractor(hidden_channels)

        # ---- Larger Feature Projections ----
        proj_input_fast = feat_dim_fast if not use_spatial_attention else hidden_channels
        proj_input_slow = feat_dim_slow if not use_spatial_attention else hidden_channels
        
        self.feature_proj_fast = nn.Sequential(
            nn.Linear(proj_input_fast, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.feature_proj_slow = nn.Sequential(
            nn.Linear(proj_input_slow, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # ---- Multi-Scale Temporal Processing for Different Action Durations ----
        # Fast pathway: captures quick actions (1-2 seconds)
        self.temporal_cnn_fast = EnhancedTemporalCNN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_temporal_layers,
            dropout=dropout,
        )
        # Slow pathway: captures longer actions and context (3-5 seconds)
        self.temporal_cnn_slow = EnhancedTemporalCNN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_temporal_layers,
            dropout=dropout,
        )
        
        # Additional temporal scales for fine-grained action modeling
        self.micro_temporal_cnn = EnhancedTemporalCNN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels // 2,
            num_layers=4,  # Fewer layers for micro-actions
            dropout=dropout,
        )
        
        # Projection layer to match micro features with main features
        self.micro_projection = nn.Linear(hidden_channels // 2, hidden_channels)

        # ---- Cross-Modal Fusion with Attention ----
        self.cross_modal_fusion = CrossModalAttentionFusion(hidden_channels, attn_heads)
        
        # ---- Hierarchical Triplet Prediction ----
        if use_hierarchical_prediction:
            # Stage 1: Predict verb (simplest, 4 classes) - optimized for short sequences
            self.verb_encoder = TransformerEncoder(
                d_model=hidden_channels,
                num_layers=4,  # Reduced for shorter sequences
                num_heads=attn_heads,
                dropout=dropout * 0.3 if USE_LIGHT_AUGMENTATION else dropout * 0.1,
            )
            self.verb_head = self._make_prediction_head(hidden_channels, NUM_VERBS, dropout * 0.2 if USE_LIGHT_AUGMENTATION else dropout * 0.05)
            
            # Stage 2: Predict subject conditioned on verb (3 classes)
            # Project concatenated features to maintain divisibility by num_heads
            self.subject_feature_proj = nn.Linear(hidden_channels + NUM_VERBS, hidden_channels)
            self.subject_encoder = TransformerEncoder(
                d_model=hidden_channels,  # Use projected features
                num_layers=4,  # Reduced for shorter sequences
                num_heads=attn_heads,
                dropout=dropout * 0.3 if USE_LIGHT_AUGMENTATION else dropout * 0.1,
            )
            self.subject_head = self._make_prediction_head(hidden_channels, NUM_SUBJECTS, dropout * 0.2 if USE_LIGHT_AUGMENTATION else dropout * 0.05)
            
            # Stage 3: Predict destination conditioned on verb + subject (14 classes)
            # Project concatenated features to maintain divisibility by num_heads
            self.destination_feature_proj = nn.Linear(hidden_channels + NUM_VERBS + NUM_SUBJECTS, hidden_channels)
            self.destination_encoder = TransformerEncoder(
                d_model=hidden_channels,  # Use projected features
                num_layers=6,  # Still more layers for hardest task, but reduced
                num_heads=attn_heads,
                dropout=dropout * 0.2 if USE_LIGHT_AUGMENTATION else dropout * 0.05,
            )
            self.destination_head = self._make_prediction_head(
                hidden_channels, 
                NUM_OBJECTS, 
                dropout * 0.1 if USE_LIGHT_AUGMENTATION else dropout * 0.02,
                extra_capacity=True  # Larger head for 14 classes
            )
        else:
            # Non-hierarchical (original approach)
            self.triplet_encoder = TransformerEncoder(
                d_model=hidden_channels,
                num_layers=num_temporal_layers,
                num_heads=attn_heads,
                dropout=dropout,
            )
            
            self.verb_head = self._make_prediction_head(hidden_channels, NUM_VERBS, dropout)
            self.subject_head = self._make_prediction_head(hidden_channels, NUM_SUBJECTS, dropout)
            self.destination_head = self._make_prediction_head(hidden_channels, NUM_OBJECTS, dropout, extra_capacity=True)

        # ---- Regularization Components ----
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout + i * 0.05) for i in range(4)])
        
        # Very light label smoothing or none
        if use_label_smoothing and USE_LIGHT_AUGMENTATION:
            self.label_smoother = LabelSmoothingCrossEntropy(smoothing=0.02)
        else:
            self.label_smoother = nn.CrossEntropyLoss()

        self._init_weights()

    def _make_prediction_head(self, in_features: int, num_classes: int, dropout: float, extra_capacity: bool = False):
        """Create prediction head with appropriate capacity."""
        if extra_capacity:
            # For complex tasks like destination (14 classes)
            return nn.Sequential(
                nn.Linear(in_features, in_features * 2),
                nn.LayerNorm(in_features * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(in_features * 2, in_features),
                nn.LayerNorm(in_features),
                nn.GELU(),
                nn.Dropout(dropout * 0.7),
                nn.Linear(in_features, in_features // 2),
                nn.LayerNorm(in_features // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(in_features // 2, num_classes),
            )
        else:
            # For simpler tasks like verb (4 classes) or subject (3 classes)
            return nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.LayerNorm(in_features // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(in_features // 2, in_features // 4),
                nn.LayerNorm(in_features // 4),
                nn.GELU(),
                nn.Dropout(dropout * 0.7),
                nn.Linear(in_features // 4, num_classes),
            )
    
    #

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
        ]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _pool_spatial(self, pool: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return pool(x).flatten(1)

    # _softmin method removed - unused

    # ----- forward -----
    def forward(
        self,
        frames: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None,
        training_stage: str = "triplet",
    ) -> TaskOutputs:
        B, T, C_in, H, W = frames.shape
        assert T == self.T, f"Expected sequence length {self.T}, got {T}"
        
    #

        # ---- Enhanced Feature Extraction ----
        # Pathway sampling
        frames_fast = frames
        idx_slow = torch.arange(0, T, 2, device=frames.device)
        frames_slow = frames.index_select(1, idx_slow)
        T_slow = frames_slow.shape[1]

        # Extract features with enhanced spatial processing
        xf = frames_fast.view(B * T, C_in, H, W)
        with torch.set_grad_enabled(self.backbone_fast_features.training):
            sf = self.backbone_fast_features(xf)
        vf = self._pool_spatial(self.spatial_pool_fast, sf)
        vf_seq = vf.view(B, T, -1)
        
        xs = frames_slow.view(B * T_slow, C_in, H, W)
        with torch.set_grad_enabled(self.backbone_slow_features.training):
            ss = self.backbone_slow_features(xs)
        vs = self._pool_spatial(self.spatial_pool_slow, ss)
        vs_seq = vs.view(B, T_slow, -1)

        # Enhanced feature projection
        feats_fast = torch.stack([self.feature_proj_fast(vf_seq[:, t]) for t in range(T)], dim=2)
        feats_slow = torch.stack([self.feature_proj_slow(vs_seq[:, t]) for t in range(T_slow)], dim=2)

        # Enhanced temporal processing
        tf = self.temporal_cnn_fast(feats_fast)
        ts = self.temporal_cnn_slow(feats_slow)

        # Multi-scale temporal fusion for different action durations
        ts_up = F.interpolate(ts, size=T, mode="linear", align_corners=False)
        
        # Micro-temporal features for fine-grained actions
        micro_features = self.micro_temporal_cnn(feats_fast)  # [B, C//2, T]
        
        # Cross-modal attention fusion with micro-temporal context
        fused_features = self.cross_modal_fusion(tf, ts_up)  # [B, C, T]
        
        # Project and integrate micro-temporal features - always merge
        micro_up = F.interpolate(micro_features, size=T, mode="linear", align_corners=False)
        micro_projected = micro_up.transpose(1, 2)  # [B, T, C//2]
        micro_enhanced = self.micro_projection(micro_projected).transpose(1, 2)  # [B, C, T]
        
        # Always combine micro-temporal features with main features
        enhanced_features = fused_features + 0.3 * micro_enhanced  # Weighted combination
        
        sequence_features = enhanced_features.transpose(1, 2)  # [B, T, C]
        
        # ---- Motion Prior Integration ----
        if self.use_motion_priors:
            motion_attention = self.motion_prior(frames)  # [B, hidden_channels]
            # Apply motion attention to sequence features - broadcast correctly
            # motion_attention: [B, C] -> [B, 1, C] to match sequence_features: [B, T, C]
            if motion_attention.shape[-1] == sequence_features.shape[-1]:
                sequence_features = sequence_features * motion_attention.unsqueeze(1)  # [B, T, C]

        # ---- Hierarchical Triplet Prediction ----
        if self.use_hierarchical_prediction:
            return self._hierarchical_forward(sequence_features)
        else:
            return self._standard_forward(sequence_features)
    
    def _hierarchical_forward(self, sequence_features: torch.Tensor):
        """Hierarchical prediction: verb -> subject -> destination"""
        B, T, C = sequence_features.shape
        
        # Stage 1: Predict verb
        verb_features = self.verb_encoder(sequence_features)
        verb_global = verb_features.mean(dim=1)  # Global temporal pooling
        verb_logits = self.verb_head(verb_global)
        
        # Stage 2: Predict subject conditioned on verb
        verb_probs = F.softmax(verb_logits, dim=1)
        verb_conditioned_features = torch.cat([
            sequence_features, 
            verb_probs.unsqueeze(1).expand(-1, T, -1)
        ], dim=2)
        
        # Project to maintain proper embedding dimension
        projected_verb_features = self.subject_feature_proj(verb_conditioned_features)
        subject_features = self.subject_encoder(projected_verb_features)
        subject_global = subject_features.mean(dim=1)
        subject_logits = self.subject_head(subject_global)
        
        # Stage 3: Predict destination conditioned on verb + subject
        subject_probs = F.softmax(subject_logits, dim=1)
        full_conditioned_features = torch.cat([
            sequence_features,
            verb_probs.unsqueeze(1).expand(-1, T, -1),
            subject_probs.unsqueeze(1).expand(-1, T, -1)
        ], dim=2)
        
        # Project to maintain proper embedding dimension
        projected_full_features = self.destination_feature_proj(full_conditioned_features)
        destination_features = self.destination_encoder(projected_full_features)
        destination_global = destination_features.mean(dim=1)
        destination_logits = self.destination_head(destination_global)

        return TaskOutputs(
            anticipation=None,
            phase_logits=None,
            completion=None,
            triplet_verb_logits=verb_logits,
            triplet_subject_logits=subject_logits,
            triplet_destination_logits=destination_logits,
        )
    
    def _standard_forward(self, sequence_features: torch.Tensor):
        """Non-hierarchical triplet prediction"""
        B, T, C = sequence_features.shape
        
        # Single encoder for all triplet components
        triplet_features = self.triplet_encoder(sequence_features)
        triplet_global = triplet_features.mean(dim=1)  # [B, C]
        
        # Individual triplet component predictions
        verb_logits = self.verb_head(triplet_global)
        subject_logits = self.subject_head(triplet_global)
        destination_logits = self.destination_head(triplet_global)

        return TaskOutputs(
            anticipation=None,
            phase_logits=None,
            completion=None,
            triplet_verb_logits=verb_logits,
            triplet_subject_logits=subject_logits,
            triplet_destination_logits=destination_logits,
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
NUM_OBJECTS = len(TRIPLET_TARGETS)   # dynamically: 5 pegs + 4 rings + center + outside + 2 arms + null

SEQ_LEN = 8  # Optimized for action frequency: covers 8 seconds (2-3 typical actions)
STRIDE = 1   # Keep stride=1 for dense sampling

BATCH_SIZE = 16  # Can increase due to shorter sequences (8 vs 16 frames)
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 16 * 2 = 32  
USE_GRADIENT_ACCUMULATION = False  # Enable/disable gradient accumulation
NUM_WORKERS = 30
EPOCHS = 250  # More epochs due to faster convergence with shorter sequences

LR = 1.5e-4  # Slightly higher LR for shorter sequences (faster convergence)
WEIGHT_DECAY = 1.5e-4  # Reduced weight decay (shorter sequences = less overfitting risk)

TIME_HORIZON = 2.0

PRINT_EVERY = 40
CKPT_PATH = Path("peg_and_ring_slowfast_transformer_triplets_v2_enhanced.pth")
LAST_CKPT_PATH = Path("peg_and_ring_slowfast_transformer_triplets_v2_enhanced_last.pth")  # resume point

# control flags
DO_TRAIN = True
RESUME_IF_CRASH = True
USE_WANDB = True  # Enable Weights & Biases logging
USE_LIGHT_AUGMENTATION = False  # Enable very light data augmentation (recommended: False for small datasets)
WANDB_PROJECT = "peg_ring_slowfast_triplets_v2_enhanced"  # wandb project name
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


def calculate_comprehensive_triplet_metrics(
    verb_preds, subject_preds, dest_preds,
    verb_gts, subject_gts, dest_gts,
    verb_probs, subject_probs, dest_probs
) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics for triplet classification."""
    
    # Individual component accuracies
    verb_acc = (verb_preds == verb_gts).mean()
    subject_acc = (subject_preds == subject_gts).mean() 
    dest_acc = (dest_preds == dest_gts).mean()
    
    # Combination accuracies
    verb_subject_acc = ((verb_preds == verb_gts) & (subject_preds == subject_gts)).mean()
    verb_dest_acc = ((verb_preds == verb_gts) & (dest_preds == dest_gts)).mean()
    subject_dest_acc = ((subject_preds == subject_gts) & (dest_preds == dest_gts)).mean()
    complete_acc = ((verb_preds == verb_gts) & (subject_preds == subject_gts) & (dest_preds == dest_gts)).mean()
    
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
    subject_map = calculate_multiclass_map(subject_gts, subject_probs, NUM_SUBJECTS)
    dest_map = calculate_multiclass_map(dest_gts, dest_probs, NUM_OBJECTS)
    
    # Overall mAP (average of component mAPs)
    overall_map = (verb_map + subject_map + dest_map) / 3.0
    
    return {
        # Individual accuracies
        "triplet_verb_acc": float(verb_acc),
        "triplet_subject_acc": float(subject_acc), 
        "triplet_destination_acc": float(dest_acc),
        
        # Combination accuracies
        "triplet_verb_subject_acc": float(verb_subject_acc),
        "triplet_verb_destination_acc": float(verb_dest_acc),
        "triplet_subject_destination_acc": float(subject_dest_acc),
        "triplet_complete_acc": float(complete_acc),
        
        # mAP metrics
        "triplet_verb_map": float(verb_map),
        "triplet_subject_map": float(subject_map),
        "triplet_destination_map": float(dest_map),
        "triplet_overall_map": float(overall_map),
        
        # Additional derived metrics
        "triplet_avg_component_acc": float((verb_acc + subject_acc + dest_acc) / 3.0),
        "triplet_avg_pair_acc": float((verb_subject_acc + verb_dest_acc + subject_dest_acc) / 3.0),
    }


def save_detailed_metrics_to_file(metrics: Dict[str, Any], split: str):
    """Save detailed metrics to a text file."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"triplet_evaluation_metrics_{split}_{timestamp}.txt"
    filepath = Path("results") / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write(f"Triplet Classification Evaluation Metrics - {split.upper()} Set\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        # Individual component metrics
        f.write("INDIVIDUAL COMPONENT METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Verb Accuracy:        {metrics['triplet_verb_acc']:.4f}\n")
        f.write(f"Subject Accuracy:     {metrics['triplet_subject_acc']:.4f}\n")
        f.write(f"Destination Accuracy: {metrics['triplet_destination_acc']:.4f}\n\n")
        
        f.write(f"Verb mAP:             {metrics['triplet_verb_map']:.4f}\n")
        f.write(f"Subject mAP:          {metrics['triplet_subject_map']:.4f}\n")
        f.write(f"Destination mAP:      {metrics['triplet_destination_map']:.4f}\n\n")
        
        # Combination metrics
        f.write("COMBINATION METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Verb + Subject:       {metrics['triplet_verb_subject_acc']:.4f}\n")
        f.write(f"Verb + Destination:   {metrics['triplet_verb_destination_acc']:.4f}\n")
        f.write(f"Subject + Destination: {metrics['triplet_subject_destination_acc']:.4f}\n")
        f.write(f"Complete Triplet:     {metrics['triplet_complete_acc']:.4f}\n\n")
        
        # Summary metrics
        f.write("SUMMARY METRICS:\n")
        f.write("-" * 16 + "\n")
        f.write(f"Overall mAP:          {metrics['triplet_overall_map']:.4f}\n")
        f.write(f"Avg Component Acc:    {metrics['triplet_avg_component_acc']:.4f}\n")
        f.write(f"Avg Pair Acc:         {metrics['triplet_avg_pair_acc']:.4f}\n")
        f.write(f"Cross Entropy Loss:   {metrics['triplet_ce']:.4f}\n")
        f.write(f"Total Samples:        {metrics['samples']}\n\n")
        
        # Additional info
        f.write("DATASET INFO:\n")
        f.write("-" * 13 + "\n")
        f.write(f"Number of Verbs:      {NUM_VERBS}\n")
        f.write(f"Number of Subjects:   {NUM_SUBJECTS}\n")
        f.write(f"Number of Objects:    {NUM_OBJECTS}\n")
    
    print(f"📊 Detailed metrics saved to: {filepath}")


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
    model: nn.Module, loader: DataLoader, device: torch.device, time_horizon: float, split: str = "val"
) -> Dict[str, Any]:
    model.eval()

    # Collect all predictions and ground truths for comprehensive metrics
    all_verb_preds = []
    all_subject_preds = []
    all_dest_preds = []
    all_verb_gts = []
    all_subject_gts = []
    all_dest_gts = []
    all_verb_probs = []
    all_subject_probs = []
    all_dest_probs = []
    
    # Legacy metrics for compatibility
    total_triplet_ce = 0.0
    total_samples = 0
    ce_loss = nn.CrossEntropyLoss()

    for _, (frames, meta) in enumerate(loader):
        frames = frames.to(device)
        
        # Triplet ground truth labels
        triplet_gt = meta["triplet_classification"].to(device).long()  # [B, 3]
        triplet_verb_gt = triplet_gt[:, 0]     # [B]
        triplet_subject_gt = triplet_gt[:, 1]  # [B]
        triplet_dest_gt = triplet_gt[:, 2]     # [B]

        outputs = model(frames, meta)

        # Triplet predictions and probabilities
        triplet_verb_logits = outputs.triplet_verb_logits
        triplet_subject_logits = outputs.triplet_subject_logits
        triplet_dest_logits = outputs.triplet_destination_logits
        
        triplet_verb_pred = triplet_verb_logits.argmax(dim=1)
        triplet_subject_pred = triplet_subject_logits.argmax(dim=1)
        triplet_dest_pred = triplet_dest_logits.argmax(dim=1)
        
        # Collect for comprehensive metrics
        all_verb_preds.extend(triplet_verb_pred.cpu().numpy().tolist())
        all_subject_preds.extend(triplet_subject_pred.cpu().numpy().tolist())
        all_dest_preds.extend(triplet_dest_pred.cpu().numpy().tolist())
        all_verb_gts.extend(triplet_verb_gt.cpu().numpy().tolist())
        all_subject_gts.extend(triplet_subject_gt.cpu().numpy().tolist())
        all_dest_gts.extend(triplet_dest_gt.cpu().numpy().tolist())
        
        # Probabilities for mAP calculation
        all_verb_probs.extend(torch.softmax(triplet_verb_logits, dim=1).cpu().numpy().tolist())
        all_subject_probs.extend(torch.softmax(triplet_subject_logits, dim=1).cpu().numpy().tolist())
        all_dest_probs.extend(torch.softmax(triplet_dest_logits, dim=1).cpu().numpy().tolist())
        
        # Legacy cross-entropy loss
        triplet_verb_ce = ce_loss(triplet_verb_logits, triplet_verb_gt)
        triplet_subject_ce = ce_loss(triplet_subject_logits, triplet_subject_gt)
        triplet_dest_ce = ce_loss(triplet_dest_logits, triplet_dest_gt)
        batch_triplet_ce = (triplet_verb_ce + triplet_subject_ce + triplet_dest_ce) / 3.0
        total_triplet_ce += float(batch_triplet_ce.item()) * frames.size(0)
        
        total_samples += frames.size(0)

    # Convert to numpy arrays for sklearn
    all_verb_preds = np.array(all_verb_preds)
    all_subject_preds = np.array(all_subject_preds)
    all_dest_preds = np.array(all_dest_preds)
    all_verb_gts = np.array(all_verb_gts)
    all_subject_gts = np.array(all_subject_gts)
    all_dest_gts = np.array(all_dest_gts)
    all_verb_probs = np.array(all_verb_probs)
    all_subject_probs = np.array(all_subject_probs)
    all_dest_probs = np.array(all_dest_probs)
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_triplet_metrics(
        all_verb_preds, all_subject_preds, all_dest_preds,
        all_verb_gts, all_subject_gts, all_dest_gts,
        all_verb_probs, all_subject_probs, all_dest_probs
    )
    
    # Add legacy metrics for compatibility
    metrics.update({
        "mae": 0.0,  # Dummy for compatibility
        "ce": 0.0,   # Dummy for compatibility
        "acc": 0.0,  # Dummy for compatibility
        "compl_mae": 0.0,  # Dummy for compatibility
        "triplet_ce": total_triplet_ce / max(1, total_samples),
        "samples": total_samples,
    })
    
    # Save detailed metrics to text file
    save_detailed_metrics_to_file(metrics, split)
    
    return metrics


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
    
    # Enhanced loss functions
    ce_loss = nn.CrossEntropyLoss()
    if hasattr(model, 'use_label_smoothing') and model.use_label_smoothing:
        smooth_loss = model.label_smoother
    else:
        smooth_loss = ce_loss
    
    # Single optimizer for enhanced model
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Optimized scheduler for action-frequency data
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )  # Warm restarts every 50 epochs to escape local minima

    # Resume if requested  
    start_epoch, best_val_mae = _try_resume(model, optimizer, scheduler, device)

    # Run initial validation before training starts
    if start_epoch == 1:
        print("Running initial validation before training...")
        val_stats_initial = evaluate(model, val_loader, device, TIME_HORIZON, "val_initial")
        print(f"Initial validation: complete_acc={val_stats_initial['triplet_complete_acc']:.4f} overall_mAP={val_stats_initial['triplet_overall_map']:.4f}")
        
        # Log comprehensive initial metrics to wandb
        initial_wandb_metrics = {f"val_initial_{k}": v for k, v in val_stats_initial.items() if isinstance(v, (int, float))}
        log_to_wandb(initial_wandb_metrics, step=0)

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_triplet_verb_acc = epoch_triplet_subject_acc = 0.0
        epoch_triplet_dest_acc = epoch_triplet_complete_acc = 0.0
        seen = 0
        t0 = time.time()

        for it, (frames, meta) in enumerate(train_loader, start=1):
            frames = frames.to(device)
            
            # Triplet ground truth labels
            triplet_gt = meta["triplet_classification"].to(device).long()  # [B, 3]
            triplet_verb_gt = triplet_gt[:, 0]
            triplet_subject_gt = triplet_gt[:, 1]
            triplet_dest_gt = triplet_gt[:, 2]

            outputs = model(frames, meta)

            # Progressive hierarchical loss weighting based on epoch
            # Early epochs: focus on easier tasks (verb, subject)
            # Later epochs: increase weight on harder task (destination)
            progress = min(epoch / (EPOCHS * 0.7), 1.0)  # Progress from 0 to 1 over 70% of training
            
            # Use label smoothing for better generalization
            if hasattr(model, 'use_label_smoothing') and model.use_label_smoothing and epoch > 10:
                loss_triplet_verb = smooth_loss(outputs.triplet_verb_logits, triplet_verb_gt)
                loss_triplet_subject = smooth_loss(outputs.triplet_subject_logits, triplet_subject_gt)
                loss_triplet_dest = smooth_loss(outputs.triplet_destination_logits, triplet_dest_gt)
            else:
                loss_triplet_verb = ce_loss(outputs.triplet_verb_logits, triplet_verb_gt)
                loss_triplet_subject = ce_loss(outputs.triplet_subject_logits, triplet_subject_gt)
                loss_triplet_dest = ce_loss(outputs.triplet_destination_logits, triplet_dest_gt)
            
            # Hierarchical loss weighting: progressive emphasis on harder tasks
            verb_weight = 1.0
            subject_weight = 1.0 + 0.2 * progress  # Slightly increase with training
            dest_weight = 1.5 + 1.0 * progress    # Significantly increase with training
            
            loss_total = (
                verb_weight * loss_triplet_verb +
                subject_weight * loss_triplet_subject +
                dest_weight * loss_triplet_dest
            ) / (verb_weight + subject_weight + dest_weight)  # Normalize
            
            # Scale loss for gradient accumulation if enabled
            if USE_GRADIENT_ACCUMULATION:
                loss_total = loss_total / GRADIENT_ACCUMULATION_STEPS

            loss_total.backward()
            
            # Gradient accumulation: only step optimizer every N accumulation steps if enabled
            if USE_GRADIENT_ACCUMULATION:
                if it % GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            else:
                # Normal training: step optimizer every iteration
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                # Triplet metrics only
                triplet_verb_pred = outputs.triplet_verb_logits.argmax(dim=1)
                triplet_subject_pred = outputs.triplet_subject_logits.argmax(dim=1)
                triplet_dest_pred = outputs.triplet_destination_logits.argmax(dim=1)
                
                triplet_verb_acc = (triplet_verb_pred == triplet_verb_gt).float().mean()
                triplet_subject_acc = (triplet_subject_pred == triplet_subject_gt).float().mean()
                triplet_dest_acc = (triplet_dest_pred == triplet_dest_gt).float().mean()
                triplet_complete_acc = (
                    (triplet_verb_pred == triplet_verb_gt) &
                    (triplet_subject_pred == triplet_subject_gt) &
                    (triplet_dest_pred == triplet_dest_gt)
                ).float().mean()

            bs = frames.size(0)
            # Unscale loss for reporting if gradient accumulation is used
            reported_loss = loss_total.item() * (GRADIENT_ACCUMULATION_STEPS if USE_GRADIENT_ACCUMULATION else 1)
            epoch_loss += float(reported_loss) * bs
            epoch_triplet_verb_acc += float(triplet_verb_acc.item()) * bs
            epoch_triplet_subject_acc += float(triplet_subject_acc.item()) * bs
            epoch_triplet_dest_acc += float(triplet_dest_acc.item()) * bs
            epoch_triplet_complete_acc += float(triplet_complete_acc.item()) * bs
            seen += bs

            if it % PRINT_EVERY == 0:
                cur_lr = scheduler.get_last_lr()[0]
                # Show unscaled loss for better interpretability
                unscaled_loss = loss_total.item() * (GRADIENT_ACCUMULATION_STEPS if USE_GRADIENT_ACCUMULATION else 1)
                print(
                    f"[Epoch {epoch:02d} | it {it:04d}] "
                    f"loss={unscaled_loss:.4f} | "
                    f"v_acc={triplet_verb_acc.item():.4f} | s_acc={triplet_subject_acc.item():.4f} | "
                    f"d_acc={triplet_dest_acc.item():.4f} | complete_acc={triplet_complete_acc.item():.4f} | "
                    f"lr={cur_lr:.2e}"
                )

        # Curriculum learning: adjust model behavior based on training progress
        training_progress = epoch / EPOCHS
        
        # Reduce regularization as training progresses for fine-tuning
        if hasattr(model, 'dropout_layers'):
            for i, dropout_layer in enumerate(model.dropout_layers):
                # Reduce dropout in later stages of training
                if training_progress > 0.8:
                    dropout_layer.p = max(0.1, dropout_layer.p * 0.95)
        
        if epoch % 5 == 0:
            gpu_mem_allocated = torch.cuda.memory_allocated() / 1e9
            gpu_mem_cached = torch.cuda.memory_reserved() / 1e9
            print(f"GPU mem: {gpu_mem_allocated:.4f}GB")
            print(f"GPU cached: {gpu_mem_cached:.4f}GB")
            print(f"Training progress: {training_progress:.2%}")
            
            # Log GPU memory usage and training progress to wandb
            log_to_wandb({
                "system_gpu_memory_allocated_gb": gpu_mem_allocated,
                "system_gpu_memory_cached_gb": gpu_mem_cached,
                "training_progress": training_progress,
                "current_verb_weight": verb_weight if 'verb_weight' in locals() else 1.0,
                "current_subject_weight": subject_weight if 'subject_weight' in locals() else 1.0,
                "current_destination_weight": dest_weight if 'dest_weight' in locals() else 2.0,
            }, step=epoch)

            # collect
            gc.collect()
            torch.cuda.empty_cache()

        # Final gradient step if there are remaining accumulated gradients and gradient accumulation is enabled
        if USE_GRADIENT_ACCUMULATION and len(train_loader) % GRADIENT_ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # step scheduler once per epoch
        scheduler.step()

        train_loss_avg = epoch_loss / max(1, seen)
        train_triplet_verb_acc_avg = epoch_triplet_verb_acc / max(1, seen)
        train_triplet_subject_acc_avg = epoch_triplet_subject_acc / max(1, seen)
        train_triplet_dest_acc_avg = epoch_triplet_dest_acc / max(1, seen)
        train_triplet_complete_acc_avg = epoch_triplet_complete_acc / max(1, seen)
        
        print(
            f"\nEpoch {epoch:02d} [{time.time()-t0:.1f}s] "
            f"| train_loss={train_loss_avg:.4f} "
            f"v_acc={train_triplet_verb_acc_avg:.4f} s_acc={train_triplet_subject_acc_avg:.4f} "
            f"d_acc={train_triplet_dest_acc_avg:.4f} complete_acc={train_triplet_complete_acc_avg:.4f}"
        )
        
        # Log epoch-level training metrics to wandb
        epoch_duration = time.time() - t0
        log_to_wandb({
            "epoch": epoch,
            "train_total_loss": train_loss_avg,
            "train_triplet_verb_accuracy": train_triplet_verb_acc_avg,
            "train_triplet_subject_accuracy": train_triplet_subject_acc_avg,
            "train_triplet_destination_accuracy": train_triplet_dest_acc_avg,
            "train_triplet_complete_accuracy": train_triplet_complete_acc_avg,
            "train_epoch_duration_seconds": epoch_duration,
            "train_learning_rate": scheduler.get_last_lr()[0],
            "train_samples": seen,
        }, step=epoch)

        # Validation
        val_stats = evaluate(model, val_loader, device, TIME_HORIZON, "val")
        print(
            f"           val_triplet: ce={val_stats['triplet_ce']:.4f} mAP={val_stats['triplet_overall_map']:.4f} "
            f"v_acc={val_stats['triplet_verb_acc']:.4f} s_acc={val_stats['triplet_subject_acc']:.4f} "
            f"d_acc={val_stats['triplet_destination_acc']:.4f} complete_acc={val_stats['triplet_complete_acc']:.4f}"
        )
        
        # Log comprehensive validation metrics to wandb
        val_wandb_metrics = {f"val_{k}": v for k, v in val_stats.items() if isinstance(v, (int, float)) and k != 'samples'}
        val_wandb_metrics['val_samples'] = val_stats['samples']
        log_to_wandb(val_wandb_metrics, step=epoch)

        # Save best model based on triplet complete accuracy (higher is better)
        is_best = val_stats["triplet_complete_acc"] > best_val_mae  # Reusing variable name but now for accuracy
        if is_best:
            best_val_mae = val_stats["triplet_complete_acc"]
            torch.save(model.state_dict(), CKPT_PATH)
            print(
                f"✅  New best triplet_complete_acc={best_val_mae:.4f} — saved to: {CKPT_PATH}"
            )
            
            # Log best model metrics to wandb
            log_to_wandb({
                "best_val_triplet_complete_accuracy": best_val_mae,
                "best_val_triplet_destination_accuracy": val_stats['triplet_destination_acc'],
                "best_epoch": epoch,
            }, step=epoch)

        # Always save last for resume (even if not best)
        _save_last_checkpoint(model, optimizer, scheduler, epoch, best_val_mae, LAST_CKPT_PATH)


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
                "model_hidden_channels": 768,
                "model_num_temporal_layers": 8,
                "model_dropout": 0.25,
                "model_attn_heads": 12,
                "data_optimized": True,
                "action_frequency_optimized": "2-4_seconds",
                "sequence_coverage": "8_seconds",
                "backbone_fast": "resnet50",
                "backbone_slow": "resnet50",
                "architecture": "SlowFastTripletPredictorV2",
                "use_hierarchical_prediction": True,
                "use_motion_priors": True,
                # "use_mixup_augmentation": removed
                "use_label_smoothing": True,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                # Multi-task learning config
                "use_dynamic_loss_weighting": USE_DYNAMIC_LOSS_WEIGHTING,
                "use_separate_task_encoders": USE_SEPARATE_TASK_ENCODERS,
                "triplet_encoder_extra_layers": TRIPLET_ENCODER_EXTRA_LAYERS,
                "cross_task_attention_weight": CROSS_TASK_ATTENTION_WEIGHT,
                "multitask_strategy": "uncertainty_weighting_separate_encoders",
            },
            tags=["slowfast", "triplets-only", "hierarchical", "motion-priors", "enhanced-regularization", "small-dataset-optimized"]
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

    # Data-Optimized Model - Tuned for 8-frame sequences and 2-4s action frequency
    model = SlowFastTripletPredictorV2(
        sequence_length=SEQ_LEN,
        backbone_fast="resnet50",
        backbone_slow="resnet50",
        pretrained_backbone_fast=True,
        pretrained_backbone_slow=True,
        freeze_backbone_fast=False,
        freeze_backbone_slow=False,
        hidden_channels=768,  # Optimized for shorter sequences
        num_temporal_layers=8,   # Balanced for 8-frame context
        dropout=0.15 if USE_LIGHT_AUGMENTATION else 0.05,  # Minimal dropout when no augmentation
        use_spatial_attention=True,
        attn_heads=12,  # 12 heads × 64 = 768 (divisible)
        use_hierarchical_prediction=True,  # Enable hierarchical learning
        use_motion_priors=True,  # Important for manipulation tasks
    # use_mixup removed
        use_label_smoothing=USE_LIGHT_AUGMENTATION,  # Only enable if light augmentation is on (recommended: False)
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

    test_stats = evaluate(model, test_loader, device, TIME_HORIZON, "test")
    print(
        f"\nTEST RESULTS:"
        f"\n  Cross Entropy:     {test_stats['triplet_ce']:.4f}"
        f"\n  Overall mAP:       {test_stats['triplet_overall_map']:.4f}"
        f"\n  Individual Acc:    V={test_stats['triplet_verb_acc']:.4f} S={test_stats['triplet_subject_acc']:.4f} D={test_stats['triplet_destination_acc']:.4f}"
        f"\n  Combination Acc:   V+S={test_stats['triplet_verb_subject_acc']:.4f} V+D={test_stats['triplet_verb_destination_acc']:.4f} S+D={test_stats['triplet_subject_destination_acc']:.4f}"
        f"\n  Complete Acc:      {test_stats['triplet_complete_acc']:.4f}"
        f"\n  Samples:           {test_stats['samples']}"
    )
    
    # Log comprehensive test results to wandb
    test_wandb_metrics = {f"test_{k}": v for k, v in test_stats.items() if isinstance(v, (int, float)) and k != 'samples'}
    test_wandb_metrics['test_samples'] = test_stats['samples']
    log_to_wandb(test_wandb_metrics)
    
    # Create comprehensive summary table for wandb
    if USE_WANDB and wandb.run:
        summary_table = wandb.Table(
            columns=["Metric Category", "Metric", "Value"],
            data=[
                # Individual Component Metrics
                ["Individual Accuracy", "Verb Accuracy", f"{test_stats['triplet_verb_acc']:.4f}"],
                ["Individual Accuracy", "Subject Accuracy", f"{test_stats['triplet_subject_acc']:.4f}"],
                ["Individual Accuracy", "Destination Accuracy", f"{test_stats['triplet_destination_acc']:.4f}"],
                
                # mAP Metrics
                ["mAP", "Verb mAP", f"{test_stats['triplet_verb_map']:.4f}"],
                ["mAP", "Subject mAP", f"{test_stats['triplet_subject_map']:.4f}"],
                ["mAP", "Destination mAP", f"{test_stats['triplet_destination_map']:.4f}"],
                ["mAP", "Overall mAP", f"{test_stats['triplet_overall_map']:.4f}"],
                
                # Combination Metrics
                ["Combination Accuracy", "Verb + Subject", f"{test_stats['triplet_verb_subject_acc']:.4f}"],
                ["Combination Accuracy", "Verb + Destination", f"{test_stats['triplet_verb_destination_acc']:.4f}"],
                ["Combination Accuracy", "Subject + Destination", f"{test_stats['triplet_subject_destination_acc']:.4f}"],
                ["Combination Accuracy", "Complete Triplet", f"{test_stats['triplet_complete_acc']:.4f}"],
                
                # Summary Metrics
                ["Summary", "Avg Component Accuracy", f"{test_stats['triplet_avg_component_acc']:.4f}"],
                ["Summary", "Avg Pair Accuracy", f"{test_stats['triplet_avg_pair_acc']:.4f}"],
                ["Summary", "Cross Entropy Loss", f"{test_stats['triplet_ce']:.4f}"],
                
                # Model Info
                ["Model", "Total Parameters", f"{sum(p.numel() for p in model.parameters()):,}"],
                ["Model", "Trainable Parameters", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"],
                ["Model", "Test Samples", f"{test_stats['samples']}"],
            ]
        )
        wandb.log({"comprehensive_test_summary": summary_table})

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
