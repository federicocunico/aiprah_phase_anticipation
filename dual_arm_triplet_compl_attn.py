#!/usr/bin/env python3
"""
Dual-Arm Triplet Predictor with Multi-Modal Cross-Attention

This implementation extends the DualArmTripletPredictor with rich multi-modal signals:
1. Phase Information: Current surgical phase (0-5) providing workflow context
2. Time-to-Next-Phase: Anticipation signals for all 6 phases indicating upcoming transitions
3. Completion Percentage: Task progress indicator from dataset (0-1)
4. Previous Context: Historical anticipation values providing temporal memory
5. Multi-modal Cross-Attention: Each signal type attends to temporal features independently

Cross-attention mechanisms: Attn(Q=signal_embedding, K=F(X), V=F(X)) where:
- Q = phase/anticipation/completion/context queries from dataset metadata
- K, V = temporal features from sequence
- Multiple attention heads focus on different aspects of temporal dynamics

Key improvements over base model:
- Multi-modal contextual understanding of surgical workflow
- Rich anticipation signals guide action prediction
- Phase-aware temporal modeling for better surgical understanding
- Historical context for temporal consistency
- Enhanced temporal modeling for better bimanual manipulation understanding

The dual-arm structure remains:
- Left arm: verb (4 classes) + target (14 classes)
- Right arm: verb (4 classes) + target (14 classes)
- Each arm operates independently with null operations when inactive
"""

import gc
import os
import time
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from collections import defaultdict
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import average_precision_score, classification_report

from datasets.peg_and_ring_workflow import (
    PegAndRing, VideoBatchSampler, 
    TRIPLET_VERBS, TRIPLET_SUBJECTS, TRIPLET_TARGETS
)

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Backbone imports
try:
    from torchvision.models import resnet50, ResNet50_Weights
    _HAS_TORCHVISION_WEIGHTS_ENUM = True
except Exception:
    import torchvision
    _HAS_TORCHVISION_WEIGHTS_ENUM = False

# Constants
SEED = 42
ROOT_DIR = "data/peg_and_ring_workflow"
TIME_UNIT = "minutes"

# Dual-arm triplet dimensions
NUM_VERBS = len(TRIPLET_VERBS)      # 4: reach, grasp, release, null-verb
NUM_TARGETS = len(TRIPLET_TARGETS)   # 14: pegs, rings, center, outside, arms, null-target

# Training hyperparameters - optimized for dual-arm learning with completion attention
SEQ_LEN = 8           # 8 frames = ~8 seconds at 1fps
STRIDE = 1            # Maximum data utilization - use all possible windows
BATCH_SIZE = 32       # Smaller batch size for more frequent updates and better gradients
NUM_WORKERS = 16
EPOCHS = 80           # Fewer epochs with early stopping to prevent overfitting

LR = 3e-4            # Increased learning rate for actual learning
WEIGHT_DECAY = 1e-4   # Moderate weight decay
WARMUP_EPOCHS = 5     # Shorter warmup to start learning faster
EMA_DECAY = 0.999     # Standard EMA decay

# Model architecture hyperparameters
HIDDEN_DIM = 256      # Reduced to prevent overfitting
DROPOUT = 0.3         # Moderate dropout to allow learning
NUM_TEMPORAL_LAYERS = 2  # Fewer layers to reduce overfitting
NUM_ATTENTION_HEADS = 8  # Multi-head attention for completion cross-attention

# Control flags
DO_TRAIN = True
USE_WANDB = True
WANDB_PROJECT = "dual_arm_triplet_predictor"
WANDB_RUN_NAME = "completion_attention_v1"

# Paths
CKPT_PATH = Path("dual_arm_triplet_compl_attn_best.pth")
LAST_CKPT_PATH = Path("dual_arm_triplet_compl_attn_last.pth")

# Evaluation directories
EVAL_ROOT = Path("results/dual_arm_triplet_compl_attn_eval")
EVAL_ROOT.mkdir(parents=True, exist_ok=True)


class CompletionCrossAttention(nn.Module):
    """
    Cross-attention module between completion percentage queries and temporal features.
    
    Uses Attn(Q=z, K=F(X), V=F(X)) where:
    - Q (queries): completion percentage embeddings [B, 1, hidden_dim]  
    - K, V (keys, values): temporal features [B, hidden_dim, T] -> [B, T, hidden_dim]
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights."""
        for module in [self.query_proj, self.key_proj, self.value_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, completion_queries: torch.Tensor, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of completion cross-attention.
        
        Args:
            completion_queries: [B, 1, hidden_dim] - completion percentage queries
            temporal_features: [B, hidden_dim, T] - temporal features from sequence
            
        Returns:
            attended_features: [B, hidden_dim] - completion-aware features
        """
        B, hidden_dim, T = temporal_features.shape
        
        # Transpose temporal_features to [B, T, hidden_dim] for attention
        temporal_features = temporal_features.transpose(1, 2)  # [B, T, hidden_dim]
        
        # Project to Q, K, V
        Q = self.query_proj(completion_queries)  # [B, 1, hidden_dim]
        K = self.key_proj(temporal_features)     # [B, T, hidden_dim]
        V = self.value_proj(temporal_features)   # [B, T, hidden_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)        # [B, num_heads, 1, head_dim]
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)        # [B, num_heads, T, head_dim]
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)        # [B, num_heads, T, head_dim]
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale        # [B, num_heads, 1, T]
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)                          # [B, num_heads, 1, head_dim]
        
        # Concatenate heads and reshape
        attended = attended.transpose(1, 2).contiguous().view(B, 1, hidden_dim)  # [B, 1, hidden_dim]
        
        # Output projection
        output = self.out_proj(attended)  # [B, 1, hidden_dim]
        
        # Residual connection and normalization
        output = self.norm(output + completion_queries)  # [B, 1, hidden_dim]
        
        # Return squeezed output [B, hidden_dim]
        return output.squeeze(1)


class DualArmTripletMultiModalPredictor(nn.Module):
    """
    Enhanced dual-arm triplet predictor with multi-modal cross-attention.
    
    Architecture:
    1. ResNet50 backbone for visual features
    2. Temporal 1D convolutions for sequence modeling
    3. Multi-modal signal embeddings:
       - Phase information (surgical workflow context)
       - Time-to-next-phase (anticipation signals)
       - Completion percentage (progress indicator)
       - Previous context (temporal memory)
    4. Multi-head cross-attention between signals and temporal features
    5. Dual-stream classification: separate heads for each arm
    """
    
    def __init__(
        self,
        seq_len: int = SEQ_LEN,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
        num_temporal_layers: int = NUM_TEMPORAL_LAYERS,
        num_attention_heads: int = NUM_ATTENTION_HEADS
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Visual backbone - pretrained ResNet50 with frozen early layers
        if _HAS_TORCHVISION_WEIGHTS_ENUM:
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            backbone = torchvision.models.resnet50(pretrained=True)
        
        # Remove final classification layers
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove avgpool and fc
        backbone_dim = 2048  # ResNet50 feature dimension
        
        # Freeze more layers to prevent overfitting on small dataset
        for i, child in enumerate(self.backbone.children()):
            if i < 7:  # Freeze first 7 layers (conv1, bn1, relu, maxpool, layer1, layer2, layer3)
                for param in child.parameters():
                    param.requires_grad = False
        
        # Spatial pooling with dropout
        self.spatial_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(dropout * 0.5)  # Spatial dropout
        )
        
        # Feature projection with layer normalization
        self.feature_proj = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.3)
        )
        
        # Temporal modeling with better regularization
        self.temporal_layers = nn.ModuleList()
        for i in range(num_temporal_layers):
            # Use smaller dilation rates for stability
            dilation = min(2 ** i, 4)  # Cap dilation at 4
            layer = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, 
                         dilation=dilation, padding=dilation),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5)
            )
            self.temporal_layers.append(layer)
        
        # Multi-modal signal embeddings
        # 1. Phase embedding (6 surgical phases)
        self.phase_embedding = nn.Embedding(6, hidden_dim // 4)
        self.phase_proj = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.2)
        )
        
        # 2. Time-to-next-phase embedding (anticipation signals for 6 phases)
        self.anticipation_embedding = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),  # 6 phase anticipation values
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.2)
        )
        
        # 3. Completion percentage embedding
        self.completion_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),  # Map percentage [0,1] to embedding
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.2)
        )
        
        # 4. Previous context embedding (historical phases)
        self.prev_context_embedding = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),  # Previous anticipation context
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.2)
        )
        
        # Multi-modal cross-attention mechanisms
        self.phase_cross_attention = CompletionCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        self.anticipation_cross_attention = CompletionCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        self.completion_cross_attention = CompletionCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        self.context_cross_attention = CompletionCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Multi-modal fusion layer
        self.multi_modal_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # Combine 4 attention outputs
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.3)
        )
        
        # Final feature fusion after multi-modal attention
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.3)
        )
        
        # Dual-stream architecture: separate feature processing for each arm
        self.left_arm_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.7)
        )
        
        self.right_arm_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.7)
        )
        
        # Left arm classification heads (input is hidden_dim//2 from arm projections)
        self.left_verb_head = self._make_verb_classifier(hidden_dim // 2, dropout)
        self.left_target_head = self._make_target_classifier(hidden_dim // 2, dropout)
        
        # Right arm classification heads (input is hidden_dim//2 from arm projections)
        self.right_verb_head = self._make_verb_classifier(hidden_dim // 2, dropout)
        self.right_target_head = self._make_target_classifier(hidden_dim // 2, dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _make_verb_classifier(self, in_dim: int, dropout: float) -> nn.Module:
        """Create verb classifier (4 classes)."""
        return nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.LayerNorm(in_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 4, NUM_VERBS)
        )
    
    def _make_target_classifier(self, in_dim: int, dropout: float) -> nn.Module:
        """Create target classifier (14 classes) - needs more capacity."""
        return nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, in_dim // 4),
            nn.LayerNorm(in_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 4, NUM_TARGETS)
        )
    
    def _init_weights(self):
        """Initialize classifier weights with Xavier initialization."""
        modules_to_init = [
            self.feature_proj, self.phase_proj, self.anticipation_embedding,
            self.completion_embedding, self.prev_context_embedding,
            self.multi_modal_fusion, self.feature_fusion,
            self.left_arm_proj, self.right_arm_proj,
            self.left_verb_head, self.left_target_head,
            self.right_verb_head, self.right_target_head
        ]
        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    # Use smaller initialization for stability
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def _process_multi_modal_inputs(self, 
                                   completion: torch.Tensor,
                                   phase_label: torch.Tensor,
                                   time_to_next_phase: torch.Tensor,
                                   last_anticipation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process multi-modal inputs from dataset.
        
        Args:
            completion: [B] or [B, T] completion percentage
            phase_label: [B] current phase labels (0-5)
            time_to_next_phase: [B, 6] time to next phase for each phase class
            last_anticipation: [B, 6] previous anticipation values
            
        Returns:
            Dictionary of processed inputs ready for embeddings
        """
        # Process completion
        if completion.dim() == 1:
            completion_processed = completion.unsqueeze(1)  # [B, 1]
        elif completion.dim() == 2:
            completion_processed = completion[:, -1:]  # [B, 1] (use last frame)
        else:
            raise ValueError(f"Unexpected completion shape: {completion.shape}")
        
        # Ensure phase labels are long for embedding
        phase_processed = phase_label.long()  # [B]
        
        # Process anticipation signals - clamp to reasonable range
        time_to_next_processed = torch.clamp(time_to_next_phase, min=-1.0, max=10.0)  # [B, 6]
        last_anticipation_processed = torch.clamp(last_anticipation, min=-1.0, max=10.0)  # [B, 6]
        
        return {
            'completion': completion_processed,  # [B, 1]
            'phase': phase_processed,           # [B]
            'anticipation': time_to_next_processed,    # [B, 6]
            'prev_context': last_anticipation_processed  # [B, 6]
        }
    
    def forward(self, 
                frames: torch.Tensor, 
                completion: torch.Tensor,
                phase_label: torch.Tensor,
                time_to_next_phase: torch.Tensor,
                last_anticipation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-modal cross-attention.
        
        Args:
            frames: [B, T, C, H, W] where T=seq_len
            completion: [B] or [B, T] completion percentage from dataset
            phase_label: [B] current phase labels (0-5)
            time_to_next_phase: [B, 6] time to next phase for each phase class
            last_anticipation: [B, 6] previous anticipation values
            
        Returns:
            Dictionary with logits for each arm and component
        """
        B, T, C, H, W = frames.shape
        assert T == self.seq_len, f"Expected {self.seq_len} frames, got {T}"
        device = frames.device
        
        # Extract visual features
        # Reshape to process all frames at once: [B*T, C, H, W]
        frames_flat = frames.view(B * T, C, H, W)
        
        # Extract features using backbone
        with torch.set_grad_enabled(self.backbone.training):
            visual_features = self.backbone(frames_flat)  # [B*T, 2048, H', W']
        
        # Spatial pooling
        pooled_features = self.spatial_pool(visual_features)  # [B*T, 2048, 1, 1]
        pooled_features = pooled_features.flatten(1)  # [B*T, 2048]
        
        # Reshape back to sequence: [B, T, 2048]
        sequence_features = pooled_features.view(B, T, -1)
        
        # Project to hidden dimension
        projected_features = []
        for t in range(T):
            proj_feat = self.feature_proj(sequence_features[:, t])  # [B, hidden_dim]
            projected_features.append(proj_feat)
        projected_features = torch.stack(projected_features, dim=2)  # [B, hidden_dim, T]
        
        # Temporal modeling with 1D convolutions and residual connections
        temporal_features = projected_features
        for temporal_layer in self.temporal_layers:
            residual = temporal_features
            temporal_features = temporal_layer(temporal_features)
            # Add residual connection
            temporal_features = temporal_features + residual
        
        # Process multi-modal inputs from dataset
        modal_inputs = self._process_multi_modal_inputs(
            completion, phase_label, time_to_next_phase, last_anticipation
        )
        
        # Create embeddings for each modality
        # 1. Phase embedding
        phase_embed = self.phase_embedding(modal_inputs['phase'])  # [B, hidden_dim//4]
        phase_queries = self.phase_proj(phase_embed).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # 2. Anticipation embedding  
        anticipation_queries = self.anticipation_embedding(modal_inputs['anticipation']).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # 3. Completion embedding
        completion_queries = self.completion_embedding(modal_inputs['completion']).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # 4. Previous context embedding
        context_queries = self.prev_context_embedding(modal_inputs['prev_context']).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Apply multi-modal cross-attention
        # Each modality attends to temporal features independently
        phase_attended = self.phase_cross_attention(phase_queries, temporal_features)           # [B, hidden_dim]
        anticipation_attended = self.anticipation_cross_attention(anticipation_queries, temporal_features)  # [B, hidden_dim]
        completion_attended = self.completion_cross_attention(completion_queries, temporal_features)        # [B, hidden_dim]
        context_attended = self.context_cross_attention(context_queries, temporal_features)     # [B, hidden_dim]
        
        # Fuse all multi-modal attended features
        multi_modal_features = torch.cat([
            phase_attended, anticipation_attended, 
            completion_attended, context_attended
        ], dim=1)  # [B, hidden_dim * 4]
        
        fused_features = self.multi_modal_fusion(multi_modal_features)  # [B, hidden_dim]
        
        # Final feature fusion
        global_features = self.feature_fusion(fused_features)  # [B, hidden_dim]
        
        # Dual-stream processing
        left_features = self.left_arm_proj(global_features)   # [B, hidden_dim//2]
        right_features = self.right_arm_proj(global_features) # [B, hidden_dim//2]
        
        # Classification for each arm
        left_verb_logits = self.left_verb_head(left_features)
        left_target_logits = self.left_target_head(left_features)
        
        right_verb_logits = self.right_verb_head(right_features)
        right_target_logits = self.right_target_head(right_features)
        
        return {
            'left_verb_logits': left_verb_logits,
            'left_target_logits': left_target_logits,
            'right_verb_logits': right_verb_logits,
            'right_target_logits': right_target_logits
        }


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def calculate_dual_arm_metrics(outputs: Dict[str, torch.Tensor], 
                              left_targets: torch.Tensor,
                              right_targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate comprehensive dual-arm metrics.
    
    Args:
        outputs: Dictionary with logits for each arm and component
        left_targets: [B, 3] ground truth left arm triplet (verb, subject, target)
        right_targets: [B, 3] ground truth right arm triplet (verb, subject, target)
    
    Returns:
        Dictionary of metrics
    """
    # Predictions
    left_verb_pred = outputs['left_verb_logits'].argmax(dim=1)
    left_target_pred = outputs['left_target_logits'].argmax(dim=1)
    right_verb_pred = outputs['right_verb_logits'].argmax(dim=1)
    right_target_pred = outputs['right_target_logits'].argmax(dim=1)
    
    # Ground truth (we use verb and target, ignoring subject since it's implicit)
    left_verb_gt = left_targets[:, 0]      # verb index
    left_target_gt = left_targets[:, 2]    # target index (skip subject at index 1)
    right_verb_gt = right_targets[:, 0]    # verb index  
    right_target_gt = right_targets[:, 2]  # target index (skip subject at index 1)
    
    # Individual component accuracies
    left_verb_acc = (left_verb_pred == left_verb_gt).float().mean()
    left_target_acc = (left_target_pred == left_target_gt).float().mean()
    right_verb_acc = (right_verb_pred == right_verb_gt).float().mean()
    right_target_acc = (right_target_pred == right_target_gt).float().mean()
    
    # Arm-level accuracies (both verb and target correct for each arm)
    left_arm_acc = ((left_verb_pred == left_verb_gt) & 
                   (left_target_pred == left_target_gt)).float().mean()
    right_arm_acc = ((right_verb_pred == right_verb_gt) & 
                     (right_target_pred == right_target_gt)).float().mean()
    
    # Complete dual-arm accuracy (all components correct)
    complete_acc = ((left_verb_pred == left_verb_gt) & 
                   (left_target_pred == left_target_gt) & 
                   (right_verb_pred == right_verb_gt) & 
                   (right_target_pred == right_target_gt)).float().mean()
    
    return {
        'left_verb_acc': float(left_verb_acc),
        'left_target_acc': float(left_target_acc),
        'right_verb_acc': float(right_verb_acc),
        'right_target_acc': float(right_target_acc),
        'left_arm_acc': float(left_arm_acc),
        'right_arm_acc': float(right_arm_acc),
        'complete_acc': float(complete_acc),
        'avg_verb_acc': float((left_verb_acc + right_verb_acc) / 2.0),
        'avg_target_acc': float((left_target_acc + right_target_acc) / 2.0),
        'avg_arm_acc': float((left_arm_acc + right_arm_acc) / 2.0)
    }


def calculate_dual_arm_map_metrics(outputs: Dict[str, torch.Tensor],
                                  left_targets: torch.Tensor,
                                  right_targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate Average Precision (mAP) metrics for dual-arm prediction.
    
    Args:
        outputs: Dictionary with logits for each arm and component
        left_targets: [B, 3] ground truth left arm triplet
        right_targets: [B, 3] ground truth right arm triplet
    
    Returns:
        Dictionary of mAP metrics
    """
    def safe_average_precision_multiclass(y_true, y_scores):
        """Calculate multiclass AP with proper handling."""
        try:
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                # Only one class present, return accuracy as fallback
                return float(np.mean(y_true == np.argmax(y_scores, axis=1)))
            
            # Calculate per-class AP and average
            aps = []
            for class_id in range(y_scores.shape[1]):
                # Create binary labels for this class
                y_binary = (y_true == class_id).astype(int)
                if len(np.unique(y_binary)) > 1:  # Both positive and negative samples
                    ap = average_precision_score(y_binary, y_scores[:, class_id])
                    aps.append(ap)
                else:
                    # Only one class present, use accuracy
                    aps.append(float(np.mean(y_binary == (y_scores[:, class_id] > 0.5))))
            
            return float(np.mean(aps))
        except Exception as e:
            print(f"Warning: mAP calculation failed: {e}")
            return 0.0
    
    # Convert to numpy for sklearn
    left_verb_probs = F.softmax(outputs['left_verb_logits'], dim=1).cpu().numpy()
    left_target_probs = F.softmax(outputs['left_target_logits'], dim=1).cpu().numpy()
    right_verb_probs = F.softmax(outputs['right_verb_logits'], dim=1).cpu().numpy()
    right_target_probs = F.softmax(outputs['right_target_logits'], dim=1).cpu().numpy()
    
    left_verb_gt = left_targets[:, 0].cpu().numpy()
    left_target_gt = left_targets[:, 2].cpu().numpy()
    right_verb_gt = right_targets[:, 0].cpu().numpy()
    right_target_gt = right_targets[:, 2].cpu().numpy()
    
    # Calculate mAP for each component
    map_metrics = {}
    
    # Individual component mAPs using fixed multiclass AP calculation
    map_metrics['left_verb_map'] = safe_average_precision_multiclass(left_verb_gt, left_verb_probs)
    map_metrics['left_target_map'] = safe_average_precision_multiclass(left_target_gt, left_target_probs)
    map_metrics['right_verb_map'] = safe_average_precision_multiclass(right_verb_gt, right_verb_probs)
    map_metrics['right_target_map'] = safe_average_precision_multiclass(right_target_gt, right_target_probs)
    
    # Average mAPs
    map_metrics['avg_verb_map'] = (map_metrics['left_verb_map'] + map_metrics['right_verb_map']) / 2.0
    map_metrics['avg_target_map'] = (map_metrics['left_target_map'] + map_metrics['right_target_map']) / 2.0
    map_metrics['overall_map'] = np.mean(list(map_metrics.values()))
    
    return map_metrics


def save_detailed_dual_arm_metrics(metrics: Dict[str, float], map_metrics: Dict[str, float],
                                  epoch: int, split: str = "test") -> None:
    """Save detailed metrics to text file."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = EVAL_ROOT / f"dual_arm_compl_attn_metrics_{split}_epoch{epoch}_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Dual-Arm Triplet Prediction with Completion Attention - {split.upper()} Split\n")
        f.write("=" * 70 + "\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: DualArmTripletCompletionPredictor\n")
        f.write(f"Enhancement: Completion percentage cross-attention\n\n")
        
        f.write("ACCURACY METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Left Arm Verb Accuracy:    {metrics['left_verb_acc']:.4f}\n")
        f.write(f"Left Arm Target Accuracy:  {metrics['left_target_acc']:.4f}\n")
        f.write(f"Left Arm Complete:         {metrics['left_arm_acc']:.4f}\n\n")
        
        f.write(f"Right Arm Verb Accuracy:   {metrics['right_verb_acc']:.4f}\n")
        f.write(f"Right Arm Target Accuracy: {metrics['right_target_acc']:.4f}\n")
        f.write(f"Right Arm Complete:        {metrics['right_arm_acc']:.4f}\n\n")
        
        f.write(f"Average Verb Accuracy:     {metrics['avg_verb_acc']:.4f}\n")
        f.write(f"Average Target Accuracy:   {metrics['avg_target_acc']:.4f}\n")
        f.write(f"Average Arm Accuracy:      {metrics['avg_arm_acc']:.4f}\n")
        f.write(f"Complete Dual-Arm Acc:     {metrics['complete_acc']:.4f}\n\n")
        
        f.write("AVERAGE PRECISION (mAP) METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Left Arm Verb mAP:         {map_metrics['left_verb_map']:.4f}\n")
        f.write(f"Left Arm Target mAP:       {map_metrics['left_target_map']:.4f}\n")
        f.write(f"Right Arm Verb mAP:        {map_metrics['right_verb_map']:.4f}\n")
        f.write(f"Right Arm Target mAP:      {map_metrics['right_target_map']:.4f}\n\n")
        
        f.write(f"Average Verb mAP:          {map_metrics['avg_verb_map']:.4f}\n")
        f.write(f"Average Target mAP:        {map_metrics['avg_target_map']:.4f}\n")
        f.write(f"Overall mAP:               {map_metrics['overall_map']:.4f}\n")
    
    print(f"📊 Detailed completion attention metrics saved to: {filename}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, 
            split: str = "val", epoch: int = 0) -> Dict[str, float]:
    """Evaluate model on given data loader."""
    model.eval()
    
    all_losses = []
    all_outputs = []
    all_left_targets = []
    all_right_targets = []
    all_completion_percentages = []
    all_phase_labels = []
    all_anticipation_signals = []
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    for frames, meta in loader:
        frames = frames.to(device)
        left_triplet_gt = meta["triplet_left_classification"].to(device).long()   # [B, 3]
        right_triplet_gt = meta["triplet_right_classification"].to(device).long() # [B, 3]
        
        # Get multi-modal signals from dataset metadata
        if "phase_completition_dense" in meta:
            completion = meta["phase_completition_dense"].to(device).float()  # [B, T]
        else:
            completion = meta["phase_completition"].to(device).float()        # [B]
            
        phase_label = meta["phase_label"].to(device).float()                 # [B]
        time_to_next_phase = meta["time_to_next_phase"].to(device).float()   # [B, 6]
        last_anticipation = meta["last_anticipation"].to(device).float()     # [B, 6]
        
        outputs = model(frames, completion, phase_label, time_to_next_phase, last_anticipation)
        
        # Calculate losses for each component with arm-specific weighting
        left_verb_loss = criterion(outputs['left_verb_logits'], left_triplet_gt[:, 0])
        left_target_loss = criterion(outputs['left_target_logits'], left_triplet_gt[:, 2])
        right_verb_loss = criterion(outputs['right_verb_logits'], right_triplet_gt[:, 0])
        right_target_loss = criterion(outputs['right_target_logits'], right_triplet_gt[:, 2])
        
        # Balanced loss weighting for dual-arm prediction
        verb_weight = 1.0
        target_weight = 1.2  # Slightly higher weight for more complex target prediction
        
        total_loss = (verb_weight * (left_verb_loss + right_verb_loss) + 
                     target_weight * (left_target_loss + right_target_loss)) / (2 * (verb_weight + target_weight))
        
        all_losses.append(total_loss.item())
        all_outputs.append({k: v.cpu() for k, v in outputs.items()})
        all_left_targets.append(left_triplet_gt.cpu())
        all_right_targets.append(right_triplet_gt.cpu())
        
        # Track multi-modal signals for analysis (from input data)
        completion_for_stats = completion[:, -1] if completion.dim() == 2 else completion
        all_completion_percentages.append(completion_for_stats.cpu())
        all_phase_labels.append(phase_label.cpu())
        all_anticipation_signals.append(time_to_next_phase.cpu())
    
    # Concatenate all results
    concat_outputs = {}
    for key in all_outputs[0].keys():
        concat_outputs[key] = torch.cat([out[key] for out in all_outputs], dim=0)
    
    concat_left_targets = torch.cat(all_left_targets, dim=0)
    concat_right_targets = torch.cat(all_right_targets, dim=0)
    
    # Calculate accuracy metrics
    acc_metrics = calculate_dual_arm_metrics(concat_outputs, concat_left_targets, concat_right_targets)
    acc_metrics['loss'] = np.mean(all_losses)
    
    # Calculate mAP metrics
    map_metrics = calculate_dual_arm_map_metrics(concat_outputs, concat_left_targets, concat_right_targets)
    
    # Combine all metrics
    final_metrics = {**acc_metrics, **map_metrics}
    
    # Add multi-modal signal statistics
    if all_completion_percentages:
        all_completion = torch.cat(all_completion_percentages, dim=0)
        final_metrics['completion_mean'] = float(all_completion.mean())
        final_metrics['completion_std'] = float(all_completion.std())
        
    if all_phase_labels:
        all_phases = torch.cat(all_phase_labels, dim=0)
        final_metrics['phase_mean'] = float(all_phases.mean())
        
    if all_anticipation_signals:
        all_anticipation = torch.cat(all_anticipation_signals, dim=0)
        final_metrics['anticipation_mean'] = float(all_anticipation.mean())
    
    if not split.endswith('no_ema'):  # Only print for main validation
        print(f"[{split.upper()}] Loss: {final_metrics['loss']:.4f} | "
              f"L-Arm: {final_metrics['left_arm_acc']:.3f} | "
              f"R-Arm: {final_metrics['right_arm_acc']:.3f} | "
              f"Complete: {final_metrics['complete_acc']:.3f} | "
              f"mAP: {final_metrics['overall_map']:.3f}")
        
        if 'completion_mean' in final_metrics:
            print(f"         Multi-modal - Completion: μ={final_metrics['completion_mean']:.3f}, "
                  f"σ={final_metrics['completion_std']:.3f}", end="")
            if 'phase_mean' in final_metrics:
                print(f" | Phase: μ={final_metrics['phase_mean']:.2f}", end="")
            if 'anticipation_mean' in final_metrics:
                print(f" | Anticipation: μ={final_metrics['anticipation_mean']:.3f}", end="")
            print()  # New line
    
    # Save detailed metrics for test split
    if split == "test":
        save_detailed_dual_arm_metrics(acc_metrics, map_metrics, epoch, split)
    
    return final_metrics


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, 
                   scheduler: optim.lr_scheduler._LRScheduler,
                   epoch: int, best_acc: float, path: Path):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, 
                   scheduler: optim.lr_scheduler._LRScheduler,
                   path: Path, device: torch.device) -> Tuple[int, float]:
    """Load training checkpoint. Returns (start_epoch, best_acc)."""
    if not path.exists():
        return 1, 0.0
    
    try:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Resumed from epoch {checkpoint['epoch']}, best_acc: {best_acc:.4f}")
        return start_epoch, best_acc
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return 1, 0.0


class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
         device: torch.device):
    """Main training loop for dual-arm prediction with completion attention."""
    
    # Use focused loss functions for effective learning
    print("🎯 Using focused loss functions for effective learning with multi-modal attention...")
    left_verb_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # Reduced smoothing
    left_target_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    right_verb_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    right_target_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # Use Adam with slightly more momentum for better learning dynamics
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, 
                           betas=(0.9, 0.98), eps=1e-8)  # Slightly higher beta2 for stability
    
    # Warm-up scheduler with cosine annealing for better learning
    def lr_schedule(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS  # Linear warmup
        else:
            # Cosine decay after warmup
            progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    
    # Exponential Moving Average
    ema = EMA(model, decay=EMA_DECAY)
    
    # Try to resume from checkpoint
    start_epoch, best_acc = load_checkpoint(model, optimizer, scheduler, LAST_CKPT_PATH, device)
    
    print(f"Starting dual-arm multi-modal attention training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        
        epoch_losses = []
        epoch_metrics = []
        epoch_completion_stats = []
        
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 50)
        
        for batch_idx, (frames, meta) in enumerate(train_loader):
            frames = frames.to(device)
            left_triplet_gt = meta["triplet_left_classification"].to(device).long()   # [B, 3]
            right_triplet_gt = meta["triplet_right_classification"].to(device).long() # [B, 3]
            
            # Get multi-modal signals from dataset metadata
            if "phase_completition_dense" in meta:
                completion = meta["phase_completition_dense"].to(device).float()  # [B, T]
            else:
                completion = meta["phase_completition"].to(device).float()        # [B]
                
            phase_label = meta["phase_label"].to(device).float()                 # [B]
            time_to_next_phase = meta["time_to_next_phase"].to(device).float()   # [B, 6]
            last_anticipation = meta["last_anticipation"].to(device).float()     # [B, 6]
            
            outputs = model(frames, completion, phase_label, time_to_next_phase, last_anticipation)
            
            # Calculate losses for each component using weighted criteria
            left_verb_loss = left_verb_criterion(outputs['left_verb_logits'], left_triplet_gt[:, 0])
            left_target_loss = left_target_criterion(outputs['left_target_logits'], left_triplet_gt[:, 2])
            right_verb_loss = right_verb_criterion(outputs['right_verb_logits'], right_triplet_gt[:, 0])
            right_target_loss = right_target_criterion(outputs['right_target_logits'], right_triplet_gt[:, 2])
            
            # Balanced dual-arm loss weighting
            verb_weight = 1.0
            target_weight = 1.2  # Slightly higher weight for targets
            
            total_loss = (verb_weight * (left_verb_loss + right_verb_loss) + 
                         target_weight * (left_target_loss + right_target_loss)) / (2 * (verb_weight + target_weight))
            
            # Add multi-modal statistics tracking (from input data)
            completion_for_stats = completion[:, -1] if completion.dim() == 2 else completion
            completion_mean = completion_for_stats.mean().item()
            completion_std = completion_for_stats.std().item()
            phase_mean = phase_label.float().mean().item()
            anticipation_mean = time_to_next_phase.mean().item()
            epoch_completion_stats.append((completion_mean, completion_std, phase_mean, anticipation_mean))
            
            # Add detailed loss logging every 50 batches for debugging
            # if batch_idx % 50 == 0:
            #     print(f"\n    Loss breakdown - L_verb: {left_verb_loss.item():.3f}, "
            #           f"L_target: {left_target_loss.item():.3f}, "
            #           f"R_verb: {right_verb_loss.item():.3f}, "
            #           f"R_target: {right_target_loss.item():.3f}")
            #     if epoch_completion_stats:
            #         comp_mean, comp_std = epoch_completion_stats[-1]
            #         print(f"    Completion: μ={comp_mean:.3f}, σ={comp_std:.3f}")
            
            total_loss.backward()
            
            # Looser gradient clipping to allow more learning
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            # Monitor gradient health
            if batch_idx % 50 == 0:
                print(f"Grad norm: {grad_norm:.3f}", end=" | ")
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Update EMA
            ema.update()
            
            # Track metrics with explosion detection
            loss_value = total_loss.item()
            if loss_value > 1000.0:  # Catch explosive loss early
                print(f"🚨 EXPLOSIVE LOSS DETECTED: {loss_value:.2f} - Stopping training")
                return model  # Early exit
            
            epoch_losses.append(loss_value)
            with torch.no_grad():
                metrics = calculate_dual_arm_metrics(outputs, left_triplet_gt, right_triplet_gt)
                epoch_metrics.append(metrics)
            
            # Print progress with more details
            if batch_idx % 20 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {total_loss.item():.4f} | "
                      f"Complete Acc: {metrics['complete_acc']:.3f} | "
                      f"Avg Arm: {metrics['avg_arm_acc']:.3f} | "
                      f"LR: {current_lr:.2e}")
        
        # Update learning rate after each epoch
        scheduler.step()
        
        # Calculate epoch averages
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
        
        # Calculate multi-modal statistics
        if epoch_completion_stats:
            avg_completion_mean = np.mean([stats[0] for stats in epoch_completion_stats])
            avg_completion_std = np.mean([stats[1] for stats in epoch_completion_stats])
            avg_phase_mean = np.mean([stats[2] for stats in epoch_completion_stats])
            avg_anticipation_mean = np.mean([stats[3] for stats in epoch_completion_stats])
        else:
            avg_completion_mean = avg_completion_std = avg_phase_mean = avg_anticipation_mean = 0.0
        
        print(f"\nTRAIN - Loss: {avg_loss:.4f} | "
              f"Complete Acc: {avg_metrics['complete_acc']:.3f} | "
              f"Avg Arm: {avg_metrics['avg_arm_acc']:.3f}")
        print(f"        Multi-modal - Completion: μ={avg_completion_mean:.3f}, σ={avg_completion_std:.3f} | "
              f"Phase: μ={avg_phase_mean:.3f} | Anticipation: μ={avg_anticipation_mean:.3f}")
        
        # Validation every epoch to monitor overfitting closely
        # Validation with EMA weights
        ema.apply_shadow()
        val_metrics = evaluate(model, val_loader, device, "val", epoch)
        ema.restore()
        
        # Also evaluate without EMA for comparison
        val_metrics_no_ema = evaluate(model, val_loader, device, "val_no_ema", epoch)
        
        # Store validation metrics
        train.last_val_metrics = val_metrics
        
        # Log to wandb
        if USE_WANDB:
            log_dict = {
                'epoch': epoch,
                'train_loss': avg_loss,
                'train_complete_acc': avg_metrics['complete_acc'],
                'train_left_arm_acc': avg_metrics['left_arm_acc'],
                'train_right_arm_acc': avg_metrics['right_arm_acc'],
                'train_avg_arm_acc': avg_metrics['avg_arm_acc'],
                'train_completion_mean': avg_completion_mean,
                'train_completion_std': avg_completion_std,
                'train_phase_mean': avg_phase_mean,
                'train_anticipation_mean': avg_anticipation_mean,
                'val_loss': val_metrics['loss'],
                'val_complete_acc': val_metrics['complete_acc'],
                'val_left_arm_acc': val_metrics['left_arm_acc'],
                'val_right_arm_acc': val_metrics['right_arm_acc'],
                'val_avg_arm_acc': val_metrics['avg_arm_acc'],
                'val_no_ema_complete_acc': val_metrics_no_ema['complete_acc'],
                'learning_rate': scheduler.get_last_lr()[0]
            }
            # Add validation multi-modal stats if available
            if 'completion_mean' in val_metrics:
                log_dict.update({
                    'val_completion_mean': val_metrics['completion_mean'],
                    'val_completion_std': val_metrics['completion_std']
                })
            if 'phase_mean' in val_metrics:
                log_dict['val_phase_mean'] = val_metrics['phase_mean']
            if 'anticipation_mean' in val_metrics:
                log_dict['val_anticipation_mean'] = val_metrics['anticipation_mean']
            # Add mAP metrics if available
            if 'overall_map' in val_metrics:
                log_dict.update({
                    'val_overall_map': val_metrics['overall_map'],
                    'val_avg_verb_map': val_metrics['avg_verb_map'],
                    'val_avg_target_map': val_metrics['avg_target_map']
                })
            wandb.log(log_dict)
        
        # Save best model based on complete accuracy
        if val_metrics['complete_acc'] > best_acc:
            best_acc = val_metrics['complete_acc']
            # Save EMA weights as the best model
            ema.apply_shadow()
            torch.save(model.state_dict(), CKPT_PATH)
            ema.restore()
            print(f"✅ New best multi-modal attention model saved! Complete accuracy: {best_acc:.4f}")
        
        # Early stopping if validation performance plateaus
        if epoch > 20 and val_metrics['complete_acc'] < best_acc * 0.95:
            patience_counter = getattr(train, 'patience_counter', 0) + 1
            train.patience_counter = patience_counter
            if patience_counter >= 12:  # More patience to allow learning
                print("Early stopping: validation performance has plateaued")
                break
        else:
            train.patience_counter = 0
        
        # Check if we're making progress on training loss
        if epoch > 5:
            recent_losses = getattr(train, 'recent_losses', [])
            recent_losses.append(avg_loss)
            if len(recent_losses) > 5:
                recent_losses.pop(0)
            train.recent_losses = recent_losses
            
            # If training loss hasn't improved in 5 epochs, something is wrong
            if len(recent_losses) == 5 and abs(max(recent_losses) - min(recent_losses)) < 0.01:
                print(f"⚠️  Training loss stuck at {avg_loss:.4f} - consider adjusting hyperparameters")
        else:
            train.recent_losses = [avg_loss]
        
        # Save checkpoint for resuming
        save_checkpoint(model, optimizer, scheduler, epoch, best_acc, LAST_CKPT_PATH)
        
        # Memory cleanup
        if epoch % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()


def main():
    print("🚀 Starting Dual-Arm Triplet Predictor with Multi-Modal Cross-Attention Training")
    print(f"Dataset: {ROOT_DIR}")
    print(f"Dual-arm dimensions: Verbs={NUM_VERBS}, Targets={NUM_TARGETS}")
    print("Architecture: Multi-modal cross-attention + dual-arm prediction")
    print("Enhancements: Phase-aware, Anticipation-guided, Context-enriched temporal modeling")
    print("Signals: Phase (0-5) + Time-to-Next-Phase (6D) + Completion (0-1) + Previous Context (6D)")
    
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize wandb
    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            name=f"{WANDB_RUN_NAME}",
            config={
                'seq_len': SEQ_LEN,
                'stride': STRIDE,
                'batch_size': BATCH_SIZE,
                'learning_rate': LR,
                'weight_decay': WEIGHT_DECAY,
                'hidden_dim': HIDDEN_DIM,
                'dropout': DROPOUT,
                'num_temporal_layers': NUM_TEMPORAL_LAYERS,
                'num_attention_heads': NUM_ATTENTION_HEADS,
                'epochs': EPOCHS,
                'warmup_epochs': WARMUP_EPOCHS,
                'ema_decay': EMA_DECAY,
                'architecture': 'DualArmTripletMultiModalPredictor',
                'backbone': 'ResNet50_PartiallyFrozen',
                'prediction_mode': 'dual_arm_verb_target',
                'enhancement': 'multi_modal_cross_attention',
                'attention_mechanisms': ['Phase', 'Anticipation', 'Completion', 'Context'],
                'signals': ['phase_label', 'time_to_next_phase', 'completion_percentage', 'previous_anticipation'],
                'improvements': ['EMA', 'LabelSmoothing', 'MultiModalAttention', 'DualStream', 'BalancedLoss', 'MaxDataUtilization'],
                'loss_weights': {'verb': 1.0, 'target': 1.2}
            }
        )
    
    # Create datasets
    print("\n📁 Loading datasets...")
    train_ds = PegAndRing(
        ROOT_DIR,
        mode="train",
        seq_len=SEQ_LEN,
        stride=STRIDE,
        time_unit=TIME_UNIT,
        augment=False,
        force_triplets=True
    )
    
    val_ds = PegAndRing(
        ROOT_DIR,
        mode="val", 
        seq_len=SEQ_LEN,
        stride=STRIDE,
        time_unit=TIME_UNIT,
        augment=False,
        force_triplets=True
    )
    
    test_ds = PegAndRing(
        ROOT_DIR,
        mode="val",  # Using val as test since no separate test split
        seq_len=SEQ_LEN,
        stride=STRIDE,
        time_unit=TIME_UNIT,
        augment=False,
        force_triplets=True
    )
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    
    # Create model
    print("\n🏗️  Building dual-arm completion attention model...")
    model = DualArmTripletMultiModalPredictor(
        seq_len=SEQ_LEN,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        num_temporal_layers=NUM_TEMPORAL_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test model with sample multi-modal data
    print("\n🧪 Testing model with sample multi-modal data...")
    sample_frames = torch.randn(2, SEQ_LEN, 3, 224, 224).to(device)
    sample_completion_scalar = torch.rand(2).to(device)  # [B]
    sample_completion_dense = torch.rand(2, SEQ_LEN).to(device)  # [B, T]
    sample_phase_label = torch.randint(0, 6, (2,)).to(device)  # [B] phase 0-5
    sample_time_to_next_phase = torch.rand(2, 6).to(device)  # [B, 6]
    sample_last_anticipation = torch.rand(2, 6).to(device)  # [B, 6]
    
    with torch.no_grad():
        # Test with scalar completion
        outputs_scalar = model(sample_frames, sample_completion_scalar, 
                             sample_phase_label, sample_time_to_next_phase, sample_last_anticipation)
        print(f"✅ Model works with multi-modal inputs:")
        print(f"   Completion: {sample_completion_scalar.shape}")
        print(f"   Phase: {sample_phase_label.shape}")
        print(f"   Anticipation: {sample_time_to_next_phase.shape}")
        print(f"   Context: {sample_last_anticipation.shape}")
        
        # Test with dense completion  
        outputs_dense = model(sample_frames, sample_completion_dense,
                            sample_phase_label, sample_time_to_next_phase, sample_last_anticipation)
        print(f"✅ Model works with dense completion: {sample_completion_dense.shape}")
        
        # Verify output shapes
        for key, tensor in outputs_scalar.items():
            print(f"   Output {key}: {tensor.shape}")
    
    print("🧪 Multi-modal model interface test completed successfully!")
    
    # Train model
    if DO_TRAIN:
        print("\n🎯 Starting dual-arm multi-modal attention training...")
        train(model, train_loader, val_loader, device)
    
    # Load best model for testing
    if CKPT_PATH.exists():
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        print(f"\n✅ Loaded best multi-modal attention model from {CKPT_PATH}")
    
    # Final evaluation
    print("\n📊 Final dual-arm multi-modal attention evaluation...")
    test_metrics = evaluate(model, test_loader, device, "test", EPOCHS)
    
    print("\n" + "="*80)
    print("FINAL DUAL-ARM MULTI-MODAL ATTENTION TEST RESULTS:")
    print("="*80)
    print(f"Complete Dual-Arm Accuracy: {test_metrics['complete_acc']:.4f}")
    print(f"Left Arm Accuracy:          {test_metrics['left_arm_acc']:.4f}")
    print(f"Right Arm Accuracy:         {test_metrics['right_arm_acc']:.4f}")
    print(f"Average Arm Accuracy:       {test_metrics['avg_arm_acc']:.4f}")
    print()
    print("Component Accuracies:")
    print(f"  Left Verb:                {test_metrics['left_verb_acc']:.4f}")
    print(f"  Left Target:              {test_metrics['left_target_acc']:.4f}")
    print(f"  Right Verb:               {test_metrics['right_verb_acc']:.4f}")
    print(f"  Right Target:             {test_metrics['right_target_acc']:.4f}")
    print()
    print("Average Precision (mAP):")
    print(f"  Overall mAP:              {test_metrics['overall_map']:.4f}")
    print(f"  Average Verb mAP:         {test_metrics['avg_verb_map']:.4f}")
    print(f"  Average Target mAP:       {test_metrics['avg_target_map']:.4f}")
    print()
    if 'completion_mean' in test_metrics:
        print("Completion Percentage Statistics:")
        print(f"  Mean:                     {test_metrics['completion_mean']:.4f}")
        print(f"  Standard Deviation:       {test_metrics['completion_std']:.4f}")
    print("="*80)
    
    # Log final results to wandb
    if USE_WANDB:
        final_log = {
            'test_complete_acc': test_metrics['complete_acc'],
            'test_left_arm_acc': test_metrics['left_arm_acc'],
            'test_right_arm_acc': test_metrics['right_arm_acc'],
            'test_avg_arm_acc': test_metrics['avg_arm_acc'],
            'test_left_verb_acc': test_metrics['left_verb_acc'],
            'test_left_target_acc': test_metrics['left_target_acc'],
            'test_right_verb_acc': test_metrics['right_verb_acc'],
            'test_right_target_acc': test_metrics['right_target_acc'],
            'test_overall_map': test_metrics['overall_map'],
            'test_avg_verb_map': test_metrics['avg_verb_map'],
            'test_avg_target_map': test_metrics['avg_target_map']
        }
        if 'completion_mean' in test_metrics:
            final_log.update({
                'test_completion_mean': test_metrics['completion_mean'],
                'test_completion_std': test_metrics['completion_std']
            })
        wandb.log(final_log)
        wandb.finish()
    
    print("\n🎉 Dual-arm multi-modal attention training completed!")


if __name__ == "__main__":
    main()