#!/usr/bin/env python3
"""
Simple Triplet Predictor for Surgical Action Recognition

This implementation focuses on simplicity and effectiveness over complexity.
Key principles:
1. Single CNN backbone (ResNet50) for visual features
2. Simple temporal modeling with 1D convolutions
3. Direct triplet prediction without hierarchical complexity
4. Strong data augmentation and regularization
5. Focus on getting the basics right

The triplet structure: verb (4 classes), subject (3 classes), object (14 classes)
Examples: "reach(left_arm, green_peg)", "grasp(right_arm, red_ring)", etc.
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

# Triplet dimensions
NUM_VERBS = len(TRIPLET_VERBS)      # 4: reach, grasp, release, null
NUM_SUBJECTS = len(TRIPLET_SUBJECTS) # 3: left_arm, right_arm, null  
NUM_OBJECTS = len(TRIPLET_TARGETS)   # 14: pegs, rings, center, outside, arms, null

# Training hyperparameters - optimized for stability and generalization
SEQ_LEN = 8           # 8 frames = ~8 seconds at 1fps
STRIDE = 2            # More sparse sampling to reduce overfitting
BATCH_SIZE = 16       # Smaller batch size for better generalization on small dataset
NUM_WORKERS = 16
EPOCHS = 100          # Fewer epochs to prevent overfitting

LR = 1e-4            # Lower learning rate for more stable training
WEIGHT_DECAY = 5e-4   # Moderate regularization - too high was causing instability
WARMUP_EPOCHS = 5     # Shorter warmup
EMA_DECAY = 0.999     # Exponential moving average for model weights

# Model architecture hyperparameters
HIDDEN_DIM = 256      # Even smaller to prevent overfitting
DROPOUT = 0.4         # Higher dropout for better regularization
NUM_TEMPORAL_LAYERS = 3  # Fewer layers to reduce overfitting

# Control flags
DO_TRAIN = True
USE_WANDB = True
WANDB_PROJECT = "simple_triplet_predictor"
WANDB_RUN_NAME = "simple_effective_v1"

# Paths
CKPT_PATH = Path("simple_triplet_predictor_best.pth")
LAST_CKPT_PATH = Path("simple_triplet_predictor_last.pth")

# Evaluation directories
EVAL_ROOT = Path("results/simple_triplet_eval")
EVAL_ROOT.mkdir(parents=True, exist_ok=True)


class SimpleTripletPredictor(nn.Module):
    """
    Simple and effective triplet predictor with improved stability.
    
    Architecture:
    1. ResNet50 backbone for visual features
    2. Temporal 1D convolutions for sequence modeling  
    3. Global average pooling for sequence aggregation
    4. Three separate classification heads for verb/subject/object
    """
    
    def __init__(
        self,
        seq_len: int = SEQ_LEN,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
        num_temporal_layers: int = NUM_TEMPORAL_LAYERS
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
        
        # Freeze early layers to prevent overfitting
        for i, child in enumerate(self.backbone.children()):
            if i < 6:  # Freeze first 6 layers (conv1, bn1, relu, maxpool, layer1, layer2)
                for param in child.parameters():
                    param.requires_grad = False
        
        # Spatial pooling with dropout
        self.spatial_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(dropout * 0.5)  # Spatial dropout
        )
        
        # Feature projection with layer normalization and better regularization
        self.feature_proj = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),  # GELU often works better than ReLU
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
        
        # Global temporal pooling with attention
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classification heads with better regularization
        self.verb_head = self._make_classifier(hidden_dim, NUM_VERBS, dropout)
        self.subject_head = self._make_classifier(hidden_dim, NUM_SUBJECTS, dropout)
        self.object_head = self._make_classifier(hidden_dim, NUM_OBJECTS, dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _make_classifier(self, in_dim: int, num_classes: int, dropout: float) -> nn.Module:
        """Create a simple but effective classifier head with extra capacity for object classification."""
        if num_classes == NUM_OBJECTS:  # Object head needs more capacity (14 classes)
            return nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.LayerNorm(in_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(in_dim, in_dim // 2),
                nn.LayerNorm(in_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.7),
                nn.Linear(in_dim // 2, num_classes)
            )
        else:  # Verb and subject heads (fewer classes)
            return nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.LayerNorm(in_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(in_dim // 2, num_classes)
            )
    
    def _init_weights(self):
        """Initialize classifier weights with Xavier initialization."""
        for module in [self.feature_proj, self.verb_head, self.subject_head, self.object_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            frames: [B, T, C, H, W] where T=seq_len
            
        Returns:
            Dictionary with logits for each triplet component
        """
        B, T, C, H, W = frames.shape
        assert T == self.seq_len, f"Expected {self.seq_len} frames, got {T}"
        
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
            # Add residual connection with proper dimension matching
            temporal_features = temporal_features + residual
        
        # Attention-based temporal pooling instead of simple average
        attention_weights = self.temporal_attention(temporal_features)  # [B, 1, T]
        attended_features = temporal_features * attention_weights  # [B, hidden_dim, T]
        global_features = attended_features.sum(dim=2)  # [B, hidden_dim]
        
        # Classify each triplet component
        verb_logits = self.verb_head(global_features)
        subject_logits = self.subject_head(global_features)
        object_logits = self.object_head(global_features)
        
        return {
            'verb_logits': verb_logits,
            'subject_logits': subject_logits, 
            'object_logits': object_logits
        }


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def calculate_triplet_metrics(outputs: Dict[str, torch.Tensor], 
                            targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate comprehensive triplet metrics.
    
    Args:
        outputs: Dictionary with logits for each component
        targets: [B, 3] ground truth triplet labels (verb, subject, object)
    
    Returns:
        Dictionary of metrics
    """
    verb_pred = outputs['verb_logits'].argmax(dim=1)
    subject_pred = outputs['subject_logits'].argmax(dim=1)
    object_pred = outputs['object_logits'].argmax(dim=1)
    
    verb_gt = targets[:, 0]
    subject_gt = targets[:, 1] 
    object_gt = targets[:, 2]
    
    # Individual accuracies
    verb_acc = (verb_pred == verb_gt).float().mean()
    subject_acc = (subject_pred == subject_gt).float().mean()
    object_acc = (object_pred == object_gt).float().mean()
    
    # Combination accuracies
    verb_subject_acc = ((verb_pred == verb_gt) & (subject_pred == subject_gt)).float().mean()
    verb_object_acc = ((verb_pred == verb_gt) & (object_pred == object_gt)).float().mean()
    subject_object_acc = ((subject_pred == subject_gt) & (object_pred == object_gt)).float().mean()
    
    # Complete triplet accuracy
    complete_acc = ((verb_pred == verb_gt) & 
                   (subject_pred == subject_gt) & 
                   (object_pred == object_gt)).float().mean()
    
    return {
        'verb_acc': float(verb_acc),
        'subject_acc': float(subject_acc),
        'object_acc': float(object_acc),
        'verb_subject_acc': float(verb_subject_acc),
        'verb_object_acc': float(verb_object_acc),
        'subject_object_acc': float(subject_object_acc),
        'complete_acc': float(complete_acc),
        'avg_component_acc': float((verb_acc + subject_acc + object_acc) / 3.0)
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, 
            split: str = "val") -> Dict[str, float]:
    """Evaluate model on given data loader with improved stability."""
    model.eval()
    
    all_verb_preds = []
    all_subject_preds = []
    all_object_preds = []
    all_verb_gts = []
    all_subject_gts = []
    all_object_gts = []
    all_losses = []
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    for frames, meta in loader:
        frames = frames.to(device)
        triplet_gt = meta["triplet_classification"].to(device).long()  # [B, 3]
        
        outputs = model(frames)
        
        # Calculate losses with same weighting as training
        verb_loss = criterion(outputs['verb_logits'], triplet_gt[:, 0])
        subject_loss = criterion(outputs['subject_logits'], triplet_gt[:, 1])
        object_loss = criterion(outputs['object_logits'], triplet_gt[:, 2])
        
        verb_weight, subject_weight, object_weight = 1.0, 0.8, 1.5
        total_loss = (verb_weight * verb_loss + 
                     subject_weight * subject_loss + 
                     object_weight * object_loss) / (verb_weight + subject_weight + object_weight)
        
        all_losses.append(total_loss.item())
        
        # Collect predictions and ground truths
        all_verb_preds.extend(outputs['verb_logits'].argmax(dim=1).cpu().numpy())
        all_subject_preds.extend(outputs['subject_logits'].argmax(dim=1).cpu().numpy())
        all_object_preds.extend(outputs['object_logits'].argmax(dim=1).cpu().numpy())
        all_verb_gts.extend(triplet_gt[:, 0].cpu().numpy())
        all_subject_gts.extend(triplet_gt[:, 1].cpu().numpy())
        all_object_gts.extend(triplet_gt[:, 2].cpu().numpy())
    
    # Calculate metrics on full dataset for more stability
    all_verb_preds = np.array(all_verb_preds)
    all_subject_preds = np.array(all_subject_preds)
    all_object_preds = np.array(all_object_preds)
    all_verb_gts = np.array(all_verb_gts)
    all_subject_gts = np.array(all_subject_gts)
    all_object_gts = np.array(all_object_gts)
    
    # Individual accuracies
    verb_acc = (all_verb_preds == all_verb_gts).mean()
    subject_acc = (all_subject_preds == all_subject_gts).mean()
    object_acc = (all_object_preds == all_object_gts).mean()
    
    # Complete triplet accuracy
    complete_acc = ((all_verb_preds == all_verb_gts) & 
                   (all_subject_preds == all_subject_gts) & 
                   (all_object_preds == all_object_gts)).mean()
    
    final_metrics = {
        'verb_acc': float(verb_acc),
        'subject_acc': float(subject_acc),
        'object_acc': float(object_acc),
        'complete_acc': float(complete_acc),
        'avg_component_acc': float((verb_acc + subject_acc + object_acc) / 3.0),
        'loss': np.mean(all_losses)
    }
    
    if not split.endswith('no_ema'):  # Only print for main validation
        print(f"[{split.upper()}] Loss: {final_metrics['loss']:.4f} | "
              f"Verb: {final_metrics['verb_acc']:.3f} | "
              f"Subject: {final_metrics['subject_acc']:.3f} | "
              f"Object: {final_metrics['object_acc']:.3f} | "
              f"Complete: {final_metrics['complete_acc']:.3f}")
    
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
    """Main training loop with improved stability."""
    
    # Loss function with label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, 
                           betas=(0.9, 0.999), eps=1e-8)
    
    # Cosine annealing scheduler for smoother learning rate decay
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=LR * 0.01
    )
    
    # Exponential Moving Average for more stable validation
    ema = EMA(model, decay=EMA_DECAY)
    
    # Try to resume from checkpoint
    start_epoch, best_acc = load_checkpoint(model, optimizer, scheduler, LAST_CKPT_PATH, device)
    
    print(f"Starting training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        
        epoch_losses = []
        epoch_metrics = []
        
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 50)
        
        for batch_idx, (frames, meta) in enumerate(train_loader):
            frames = frames.to(device)
            triplet_gt = meta["triplet_classification"].to(device).long()  # [B, 3]
            
            optimizer.zero_grad()
            
            outputs = model(frames)
            
            # Calculate losses for each component
            verb_loss = criterion(outputs['verb_logits'], triplet_gt[:, 0])
            subject_loss = criterion(outputs['subject_logits'], triplet_gt[:, 1])
            object_loss = criterion(outputs['object_logits'], triplet_gt[:, 2])
            
            # Adaptive loss weighting - give more weight to harder components
            # Object is hardest (14 classes), subject is easiest (3 classes)
            verb_weight = 1.0
            subject_weight = 0.8  # Reduce weight for easiest component
            object_weight = 1.5   # Increase weight for hardest component
            
            total_loss = (verb_weight * verb_loss + 
                         subject_weight * subject_loss + 
                         object_weight * object_loss) / (verb_weight + subject_weight + object_weight)
            
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update EMA
            ema.update()
            
            # Track metrics
            epoch_losses.append(total_loss.item())
            with torch.no_grad():
                metrics = calculate_triplet_metrics(outputs, triplet_gt)
                epoch_metrics.append(metrics)
            
            # Print progress
            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {total_loss.item():.4f} | "
                      f"Complete Acc: {metrics['complete_acc']:.3f}")
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch averages
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
        
        print(f"\nTRAIN - Loss: {avg_loss:.4f} | "
              f"Complete Acc: {avg_metrics['complete_acc']:.3f}")
        
        # Validate less frequently to prevent validation overfitting
        if epoch % 2 == 0 or epoch <= 10:  # Validate every 2 epochs after epoch 10
            # Validation with EMA weights for more stable evaluation
            ema.apply_shadow()
            val_metrics = evaluate(model, val_loader, device, "val")
            ema.restore()
            
            # Also evaluate without EMA for comparison (but only log, don't print)
            val_metrics_no_ema = evaluate(model, val_loader, device, "val_no_ema")
        else:
            # Use previous validation metrics
            val_metrics = getattr(train, 'last_val_metrics', {'complete_acc': 0.0, 'loss': float('inf')})
            val_metrics_no_ema = {'complete_acc': 0.0}
        
        # Store for next epoch if needed
        if epoch % 2 == 0 or epoch <= 10:
            train.last_val_metrics = val_metrics
        
        # Log to wandb
        if USE_WANDB:
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_loss,
                'train_complete_acc': avg_metrics['complete_acc'],
                'train_verb_acc': avg_metrics['verb_acc'],
                'train_subject_acc': avg_metrics['subject_acc'],
                'train_object_acc': avg_metrics['object_acc'],
                'val_loss': val_metrics['loss'],
                'val_complete_acc': val_metrics['complete_acc'],
                'val_verb_acc': val_metrics['verb_acc'],
                'val_subject_acc': val_metrics['subject_acc'],
                'val_object_acc': val_metrics['object_acc'],
                'val_no_ema_complete_acc': val_metrics_no_ema['complete_acc'],
                'learning_rate': scheduler.get_last_lr()[0]
            })
        
        # Save best model based on EMA validation performance
        if val_metrics['complete_acc'] > best_acc:
            best_acc = val_metrics['complete_acc']
            # Save EMA weights as the best model
            ema.apply_shadow()
            torch.save(model.state_dict(), CKPT_PATH)
            ema.restore()
            print(f"✅ New best model saved! Complete accuracy: {best_acc:.4f}")
        
        # Early stopping if validation performance plateaus
        if epoch > 30 and val_metrics['complete_acc'] < best_acc * 0.95:
            patience_counter = getattr(train, 'patience_counter', 0) + 1
            train.patience_counter = patience_counter
            if patience_counter >= 10:
                print("Early stopping: validation performance has plateaued")
                break
        else:
            train.patience_counter = 0
        
        # Save checkpoint for resuming
        save_checkpoint(model, optimizer, scheduler, epoch, best_acc, LAST_CKPT_PATH)
        
        # Memory cleanup
        if epoch % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()


def main():
    print("🚀 Starting Simple Triplet Predictor Training")
    print(f"Dataset: {ROOT_DIR}")
    print(f"Triplet dimensions: Verbs={NUM_VERBS}, Subjects={NUM_SUBJECTS}, Objects={NUM_OBJECTS}")
    
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize wandb
    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            name=f"{WANDB_RUN_NAME}_stable_v2",
            config={
                'seq_len': SEQ_LEN,
                'stride': STRIDE,
                'batch_size': BATCH_SIZE,
                'learning_rate': LR,
                'weight_decay': WEIGHT_DECAY,
                'hidden_dim': HIDDEN_DIM,
                'dropout': DROPOUT,
                'num_temporal_layers': NUM_TEMPORAL_LAYERS,
                'epochs': EPOCHS,
                'warmup_epochs': WARMUP_EPOCHS,
                'ema_decay': EMA_DECAY,
                'architecture': 'SimpleTripletPredictor_Stable',
                'backbone': 'ResNet50_PartiallyFrozen',
                'improvements': ['EMA', 'LabelSmoothing', 'AttentionPooling', 'AdaptiveLossWeighting', 'EarlyStopping'],
                'loss_weights': {'verb': 1.0, 'subject': 0.8, 'object': 1.5}
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
        augment=True,
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
    print("\n🏗️  Building model...")
    model = SimpleTripletPredictor(
        seq_len=SEQ_LEN,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        num_temporal_layers=NUM_TEMPORAL_LAYERS
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    if DO_TRAIN:
        print("\n🎯 Starting training...")
        train(model, train_loader, val_loader, device)
    
    # Load best model for testing
    if CKPT_PATH.exists():
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        print(f"\n✅ Loaded best model from {CKPT_PATH}")
    
    # Final evaluation
    print("\n📊 Final evaluation...")
    test_metrics = evaluate(model, test_loader, device, "test")
    
    print("\n" + "="*60)
    print("FINAL TEST RESULTS:")
    print("="*60)
    print(f"Complete Triplet Accuracy: {test_metrics['complete_acc']:.4f}")
    print(f"Verb Accuracy:            {test_metrics['verb_acc']:.4f}")
    print(f"Subject Accuracy:         {test_metrics['subject_acc']:.4f}")
    print(f"Object Accuracy:          {test_metrics['object_acc']:.4f}")
    print(f"Average Component Acc:    {test_metrics['avg_component_acc']:.4f}")
    print("="*60)
    
    # Log final results to wandb
    if USE_WANDB:
        wandb.log({
            'test_complete_acc': test_metrics['complete_acc'],
            'test_verb_acc': test_metrics['verb_acc'],
            'test_subject_acc': test_metrics['subject_acc'],
            'test_object_acc': test_metrics['object_acc'],
            'test_avg_component_acc': test_metrics['avg_component_acc']
        })
        wandb.finish()
    
    print("\n🎉 Training completed!")


if __name__ == "__main__":
    main()