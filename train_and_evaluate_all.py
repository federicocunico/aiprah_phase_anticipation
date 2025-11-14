#!/usr/bin/env python3
"""
Train and Evaluate All Models on PegAndRing Dataset

Trains phase recognition models (MuST, SWAG, SKiT) and triplet recognition models
(TDN, Rendezvous, RiT) using simplified adapters on PegAndRing dataset.

Phase Models:
- Skips training if checkpoint exists
- Evaluates on Phase Acc., MAE, inMAE, oMAE, wMAE

Triplet Models:
- Evaluates on Triplet Acc., APv (verb mAP), APt (target mAP), APvt (triplet mAP)

Usage:
    python train_and_evaluate_all.py [--epochs 50] [--force-retrain] [--eval-only]
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import time
from collections import defaultdict

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Import dataset
from datasets.peg_and_ring_workflow import PegAndRing, VideoBatchSampler

from related.phase.must import MuST
from related.phase.swag import SWAG
from related.phase.skit import SKiT

from related.triplet.tdn import TDN
from related.triplet.rendezvous import Rendezvous
from related.triplet.rit import RiT

# Configuration
SEED = 42
ROOT_DIR = "data/peg_and_ring_workflow"
TIME_UNIT = "minutes"
NUM_CLASSES = 6
NUM_TRIPLET_VERBS = 4  # reach, grasp, release, null-verb
NUM_TRIPLET_TARGETS = 14  # 5 pegs + 4 rings + center + outside + table + null-target + hands + null-target
BATCH_SIZE = 4  # Reduced for MuST's memory requirements
NUM_WORKERS = 30
SEQ_LEN = 16
STRIDE = 1
TIME_HORIZON = 2.0

# Training config
DEFAULT_EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0

# Early stopping config
EARLY_STOP_PATIENCE = 5  # Number of epochs to wait for improvement
EARLY_STOP_MIN_DELTA = 0.001  # Minimum change (0.1%) to consider as improvement

# Checkpoints
CKPT_DIR = Path("checkpoints/model_comparison")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Results
RESULTS_DIR = Path("results/model_comparison")
RESULTS_DIR = Path("results/model_comparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def calculate_phase_metrics(phase_preds, phase_gts):
    """Calculate phase recognition accuracy."""
    correct = (phase_preds == phase_gts).sum().item()
    total = len(phase_gts)
    accuracy = correct / total if total > 0 else 0.0
    return {"phase_acc": accuracy}


def calculate_anticipation_metrics(time_preds, time_gts, time_horizon):
    """
    Calculate anticipation metrics: MAE, inMAE, oMAE, wMAE.
    
    Args:
        time_preds: (N, num_classes) predictions
        time_gts: (N, num_classes) ground truth
        time_horizon: maximum prediction horizon
    """
    abs_errors = np.abs(time_preds - time_gts)
    mae = np.mean(abs_errors)
    
    h = time_horizon
    in_indices = time_gts < h
    out_indices = time_gts == h
    
    in_errors = abs_errors[in_indices] if np.any(in_indices) else np.array([])
    out_errors = np.abs(time_preds[out_indices] - h) if np.any(out_indices) else np.array([])
    
    inMAE = np.mean(in_errors) if in_errors.size > 0 else 0.0
    oMAE = np.mean(out_errors) if out_errors.size > 0 else 0.0
    wMAE = (inMAE + oMAE) / 2.0
    
    return {
        "mae": float(mae),
        "inMAE": float(inMAE),
        "oMAE": float(oMAE),
        "wMAE": float(wMAE),
    }


def calculate_triplet_metrics(verb_preds, verb_gts, target_preds, target_gts):
    """
    Calculate triplet recognition metrics.
    
    Args:
        verb_preds: (N, num_verbs) - verb logits or probabilities
        verb_gts: (N,) - ground truth verb indices
        target_preds: (N, num_targets) - target logits or probabilities
        target_gts: (N,) - ground truth target indices
    
    Returns:
        dict with keys: triplet_acc, ap_verbs, ap_targets, ap_triplet
    """
    # Convert predictions to binary for each class (multi-label format for mAP)
    num_verbs = verb_preds.shape[1]
    num_targets = target_preds.shape[1]
    
    # Triplet accuracy: both verb and target correct
    verb_pred_idx = np.argmax(verb_preds, axis=1)
    target_pred_idx = np.argmax(target_preds, axis=1)
    triplet_correct = (verb_pred_idx == verb_gts) & (target_pred_idx == target_gts)
    triplet_acc = triplet_correct.mean()
    
    # Average Precision for verbs (one-vs-rest per class, then average)
    verb_gts_onehot = np.zeros((len(verb_gts), num_verbs))
    verb_gts_onehot[np.arange(len(verb_gts)), verb_gts] = 1
    ap_verbs_per_class = []
    for i in range(num_verbs):
        if verb_gts_onehot[:, i].sum() > 0:  # Only if class exists
            ap = average_precision_score(verb_gts_onehot[:, i], verb_preds[:, i])
            ap_verbs_per_class.append(ap)
    ap_verbs = np.mean(ap_verbs_per_class) if ap_verbs_per_class else 0.0
    
    # Average Precision for targets
    target_gts_onehot = np.zeros((len(target_gts), num_targets))
    target_gts_onehot[np.arange(len(target_gts)), target_gts] = 1
    ap_targets_per_class = []
    for i in range(num_targets):
        if target_gts_onehot[:, i].sum() > 0:
            ap = average_precision_score(target_gts_onehot[:, i], target_preds[:, i])
            ap_targets_per_class.append(ap)
    ap_targets = np.mean(ap_targets_per_class) if ap_targets_per_class else 0.0
    
    # Average Precision for verb+target combinations (treat as multi-class)
    # Create combined ground truth labels
    num_triplets = num_verbs * num_targets
    triplet_gts = verb_gts * num_targets + target_gts
    triplet_gts_onehot = np.zeros((len(triplet_gts), num_triplets))
    triplet_gts_onehot[np.arange(len(triplet_gts)), triplet_gts] = 1
    
    # Create combined predictions (outer product)
    triplet_preds = np.zeros((len(verb_preds), num_triplets))
    for i in range(len(verb_preds)):
        verb_probs = verb_preds[i]  # (num_verbs,)
        target_probs = target_preds[i]  # (num_targets,)
        # Outer product: (num_verbs, num_targets)
        combined = np.outer(verb_probs, target_probs).flatten()
        triplet_preds[i] = combined
    
    ap_triplet_per_class = []
    for i in range(num_triplets):
        if triplet_gts_onehot[:, i].sum() > 0:
            ap = average_precision_score(triplet_gts_onehot[:, i], triplet_preds[:, i])
            ap_triplet_per_class.append(ap)
    ap_triplet = np.mean(ap_triplet_per_class) if ap_triplet_per_class else 0.0
    
    return {
        "triplet_acc": float(triplet_acc),
        "ap_verbs": float(ap_verbs),
        "ap_targets": float(ap_targets),
        "ap_triplet": float(ap_triplet),
    }


def create_phase_model(model_name: str, num_classes: int = 6):
    """Create phase recognition model (MuST, SWAG, or SKiT)."""
    if model_name == "MuST":
        model = MuST(
            num_scales=2,  # Reduced from 4 to save memory
            num_frames=16,
            img_size=224,
            num_phases=num_classes,
            embed_dim=96,
            depth=4,
            num_heads=8,
            tcm_layers=4
        )
    elif model_name == "SWAG":
        model = SWAG(
            img_size=224,  # Match dataset crop size
            num_classes=num_classes,
            seq_length=1440,  # 24 minutes * 60 fps
            window_size=20,
            num_context_tokens=24,
            decoder_type='single_pass',
            use_prior_knowledge=False,  # Disable prior to avoid dtype issues
            max_horizon=60
        )
    elif model_name == "SKiT":
        model = SKiT(
            img_size=224,  # Match dataset crop size
            num_phases=num_classes,
            window_size=100,
            key_dim=64,
            local_dim=512,
            spatial_dim=768
        )
    else:
        raise ValueError(f"Unknown phase model: {model_name}")
    
    return model


def create_triplet_model(model_name: str, num_verbs: int = 4, num_targets: int = 14):
    """Create triplet recognition model (TDN, Rendezvous, or RiT)."""
    # Note: num_triplets is typically num_verbs * num_targets
    num_triplets = num_verbs * num_targets
    
    if model_name == "TDN":
        model = TDN(
            num_instruments=6,  # Not used for PegAndRing but required by model
            num_verbs=num_verbs,
            num_targets=num_targets,
            num_triplets=num_triplets,
            clip_size=5,
            pretrained=True
        )
    elif model_name == "Rendezvous":
        model = Rendezvous(
            num_instruments=6,
            num_verbs=num_verbs,
            num_targets=num_targets,
            num_triplets=num_triplets,
            num_decoder_layers=8,
            pretrained=True
        )
    elif model_name == "RiT":
        model = RiT(
            num_instruments=6,
            num_verbs=num_verbs,
            num_targets=num_targets,
            num_triplets=num_triplets,
            clip_size=6,
            num_tam_layers=2,
            pretrained=True
        )
    else:
        raise ValueError(f"Unknown triplet model: {model_name}")
    
    return model


@torch.no_grad()
def evaluate_model(model, loader, device, model_name):
    """Evaluate phase model on validation set."""
    model.eval()
    
    all_phase_preds = []
    all_phase_gts = []
    all_time_preds = []
    all_time_gts = []
    
    for frames, meta in loader:
        frames = frames.to(device)
        phase_gt = meta["phase_label"].to(device).long()
        time_to_next = meta["time_to_next_phase"].to(device).float()
        time_to_next = torch.clamp(time_to_next, min=0.0, max=TIME_HORIZON)
        
        # Forward pass - handle different model output formats
        if model_name == "MuST":
            # MuST expects pyramid input: List of (B, T, C, H, W), one per scale
            # For simplicity, create 2 scales by taking different temporal strides
            B, T, C, H, W = frames.shape
            # Create multi-scale pyramid: different temporal subsampling
            pyramid = [
                frames,  # Scale 1: full temporal resolution
                frames[:, ::2, :, :, :],  # Scale 2: stride 2
            ]
            # MuST forward expects (B, F', num_scales, T, C, H, W)
            # We have 1 keyframe (F'=1), so reshape
            # Stack into tensor: (num_scales, B, T', C, H, W) where T' varies
            # Actually, let's use list format and wrap in another dimension
            # Create (B, F'=1, num_scales, T, C, H, W)
            max_T = max([p.shape[1] for p in pyramid])
            # Pad all to same temporal length
            pyramid_padded = []
            for p in pyramid:
                if p.shape[1] < max_T:
                    # Pad with last frame
                    pad_size = max_T - p.shape[1]
                    last_frame = p[:, -1:, :, :, :].repeat(1, pad_size, 1, 1, 1)
                    p = torch.cat([p, last_frame], dim=1)
                pyramid_padded.append(p)
            
            # Stack: (num_scales, B, T, C, H, W) -> (B, num_scales, T, C, H, W)
            video_pyramid_tensor = torch.stack(pyramid_padded, dim=0).permute(1, 0, 2, 3, 4, 5)
            # Add keyframe dimension: (B, F'=1, num_scales, T, C, H, W)
            video_pyramid_tensor = video_pyramid_tensor.unsqueeze(1)
            
            outputs = model(video_pyramid_tensor, use_tcm=True)
            phase_logits = outputs['tcm_logits'] if 'tcm_logits' in outputs else outputs['mtfe_logits']
            # phase_logits is (B, F', num_phases), squeeze F' dimension
            if phase_logits.dim() == 3:
                phase_logits = phase_logits.squeeze(1)
            time_pred = torch.zeros(B, NUM_CLASSES, device=device)
            
        elif model_name == "SWAG":
            # SWAG returns tuple: (recognition_logits, anticipation_logits)
            recognition_logits, anticipation_logits = model(frames)
            phase_logits = recognition_logits
            # SWAG returns (B, N, num_classes), use first time step
            time_pred = anticipation_logits[:, 0, :] if anticipation_logits.dim() == 3 else anticipation_logits
            
        elif model_name == "SKiT":
            # SKiT returns tuple: (phase_preds, heatmap_preds)
            phase_preds, heatmap_preds = model(frames, return_features=False)
            # phase_preds is (B, T, num_phases), average over time
            phase_logits = phase_preds.mean(dim=1)
            # SKiT doesn't have time prediction, use dummy
            time_pred = torch.zeros(phase_logits.shape[0], NUM_CLASSES, device=device)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Collect predictions
        phase_pred = torch.argmax(phase_logits, dim=1)
        all_phase_preds.append(phase_pred.cpu().numpy())
        all_phase_gts.append(phase_gt.cpu().numpy())
        all_time_preds.append(time_pred.cpu().numpy())
        all_time_gts.append(time_to_next.cpu().numpy())
    
    # Concatenate all
    all_phase_preds = np.concatenate(all_phase_preds)
    all_phase_gts = np.concatenate(all_phase_gts)
    all_time_preds = np.concatenate(all_time_preds)
    all_time_gts = np.concatenate(all_time_gts)
    
    # Calculate metrics
    metrics = calculate_phase_metrics(all_phase_preds, all_phase_gts)
    anticipation_metrics = calculate_anticipation_metrics(
        all_time_preds, all_time_gts, TIME_HORIZON
    )
    metrics.update(anticipation_metrics)
    
    return metrics


@torch.no_grad()
def evaluate_triplet_model(model, loader, device, model_name):
    """Evaluate triplet model on validation set."""
    model.eval()
    
    all_verb_preds = []
    all_verb_gts = []
    all_target_preds = []
    all_target_gts = []
    
    for frames, meta in loader:
        frames = frames.to(device)
        # Extract verb and target from triplet_classification (verb_idx, subject_idx, target_idx)
        triplet_classification = meta["triplet_classification"].to(device).long()
        verb_gt = triplet_classification[:, 0]  # verb index
        target_gt = triplet_classification[:, 2]  # destination/target index
        
        # Forward pass - all triplet models return dict with verb_logits, target_logits
        if model_name == "TDN":
            # TDN expects (B, C, T, H, W) - need to transpose
            B, T, C, H, W = frames.shape
            frames_tdn = frames.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            outputs = model(frames_tdn)
        else:  # Rendezvous or RiT
            # Rendezvous: single frame (B, 3, H, W) - use middle frame
            # RiT: clip (B, m, 3, H, W) format
            if model_name == "Rendezvous":
                # Use middle frame
                mid_idx = frames.shape[1] // 2
                frames_single = frames[:, mid_idx]  # (B, C, H, W)
                outputs = model(frames_single)
            else:  # RiT
                # RiT expects (B, m, C, H, W)
                outputs = model(frames)
        
        verb_logits = outputs['verb_logits']
        target_logits = outputs['target_logits']
        
        # Apply softmax to get probabilities for mAP
        verb_probs = torch.softmax(verb_logits, dim=1)
        target_probs = torch.softmax(target_logits, dim=1)
        
        # Collect predictions
        all_verb_preds.append(verb_probs.cpu().numpy())
        all_verb_gts.append(verb_gt.cpu().numpy())
        all_target_preds.append(target_probs.cpu().numpy())
        all_target_gts.append(target_gt.cpu().numpy())
    
    # Concatenate all
    all_verb_preds = np.concatenate(all_verb_preds)
    all_verb_gts = np.concatenate(all_verb_gts)
    all_target_preds = np.concatenate(all_target_preds)
    all_target_gts = np.concatenate(all_target_gts)
    
    # Calculate triplet metrics
    metrics = calculate_triplet_metrics(
        all_verb_preds, all_verb_gts, all_target_preds, all_target_gts
    )
    
    return metrics


def train_epoch(model, loader, optimizer, criterion_ce, criterion_reg, device, model_name, max_batches=None):
    """Train phase model for one epoch (or limited batches for testing)."""
    model.train()
    
    total_loss = 0.0
    total_phase_loss = 0.0
    total_time_loss = 0.0
    num_batches = 0
    
    # log steps, 10 logs per epoch
    N = len(loader) if max_batches is None else max_batches
    log_steps = torch.linspace(0, N, steps=min(10, N)).int()

    for i, (frames, meta) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        frames = frames.to(device)
        phase_gt = meta["phase_label"].to(device).long()
        time_to_next = meta["time_to_next_phase"].to(device).float()
        time_to_next = torch.clamp(time_to_next, min=0.0, max=TIME_HORIZON)
        
        # Forward pass - handle different model output formats
        if model_name == "MuST":
            # MuST expects pyramid input: List of (B, T, C, H, W), one per scale
            B, T, C, H, W = frames.shape
            pyramid = [
                frames,  # Scale 1: full temporal resolution
                frames[:, ::2, :, :, :],  # Scale 2: stride 2
            ]
            # Pad all to same temporal length
            max_T = max([p.shape[1] for p in pyramid])
            pyramid_padded = []
            for p in pyramid:
                if p.shape[1] < max_T:
                    pad_size = max_T - p.shape[1]
                    last_frame = p[:, -1:, :, :, :].repeat(1, pad_size, 1, 1, 1)
                    p = torch.cat([p, last_frame], dim=1)
                pyramid_padded.append(p)
            
            # Stack: (B, num_scales, T, C, H, W) and add keyframe dim
            video_pyramid_tensor = torch.stack(pyramid_padded, dim=1).unsqueeze(1)
            outputs = model(video_pyramid_tensor, use_tcm=True)
            phase_logits = outputs['tcm_logits'] if 'tcm_logits' in outputs else outputs['mtfe_logits']
            if phase_logits.dim() == 3:
                phase_logits = phase_logits.squeeze(1)
            time_pred = torch.zeros(B, NUM_CLASSES, device=device)
            
        elif model_name == "SWAG":
            recognition_logits, anticipation_logits = model(frames)
            phase_logits = recognition_logits
            # SWAG returns (B, N, num_classes), use first time step
            time_pred = anticipation_logits[:, 0, :] if anticipation_logits.dim() == 3 else anticipation_logits
            
        elif model_name == "SKiT":
            phase_preds, heatmap_preds = model(frames, return_features=False)
            phase_logits = phase_preds.mean(dim=1)
            time_pred = torch.zeros(phase_logits.shape[0], NUM_CLASSES, device=device)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Compute losses
        phase_loss = criterion_ce(phase_logits, phase_gt)
        time_loss = criterion_reg(time_pred, time_to_next)
        loss = phase_loss + 0.5 * time_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        
        # Accumulate
        total_loss += loss.item()
        total_phase_loss += phase_loss.item()
        total_time_loss += time_loss.item()
        num_batches += 1

        # Print every N/10 batches
        if i in log_steps:
            current_n = len(loader) if max_batches is None else max_batches
            print(f"  Batch {i+1}/{current_n} - "
                  f"Loss: {total_loss / num_batches:.4f} "
                  f"(Phase: {total_phase_loss / num_batches:.4f}, "
                  f"Time: {total_time_loss / num_batches:.4f})")

    return {
        "loss": total_loss / num_batches,
        "phase_loss": total_phase_loss / num_batches,
        "time_loss": total_time_loss / num_batches,
    }


def train_triplet_epoch(model, loader, optimizer, criterion_ce, device, model_name, max_batches=None):
    """Train triplet model for one epoch (or limited batches for testing)."""
    model.train()
    
    total_loss = 0.0
    total_verb_loss = 0.0
    total_target_loss = 0.0
    num_batches = 0
    
    # log steps, 10 logs per epoch
    N = len(loader) if max_batches is None else max_batches
    log_steps = torch.linspace(0, N, steps=min(10, N)).int()

    for i, (frames, meta) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        frames = frames.to(device)
        # Extract verb and target from triplet_classification (verb_idx, subject_idx, target_idx)
        triplet_classification = meta["triplet_classification"].to(device).long()
        verb_gt = triplet_classification[:, 0]  # verb index
        target_gt = triplet_classification[:, 2]  # destination/target index
        
        # Forward pass - handle different input formats
        if model_name == "TDN":
            B, T, C, H, W = frames.shape
            frames_tdn = frames.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            outputs = model(frames_tdn)
        else:  # Rendezvous or RiT
            if model_name == "Rendezvous":
                mid_idx = frames.shape[1] // 2
                frames_single = frames[:, mid_idx]
                outputs = model(frames_single)
            else:  # RiT
                outputs = model(frames)
        
        verb_logits = outputs['verb_logits']
        target_logits = outputs['target_logits']
        
        # Compute losses
        verb_loss = criterion_ce(verb_logits, verb_gt)
        target_loss = criterion_ce(target_logits, target_gt)
        loss = verb_loss + target_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        
        # Accumulate
        total_loss += loss.item()
        total_verb_loss += verb_loss.item()
        total_target_loss += target_loss.item()
        num_batches += 1

        # Print every N/10 batches
        if i in log_steps:
            current_n = len(loader) if max_batches is None else max_batches
            print(f"  Batch {i+1}/{current_n} - "
                  f"Loss: {total_loss / num_batches:.4f} "
                  f"(Verb: {total_verb_loss / num_batches:.4f}, "
                  f"Target: {total_target_loss / num_batches:.4f})")

    return {
        "loss": total_loss / num_batches,
        "verb_loss": total_verb_loss / num_batches,
        "target_loss": total_target_loss / num_batches,
    }


def train_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    checkpoint_path: Path,
    test_mode: bool = False,
):
    """Train a model from scratch (or test with limited batches)."""
    print(f"\n{'='*80}")
    if test_mode:
        print(f"Testing {model_name} (limited batches)")
    else:
        print(f"Training {model_name}")
    print(f"{'='*80}\n")
    
    # Create model
    model = create_phase_model(model_name, num_classes=NUM_CLASSES).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Loss functions
    criterion_ce = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training
    best_val_acc = 0.0
    best_metrics = {}
    best_epoch = 0
    
    # Early stopping variables
    patience_counter = 0
    best_avg_loss = float('inf')
    epoch_losses = []  # Track average loss per epoch
    
    # In test mode, use only a few batches
    max_train_batches = 5 if test_mode else None
    max_val_batches = 3 if test_mode else None
    
    for epoch in range(1, epochs + 1):
        # Train
        train_stats = train_epoch(
            model, train_loader, optimizer, criterion_ce, criterion_reg, device, model_name, max_train_batches
        )
        
        # Track epoch average loss
        current_avg_loss = train_stats['loss']
        epoch_losses.append(current_avg_loss)
        
        # Validate every 5 epochs (or always in test mode)
        if test_mode or epoch % 5 == 0 or epoch == epochs:
            # Quick validation in test mode
            if test_mode:
                print(f"Epoch {epoch}/{epochs} - Quick validation (limited batches)")
                # Just do a quick check, don't compute full metrics
                model.eval()
                with torch.no_grad():
                    for i, (frames, meta) in enumerate(val_loader):
                        if i >= max_val_batches:
                            break
                        frames = frames.to(device)
                        # Just run forward pass to verify no errors
                        if model_name == "MuST":
                            B, T, C, H, W = frames.shape
                            pyramid = [frames, frames[:, ::2, :, :, :]]
                            max_T = max([p.shape[1] for p in pyramid])
                            pyramid_padded = []
                            for p in pyramid:
                                if p.shape[1] < max_T:
                                    pad_size = max_T - p.shape[1]
                                    last_frame = p[:, -1:, :, :, :].repeat(1, pad_size, 1, 1, 1)
                                    p = torch.cat([p, last_frame], dim=1)
                                pyramid_padded.append(p)
                            video_pyramid_tensor = torch.stack(pyramid_padded, dim=0).permute(1, 0, 2, 3, 4, 5).unsqueeze(1)
                            outputs = model(video_pyramid_tensor, use_tcm=True)
                        elif model_name == "SWAG":
                            recognition_logits, anticipation_logits = model(frames)
                        elif model_name == "SKiT":
                            phase_preds, heatmap_preds = model(frames, return_features=False)
                
                val_metrics = {"phase_acc": 0.0, "mae": 0.0, "inMAE": 0.0, "oMAE": 0.0, "wMAE": 0.0}
            else:
                val_metrics = evaluate_model(model, val_loader, device, model_name)
            
            print(f"Epoch {epoch}/{epochs}")
            print(f"  Train Loss: {train_stats['loss']:.4f} "
                  f"(Phase: {train_stats['phase_loss']:.4f}, Time: {train_stats['time_loss']:.4f})")
            if not test_mode:
                print(f"  Val Phase Acc: {val_metrics['phase_acc']:.4f}, "
                      f"MAE: {val_metrics['mae']:.4f}, wMAE: {val_metrics['wMAE']:.4f}")
            
            # Save best model (only in real training mode)
            if not test_mode and val_metrics['phase_acc'] > best_val_acc:
                best_val_acc = val_metrics['phase_acc']
                best_metrics = val_metrics
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'metrics': val_metrics,
                    'epoch_losses': epoch_losses,
                }, checkpoint_path)
                print(f"  💾 Saved best model (acc: {best_val_acc:.4f}, epoch: {epoch})")
        
        # Early stopping check (only if not in test mode)
        if not test_mode and epoch >= EARLY_STOP_PATIENCE:
            # Calculate improvement
            if current_avg_loss < best_avg_loss:
                improvement = (best_avg_loss - current_avg_loss) / best_avg_loss
                if improvement > EARLY_STOP_MIN_DELTA:
                    # Significant improvement
                    best_avg_loss = current_avg_loss
                    patience_counter = 0
                else:
                    # Improvement too small
                    patience_counter += 1
                    print(f"  ⚠️  Small improvement ({improvement*100:.3f}%), patience: {patience_counter}/{EARLY_STOP_PATIENCE}")
            else:
                # No improvement
                patience_counter += 1
                print(f"  ⚠️  No improvement, patience: {patience_counter}/{EARLY_STOP_PATIENCE}")
            
            # Stop if patience exhausted
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n🛑 Early stopping triggered at epoch {epoch}")
                print(f"   No significant improvement for {EARLY_STOP_PATIENCE} epochs")
                print(f"   Best avg loss: {best_avg_loss:.4f}")
                # Save final state with early stop info
                torch.save({
                    'epoch': epoch,
                    'stopped_epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'metrics': best_metrics,
                    'epoch_losses': epoch_losses,
                    'early_stopped': True,
                }, checkpoint_path)
                break
        
        scheduler.step()
    
    if test_mode:
        print(f"\n✅ {model_name} Test Complete! (Model verified successfully)")
        # Return dummy metrics for test mode
        return {"phase_acc": 0.0, "mae": 0.0, "inMAE": 0.0, "oMAE": 0.0, "wMAE": 0.0, "trained_epochs": epoch}
    else:
        print(f"\n✅ {model_name} Training Complete!")
        print(f"   Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})")
        print(f"   Total Epochs: {epoch}/{epochs}")
        print(f"   Avg Loss History: {epoch_losses[-5:] if len(epoch_losses) > 5 else epoch_losses}")
        best_metrics['trained_epochs'] = epoch
        best_metrics['best_epoch'] = best_epoch
        return best_metrics


def train_triplet_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    checkpoint_path: Path,
    num_verbs: int = 4,
    num_targets: int = 14,
    test_mode: bool = False,
):
    """Train a triplet model from scratch (or test with limited batches)."""
    print(f"\n{'='*80}")
    if test_mode:
        print(f"Testing {model_name} (Triplet Task - limited batches)")
    else:
        print(f"Training {model_name} (Triplet Task)")
    print(f"{'='*80}\n")
    
    # Create model
    model = create_triplet_model(
        model_name, num_verbs=num_verbs, num_targets=num_targets
    ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Loss functions
    criterion_ce = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training
    best_val_triplet_acc = 0.0
    best_metrics = {}
    best_epoch = 0
    
    # Early stopping variables
    patience_counter = 0
    best_avg_loss = float('inf')
    epoch_losses = []  # Track average loss per epoch
    
    # In test mode, use only a few batches
    max_train_batches = 5 if test_mode else None
    max_val_batches = 3 if test_mode else None
    
    for epoch in range(1, epochs + 1):
        # Train
        train_stats = train_triplet_epoch(
            model, train_loader, optimizer, criterion_ce, device, model_name, max_train_batches
        )
        
        # Track epoch average loss
        current_avg_loss = train_stats['loss']
        epoch_losses.append(current_avg_loss)
        
        # Validate every 5 epochs (or always in test mode)
        if test_mode or epoch % 5 == 0 or epoch == epochs:
            # Quick validation in test mode
            if test_mode:
                print(f"Epoch {epoch}/{epochs} - Quick validation (limited batches)")
                # Just do a quick check, don't compute full metrics
                model.eval()
                with torch.no_grad():
                    for i, (frames, meta) in enumerate(val_loader):
                        if i >= max_val_batches:
                            break
                        frames = frames.to(device)
                        # Just run forward pass to verify no errors
                        if model_name == "TDN":
                            B, T, C, H, W = frames.shape
                            frames_tdn = frames.permute(0, 2, 1, 3, 4)
                            outputs = model(frames_tdn)
                        elif model_name == "Rendezvous":
                            mid_idx = frames.shape[1] // 2
                            frames_single = frames[:, mid_idx]
                            outputs = model(frames_single)
                        elif model_name == "RiT":
                            outputs = model(frames)
                
                val_metrics = {"triplet_acc": 0.0, "ap_verbs": 0.0, "ap_targets": 0.0, "ap_triplet": 0.0}
            else:
                val_metrics = evaluate_triplet_model(model, val_loader, device, model_name)
            
            print(f"Epoch {epoch}/{epochs}")
            print(f"  Train Loss: {train_stats['loss']:.4f} "
                  f"(Verb: {train_stats['verb_loss']:.4f}, Target: {train_stats['target_loss']:.4f})")
            if not test_mode:
                print(f"  Val Triplet Acc: {val_metrics['triplet_acc']:.4f}, "
                      f"APv: {val_metrics['ap_verbs']:.4f}, APt: {val_metrics['ap_targets']:.4f}, "
                      f"APvt: {val_metrics['ap_triplet']:.4f}")
            
            # Save best model (only in real training mode)
            if not test_mode and val_metrics['triplet_acc'] > best_val_triplet_acc:
                best_val_triplet_acc = val_metrics['triplet_acc']
                best_metrics = val_metrics
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_triplet_acc': best_val_triplet_acc,
                    'metrics': val_metrics,
                    'epoch_losses': epoch_losses,
                }, checkpoint_path)
                print(f"  💾 Saved best model (triplet_acc: {best_val_triplet_acc:.4f}, epoch: {epoch})")
        
        # Early stopping check (only if not in test mode)
        if not test_mode and epoch >= EARLY_STOP_PATIENCE:
            # Calculate improvement
            if current_avg_loss < best_avg_loss:
                improvement = (best_avg_loss - current_avg_loss) / best_avg_loss
                if improvement > EARLY_STOP_MIN_DELTA:
                    # Significant improvement
                    best_avg_loss = current_avg_loss
                    patience_counter = 0
                else:
                    # Improvement too small
                    patience_counter += 1
                    print(f"  ⚠️  Small improvement ({improvement*100:.3f}%), patience: {patience_counter}/{EARLY_STOP_PATIENCE}")
            else:
                # No improvement
                patience_counter += 1
                print(f"  ⚠️  No improvement, patience: {patience_counter}/{EARLY_STOP_PATIENCE}")
            
            # Stop if patience exhausted
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n🛑 Early stopping triggered at epoch {epoch}")
                print(f"   No significant improvement for {EARLY_STOP_PATIENCE} epochs")
                print(f"   Best avg loss: {best_avg_loss:.4f}")
                # Save final state with early stop info
                torch.save({
                    'epoch': epoch,
                    'stopped_epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_triplet_acc': best_val_triplet_acc,
                    'metrics': best_metrics,
                    'epoch_losses': epoch_losses,
                    'early_stopped': True,
                }, checkpoint_path)
                break
        
        scheduler.step()
    
    if test_mode:
        print(f"\n✅ {model_name} Test Complete! (Model verified successfully)")
        # Return dummy metrics for test mode
        return {"triplet_acc": 0.0, "ap_verbs": 0.0, "ap_targets": 0.0, "ap_triplet": 0.0, "trained_epochs": epoch}
    else:
        print(f"\n✅ {model_name} Training Complete!")
        print(f"   Best Val Triplet Acc: {best_val_triplet_acc:.4f} (Epoch {best_epoch})")
        print(f"   Total Epochs: {epoch}/{epochs}")
        print(f"   Avg Loss History: {epoch_losses[-5:] if len(epoch_losses) > 5 else epoch_losses}")
        best_metrics['trained_epochs'] = epoch
        best_metrics['best_epoch'] = best_epoch
        return best_metrics


def load_and_evaluate_triplet(
    model_name: str,
    checkpoint_path: Path,
    val_loader: DataLoader,
    device: torch.device,
    num_verbs: int = 4,
    num_targets: int = 14,
):
    """Load triplet checkpoint and evaluate."""
    print(f"\n📊 Evaluating {model_name} from checkpoint...")
    
    model = create_triplet_model(
        model_name, num_verbs=num_verbs, num_targets=num_targets
    ).to(device)
    
    # Load checkpoint (handle both old and new format)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_info = checkpoint.get('epoch', 'unknown')
        early_stopped = checkpoint.get('early_stopped', False)
        if early_stopped:
            print(f"   Model stopped early at epoch {epoch_info}")
        else:
            print(f"   Model trained for epoch {epoch_info}")
    else:
        # Old format: direct state dict
        model.load_state_dict(checkpoint)
    
    metrics = evaluate_triplet_model(model, val_loader, device, model_name)
    
    print(f"✅ {model_name} Evaluation:")
    print(f"   Triplet Acc: {metrics['triplet_acc']:.4f}")
    print(f"   APv: {metrics['ap_verbs']:.4f}, APt: {metrics['ap_targets']:.4f}, "
          f"APvt: {metrics['ap_triplet']:.4f}")
    
    return metrics


def load_and_evaluate(
    model_name: str,
    checkpoint_path: Path,
    val_loader: DataLoader,
    device: torch.device,
):
    """Load checkpoint and evaluate."""
    print(f"\n📊 Evaluating {model_name} from checkpoint...")
    
    model = create_phase_model(model_name, num_classes=NUM_CLASSES).to(device)
    
    # Load checkpoint (handle both old and new format)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_info = checkpoint.get('epoch', 'unknown')
        early_stopped = checkpoint.get('early_stopped', False)
        if early_stopped:
            print(f"   Model stopped early at epoch {epoch_info}")
        else:
            print(f"   Model trained for epoch {epoch_info}")
    else:
        # Old format: direct state dict
        model.load_state_dict(checkpoint)
    
    metrics = evaluate_model(model, val_loader, device, model_name)
    
    print(f"✅ {model_name} Evaluation:")
    print(f"   Phase Acc: {metrics['phase_acc']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}, inMAE: {metrics['inMAE']:.4f}, "
          f"oMAE: {metrics['oMAE']:.4f}, wMAE: {metrics['wMAE']:.4f}")
    
    return metrics


def print_final_table(results: Dict[str, Dict[str, float]]):
    """Print final comparison table for both phase and triplet models."""
    print("\n" + "="*110)
    print("FINAL MODEL COMPARISON RESULTS")
    print("="*110)
    
    # Separate phase and triplet models
    phase_models = {k: v for k, v in results.items() if 'phase_acc' in v}
    triplet_models = {k: v for k, v in results.items() if 'triplet_acc' in v}
    
    # Print phase models table
    if phase_models:
        print("\n🔹 PHASE RECOGNITION MODELS (MuST, SWAG, SKiT)")
        print("-"*110)
        header = f"{'Model':<15}"
        for metric in ["Phase Acc.", "MAE", "inMAE", "oMAE", "wMAE", "Epochs"]:
            header += f"{metric:>13}"
        print(header)
        print("-"*110)
        
        for model_name, metrics in phase_models.items():
            row = f"{model_name:<15}"
            for metric_key in ["phase_acc", "mae", "inMAE", "oMAE", "wMAE"]:
                if metric_key in metrics:
                    value = metrics[metric_key]
                    if metric_key == "phase_acc":
                        row += f"{value*100:>12.2f}%"
                    else:
                        row += f"{value:>13.4f}"
                else:
                    row += f"{'N/A':>13}"
            # Add epoch info
            trained_epochs = metrics.get('trained_epochs', 'N/A')
            best_epoch = metrics.get('best_epoch', 'N/A')
            row += f"{str(trained_epochs)+'/'+str(best_epoch):>13}"
            print(row)
        
        print("\nPhase Metrics Explanation:")
        print("  Phase Acc. : Current phase classification accuracy")
        print("  MAE        : Mean Absolute Error for time-to-next-phase prediction")
        print("  inMAE      : MAE for in-horizon predictions (time < horizon)")
        print("  oMAE       : MAE for out-of-horizon predictions (time >= horizon)")
        print("  wMAE       : Weighted MAE (average of inMAE and oMAE)")
        print("  Epochs     : Total trained / Best epoch")
    
    # Print triplet models table
    if triplet_models:
        print("\n🔹 TRIPLET RECOGNITION MODELS (TDN, Rendezvous, RiT)")
        print("-"*125)
        header = f"{'Model':<15}"
        for metric in ["Triplet Acc.", "APv", "APt", "APvt", "Epochs"]:
            header += f"{metric:>15}"
        print(header)
        print("-"*125)
        
        for model_name, metrics in triplet_models.items():
            row = f"{model_name:<15}"
            for metric_key in ["triplet_acc", "ap_verbs", "ap_targets", "ap_triplet"]:
                if metric_key in metrics:
                    value = metrics[metric_key]
                    if metric_key == "triplet_acc":
                        row += f"{value*100:>14.2f}%"
                    else:
                        row += f"{value:>15.4f}"
                else:
                    row += f"{'N/A':>15}"
            # Add epoch info
            trained_epochs = metrics.get('trained_epochs', 'N/A')
            best_epoch = metrics.get('best_epoch', 'N/A')
            row += f"{str(trained_epochs)+'/'+str(best_epoch):>15}"
            print(row)
        
        print("\nTriplet Metrics Explanation:")
        print("  Triplet Acc. : Joint verb+target classification accuracy")
        print("  APv          : Average Precision for verb recognition (mAP)")
        print("  APt          : Average Precision for target recognition (mAP)")
        print("  APvt         : Average Precision for verb+target combinations (mAP)")
        print("  Epochs       : Total trained / Best epoch")
    
    print("="*110)


def save_results(results: Dict[str, Dict[str, float]], filepath: Path):
    """Save results to file."""
    with open(filepath, "w") as f:
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("="*110 + "\n\n")
        
        # Separate phase and triplet models
        phase_models = {k: v for k, v in results.items() if 'phase_acc' in v}
        triplet_models = {k: v for k, v in results.items() if 'triplet_acc' in v}
        
        # Phase models
        if phase_models:
            f.write("PHASE RECOGNITION MODELS\n")
            f.write("-"*110 + "\n")
            f.write(f"{'Model':<15} {'Phase Acc.':<12} {'MAE':<10} {'inMAE':<10} {'oMAE':<10} {'wMAE':<10} {'Epochs':<10}\n")
            f.write("-"*110 + "\n")
            for model_name, metrics in phase_models.items():
                f.write(f"{model_name:<15} ")
                f.write(f"{metrics.get('phase_acc', 0)*100:<11.2f}% ")
                f.write(f"{metrics.get('mae', 0):<10.4f} ")
                f.write(f"{metrics.get('inMAE', 0):<10.4f} ")
                f.write(f"{metrics.get('oMAE', 0):<10.4f} ")
                f.write(f"{metrics.get('wMAE', 0):<10.4f} ")
                trained_epochs = metrics.get('trained_epochs', 'N/A')
                best_epoch = metrics.get('best_epoch', 'N/A')
                f.write(f"{trained_epochs}/{best_epoch}\n")
            f.write("\n")
        
        # Triplet models
        if triplet_models:
            f.write("TRIPLET RECOGNITION MODELS\n")
            f.write("-"*110 + "\n")
            f.write(f"{'Model':<15} {'Triplet Acc.':<15} {'APv':<12} {'APt':<12} {'APvt':<12} {'Epochs':<10}\n")
            f.write("-"*110 + "\n")
            for model_name, metrics in triplet_models.items():
                f.write(f"{model_name:<15} ")
                f.write(f"{metrics.get('triplet_acc', 0)*100:<14.2f}% ")
                f.write(f"{metrics.get('ap_verbs', 0):<12.4f} ")
                f.write(f"{metrics.get('ap_targets', 0):<12.4f} ")
                f.write(f"{metrics.get('ap_triplet', 0):<12.4f} ")
                trained_epochs = metrics.get('trained_epochs', 'N/A')
                best_epoch = metrics.get('best_epoch', 'N/A')
                f.write(f"{trained_epochs}/{best_epoch}\n")
            f.write("\n")
        
        f.write("="*110 + "\n")
        f.write("\nDetailed Metrics:\n")
        for model_name, metrics in results.items():
            f.write(f"\n{model_name}:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value}\n")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate all models")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                       help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--force-retrain", action="store_true",
                       help="Force retraining even if checkpoint exists")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only evaluate existing checkpoints")
    parser.add_argument("--test-mode", action="store_true",
                       help="Test mode: only run a few batches to verify models work")
    args = parser.parse_args()
    
    print("🚀 Training and Evaluating Models on PegAndRing Dataset")
    print(f"Phase Models: MuST, SWAG, SKiT (Simplified Adapters)")
    print(f"Triplet Models: TDN, Rendezvous, RiT (Simplified Adapters)")
    print(f"Epochs: {args.epochs}, Batch Size: {BATCH_SIZE}, Seq Len: {SEQ_LEN}")
    print(f"Force Retrain: {args.force_retrain}, Eval Only: {args.eval_only}")
    if args.test_mode:
        print(f"🧪 TEST MODE: Only running 5 train batches + 3 val batches per model to verify functionality")
    print()
    
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Create datasets
    print("📁 Loading datasets...")
    train_ds = PegAndRing(
        ROOT_DIR, mode="train", seq_len=SEQ_LEN, stride=STRIDE,
        time_unit=TIME_UNIT, augment=False, force_triplets=False
    )
    val_ds = PegAndRing(
        ROOT_DIR, mode="val", seq_len=SEQ_LEN, stride=STRIDE,
        time_unit=TIME_UNIT, augment=False, force_triplets=False
    )
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples\n")
    
    # Create data loaders
    gen_train = torch.Generator()
    gen_train.manual_seed(SEED)
    train_batch_sampler = VideoBatchSampler(
        train_ds, batch_size=BATCH_SIZE, drop_last=False,
        shuffle_videos=True, generator=gen_train, batch_videos=False
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=train_batch_sampler,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    gen_val = torch.Generator()
    gen_val.manual_seed(SEED)
    val_batch_sampler = VideoBatchSampler(
        val_ds, batch_size=BATCH_SIZE, drop_last=False,
        shuffle_videos=False, generator=gen_val, batch_videos=False
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=val_batch_sampler,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # Models to test
    models = ["MuST", "SWAG", "SKiT"]
    results = {}
    
    # Train/evaluate each model
    for model_name in models:
        checkpoint_path = CKPT_DIR / f"{model_name.lower()}_best.pth"
        
        if args.eval_only:
            # Only evaluate
            if checkpoint_path.exists():
                metrics = load_and_evaluate(model_name, checkpoint_path, val_loader, device)
                results[model_name] = metrics
            else:
                print(f"❌ Checkpoint not found for {model_name}: {checkpoint_path}")
                results[model_name] = {
                    "phase_acc": 0.0, "mae": float('inf'),
                    "inMAE": float('inf'), "oMAE": float('inf'), "wMAE": float('inf')
                }
        else:
            # Train if needed, then evaluate
            if checkpoint_path.exists() and not args.force_retrain:
                print(f"✓ Found checkpoint for {model_name}: {checkpoint_path}")
                metrics = load_and_evaluate(model_name, checkpoint_path, val_loader, device)
            else:
                if args.force_retrain:
                    print(f"🔄 Force retraining {model_name}...")
                else:
                    print(f"🆕 No checkpoint found for {model_name}, training from scratch...")
                
                metrics = train_model(
                    model_name, train_loader, val_loader, device,
                    args.epochs, checkpoint_path, test_mode=args.test_mode
                )
                
                # Final evaluation with best checkpoint (skip in test mode)
                if not args.test_mode:
                    print(f"\n📊 Final evaluation of {model_name}...")
                    metrics = load_and_evaluate(model_name, checkpoint_path, val_loader, device)
            
            results[model_name] = metrics
    
    # Create triplet datasets for triplet models
    print("\n📁 Loading triplet datasets (force_triplets=True)...")
    train_ds_triplet = PegAndRing(
        ROOT_DIR, mode="train", seq_len=SEQ_LEN, stride=STRIDE,
        time_unit=TIME_UNIT, augment=False, force_triplets=True
    )
    val_ds_triplet = PegAndRing(
        ROOT_DIR, mode="val", seq_len=SEQ_LEN, stride=STRIDE,
        time_unit=TIME_UNIT, augment=False, force_triplets=True
    )
    print(f"Triplet Train: {len(train_ds_triplet)} samples, Val: {len(val_ds_triplet)} samples\n")
    
    # Create triplet data loaders
    gen_train_triplet = torch.Generator()
    gen_train_triplet.manual_seed(SEED)
    train_batch_sampler_triplet = VideoBatchSampler(
        train_ds_triplet, batch_size=BATCH_SIZE, drop_last=False,
        shuffle_videos=True, generator=gen_train_triplet, batch_videos=False
    )
    train_loader_triplet = DataLoader(
        train_ds_triplet, batch_sampler=train_batch_sampler_triplet,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    gen_val_triplet = torch.Generator()
    gen_val_triplet.manual_seed(SEED)
    val_batch_sampler_triplet = VideoBatchSampler(
        val_ds_triplet, batch_size=BATCH_SIZE, drop_last=False,
        shuffle_videos=False, generator=gen_val_triplet, batch_videos=False
    )
    val_loader_triplet = DataLoader(
        val_ds_triplet, batch_sampler=val_batch_sampler_triplet,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # Triplet models to test
    triplet_models = ["TDN", "Rendezvous", "RiT"]
    
    # Train/evaluate each triplet model
    for model_name in triplet_models:
        checkpoint_path = CKPT_DIR / f"{model_name.lower()}_triplet_best.pth"
        
        if args.eval_only:
            # Only evaluate
            if checkpoint_path.exists():
                metrics = load_and_evaluate_triplet(
                    model_name, checkpoint_path, val_loader_triplet, device,
                    num_verbs=NUM_TRIPLET_VERBS, num_targets=NUM_TRIPLET_TARGETS
                )
                results[model_name] = metrics
            else:
                print(f"❌ Checkpoint not found for {model_name}: {checkpoint_path}")
                results[model_name] = {
                    "triplet_acc": 0.0, "ap_verbs": 0.0,
                    "ap_targets": 0.0, "ap_triplet": 0.0
                }
        else:
            # Train if needed, then evaluate
            if checkpoint_path.exists() and not args.force_retrain:
                print(f"✓ Found checkpoint for {model_name}: {checkpoint_path}")
                metrics = load_and_evaluate_triplet(
                    model_name, checkpoint_path, val_loader_triplet, device,
                    num_verbs=NUM_TRIPLET_VERBS, num_targets=NUM_TRIPLET_TARGETS
                )
            else:
                if args.force_retrain:
                    print(f"🔄 Force retraining {model_name}...")
                else:
                    print(f"🆕 No checkpoint found for {model_name}, training from scratch...")
                
                metrics = train_triplet_model(
                    model_name, train_loader_triplet, val_loader_triplet, device,
                    args.epochs, checkpoint_path,
                    num_verbs=NUM_TRIPLET_VERBS, num_targets=NUM_TRIPLET_TARGETS,
                    test_mode=args.test_mode
                )
                
                # Final evaluation with best checkpoint (skip in test mode)
                if not args.test_mode:
                    print(f"\n📊 Final evaluation of {model_name}...")
                    metrics = load_and_evaluate_triplet(
                        model_name, checkpoint_path, val_loader_triplet, device,
                        num_verbs=NUM_TRIPLET_VERBS, num_targets=NUM_TRIPLET_TARGETS
                    )
            
            results[model_name] = metrics
    
    # Print and save final table
    print_final_table(results)
    
    # Save results
    results_file = RESULTS_DIR / "final_results.txt"
    save_results(results, results_file)
    print(f"\n💾 Results saved to: {results_file}")
    
    # Save as CSV for easy import
    csv_file = RESULTS_DIR / "final_results.csv"
    with open(csv_file, "w") as f:
        # Write header
        f.write("Model,Type,Phase_Acc,MAE,inMAE,oMAE,wMAE,Triplet_Acc,APv,APt,APvt\n")
        
        # Write phase models
        for model_name, metrics in results.items():
            if 'phase_acc' in metrics:
                f.write(f"{model_name},Phase,")
                f.write(f"{metrics.get('phase_acc', 0)},")
                f.write(f"{metrics.get('mae', 0)},")
                f.write(f"{metrics.get('inMAE', 0)},")
                f.write(f"{metrics.get('oMAE', 0)},")
                f.write(f"{metrics.get('wMAE', 0)},")
                f.write(",,,,\n")
            elif 'triplet_acc' in metrics:
                f.write(f"{model_name},Triplet,")
                f.write(",,,,,")
                f.write(f"{metrics.get('triplet_acc', 0)},")
                f.write(f"{metrics.get('ap_verbs', 0)},")
                f.write(f"{metrics.get('ap_targets', 0)},")
                f.write(f"{metrics.get('ap_triplet', 0)}\n")
    print(f"💾 CSV results saved to: {csv_file}")


if __name__ == "__main__":
    main()
