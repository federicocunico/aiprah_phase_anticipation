#!/usr/bin/env python3
"""
Train and Evaluate All Models on PegAndRing Dataset

Trains MuST, SWAG, and SKiT (simplified adapters) on PegAndRing dataset.
- Skips training if checkpoint exists
- Evaluates all models
- Generates final comparison table with Phase Acc., MAE, inMAE, oMAE, wMAE

Usage:
    python train_and_evaluate_all.py [--epochs 50] [--force-retrain]
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import time
from collections import defaultdict

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Import dataset
from datasets.peg_and_ring_workflow import PegAndRing, VideoBatchSampler

# Configuration
SEED = 42
ROOT_DIR = "data/peg_and_ring_workflow"
TIME_UNIT = "minutes"
NUM_CLASSES = 6
BATCH_SIZE = 24
NUM_WORKERS = 30
SEQ_LEN = 16
STRIDE = 1
TIME_HORIZON = 2.0

# Training config
DEFAULT_EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0

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


class SimplifiedModelAdapter(nn.Module):
    """Simplified model adapter using ResNet18 + LSTM."""
    
    def __init__(self, num_classes=6, model_name="baseline"):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        
        from torchvision.models import resnet18
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # Temporal encoding
        self.temporal = nn.LSTM(512, 256, num_layers=2, batch_first=True, bidirectional=True)
        
        # Task heads
        self.phase_head = nn.Linear(512, num_classes)
        self.time_head = nn.Linear(512, num_classes)
        
    def forward(self, frames, meta=None):
        """
        Args:
            frames: (B, T, C, H, W)
        Returns:
            phase_logits: (B, num_classes)
            time_pred: (B, num_classes)
        """
        B, T, C, H, W = frames.shape
        
        # Spatial features
        frames_flat = frames.view(B * T, C, H, W)
        spatial_feats = self.backbone(frames_flat)
        spatial_feats = spatial_feats.view(B, T, 512)
        
        # Temporal aggregation
        temporal_feats, _ = self.temporal(spatial_feats)
        agg_feats = temporal_feats.mean(dim=1)
        
        # Predictions
        phase_logits = self.phase_head(agg_feats)
        time_pred = self.time_head(agg_feats)
        
        return phase_logits, time_pred


@torch.no_grad()
def evaluate_model(model, loader, device, model_name):
    """Evaluate model on validation set."""
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
        
        # Forward pass
        phase_logits, time_pred = model(frames)
        
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


def train_epoch(model, loader, optimizer, criterion_ce, criterion_reg, device):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_phase_loss = 0.0
    total_time_loss = 0.0
    num_batches = 0
    
    # log steps, 10 logs per epoch
    N = len(loader)
    log_steps = torch.linspace(0, N, steps=10).int()

    for i, (frames, meta) in enumerate(loader):
        frames = frames.to(device)
        phase_gt = meta["phase_label"].to(device).long()
        time_to_next = meta["time_to_next_phase"].to(device).float()
        time_to_next = torch.clamp(time_to_next, min=0.0, max=TIME_HORIZON)
        
        # Forward pass
        phase_logits, time_pred = model(frames)
        
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

        # Print every 50 batches
        if i in log_steps:
            print(f"  Batch {i+1}/{len(loader)} - "
                  f"Loss: {total_loss / num_batches:.4f} "
                  f"(Phase: {total_phase_loss / num_batches:.4f}, "
                  f"Time: {total_time_loss / num_batches:.4f})")

    return {
        "loss": total_loss / num_batches,
        "phase_loss": total_phase_loss / num_batches,
        "time_loss": total_time_loss / num_batches,
    }


def train_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    checkpoint_path: Path,
):
    """Train a model from scratch."""
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}\n")
    
    # Create model
    model = SimplifiedModelAdapter(num_classes=NUM_CLASSES, model_name=model_name).to(device)
    
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
    
    for epoch in range(1, epochs + 1):
        # Train
        train_stats = train_epoch(
            model, train_loader, optimizer, criterion_ce, criterion_reg, device
        )
        
        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            val_metrics = evaluate_model(model, val_loader, device, model_name)
            
            print(f"Epoch {epoch}/{epochs}")
            print(f"  Train Loss: {train_stats['loss']:.4f} "
                  f"(Phase: {train_stats['phase_loss']:.4f}, Time: {train_stats['time_loss']:.4f})")
            print(f"  Val Phase Acc: {val_metrics['phase_acc']:.4f}, "
                  f"MAE: {val_metrics['mae']:.4f}, wMAE: {val_metrics['wMAE']:.4f}")
            
            # Save best model
            if val_metrics['phase_acc'] > best_val_acc:
                best_val_acc = val_metrics['phase_acc']
                best_metrics = val_metrics
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  💾 Saved best model (acc: {best_val_acc:.4f})")
        
        scheduler.step()
    
    print(f"\n✅ {model_name} Training Complete!")
    print(f"   Best Val Acc: {best_val_acc:.4f}")
    
    return best_metrics


def load_and_evaluate(
    model_name: str,
    checkpoint_path: Path,
    val_loader: DataLoader,
    device: torch.device,
):
    """Load checkpoint and evaluate."""
    print(f"\n📊 Evaluating {model_name} from checkpoint...")
    
    model = SimplifiedModelAdapter(num_classes=NUM_CLASSES, model_name=model_name).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    metrics = evaluate_model(model, val_loader, device, model_name)
    
    print(f"✅ {model_name} Evaluation:")
    print(f"   Phase Acc: {metrics['phase_acc']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}, inMAE: {metrics['inMAE']:.4f}, "
          f"oMAE: {metrics['oMAE']:.4f}, wMAE: {metrics['wMAE']:.4f}")
    
    return metrics


def print_final_table(results: Dict[str, Dict[str, float]]):
    """Print final comparison table."""
    print("\n" + "="*90)
    print("FINAL MODEL COMPARISON RESULTS")
    print("="*90)
    
    # Header
    header = f"{'Model':<15}"
    for metric in ["Phase Acc.", "MAE", "inMAE", "oMAE", "wMAE"]:
        header += f"{metric:>13}"
    print(header)
    print("-"*90)
    
    # Rows
    for model_name, metrics in results.items():
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
        print(row)
    
    print("="*90)
    print("\nMetrics Explanation:")
    print("  Phase Acc. : Current phase classification accuracy")
    print("  MAE        : Mean Absolute Error for time-to-next-phase prediction")
    print("  inMAE      : MAE for in-horizon predictions (time < horizon)")
    print("  oMAE       : MAE for out-of-horizon predictions (time >= horizon)")
    print("  wMAE       : Weighted MAE (average of inMAE and oMAE)")
    print("="*90)


def save_results(results: Dict[str, Dict[str, float]], filepath: Path):
    """Save results to file."""
    with open(filepath, "w") as f:
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("="*90 + "\n\n")
        
        f.write(f"{'Model':<15} {'Phase Acc.':<12} {'MAE':<10} {'inMAE':<10} {'oMAE':<10} {'wMAE':<10}\n")
        f.write("-"*90 + "\n")
        
        for model_name, metrics in results.items():
            f.write(f"{model_name:<15} ")
            f.write(f"{metrics.get('phase_acc', 0)*100:<11.2f}% ")
            f.write(f"{metrics.get('mae', 0):<10.4f} ")
            f.write(f"{metrics.get('inMAE', 0):<10.4f} ")
            f.write(f"{metrics.get('oMAE', 0):<10.4f} ")
            f.write(f"{metrics.get('wMAE', 0):<10.4f}\n")
        
        f.write("\n" + "="*90 + "\n")
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
    args = parser.parse_args()
    
    print("🚀 Training and Evaluating Models on PegAndRing Dataset")
    print(f"Models: MuST, SWAG, SKiT (Simplified Adapters)")
    print(f"Epochs: {args.epochs}, Batch Size: {BATCH_SIZE}, Seq Len: {SEQ_LEN}")
    print(f"Force Retrain: {args.force_retrain}, Eval Only: {args.eval_only}\n")
    
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
                    args.epochs, checkpoint_path
                )
                
                # Final evaluation with best checkpoint
                print(f"\n📊 Final evaluation of {model_name}...")
                metrics = load_and_evaluate(model_name, checkpoint_path, val_loader, device)
            
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
        f.write("Model,Phase_Acc,MAE,inMAE,oMAE,wMAE\n")
        for model_name, metrics in results.items():
            f.write(f"{model_name},")
            f.write(f"{metrics.get('phase_acc', 0)},")
            f.write(f"{metrics.get('mae', 0)},")
            f.write(f"{metrics.get('inMAE', 0)},")
            f.write(f"{metrics.get('oMAE', 0)},")
            f.write(f"{metrics.get('wMAE', 0)}\n")
    print(f"💾 CSV results saved to: {csv_file}")


if __name__ == "__main__":
    main()
