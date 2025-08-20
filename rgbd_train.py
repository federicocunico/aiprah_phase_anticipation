import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set to your desired GPU index
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

# Dataset imports
from datasets.cholec80 import Cholec80Dataset
from datasets.heichole import HeiCholeDataset

# Model import
# from models.temporal_model_swin import TemporalAnticipationModel
# from models.temporal_model_swin_v2 import create_model
from models.temporal_model_swin_v3 import create_model


# Loss imports
from losses.weighted_regr_loss import WeightedMSELoss

import wandb


class TrainingMetrics:
    """Helper class to track and compute metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.ground_truth = []
        self.regression_predictions = []
        self.regression_ground_truth = []
        self.losses = []
        self.phase_losses = []
        self.anticip_losses = []
    
    def update(self, predicted_phase, labels, regression_output, time_targets, loss, loss_phase=0, loss_anticip=0):
        """Update metrics with batch results."""
        self.predictions.extend(predicted_phase.detach().cpu().tolist())
        self.ground_truth.extend(labels.detach().cpu().tolist())
        
        # Store regression outputs (per-class time predictions)
        for i in range(labels.size(0)):
            phase_idx = labels[i].item()
            self.regression_predictions.append(
                regression_output[i, phase_idx].detach().cpu().item()
            )
            self.regression_ground_truth.append(
                time_targets[i, phase_idx].detach().cpu().item()
            )
        
        self.losses.append(loss.item())
        self.phase_losses.append(loss_phase if isinstance(loss_phase, float) else loss_phase.item())
        self.anticip_losses.append(loss_anticip if isinstance(loss_anticip, float) else loss_anticip.item())
    
    def compute_metrics(self):
        """Compute all metrics."""
        metrics = {
            'accuracy': accuracy_score(self.ground_truth, self.predictions),
            'f1': f1_score(self.ground_truth, self.predictions, average='weighted'),
            'loss': np.mean(self.losses),
            'loss_phase': np.mean(self.phase_losses),
            'loss_anticip': np.mean(self.anticip_losses),
        }
        
        if self.regression_predictions:
            metrics['anticip_mse'] = nn.functional.mse_loss(
                torch.tensor(self.regression_predictions),
                torch.tensor(self.regression_ground_truth)
            ).item()
        
        return metrics


def get_input_data(batch: Tuple, use_rgbd: bool) -> torch.Tensor:
    """Extract the appropriate input data based on configuration."""
    frames, metadata = batch
    
    if use_rgbd:
        if 'frames_rgbd' in metadata:
            return metadata['frames_rgbd']
        else:
            print("Warning: RGB-D requested but not available. Using RGB only.")
            return frames
    else:
        return frames


def process_outputs(outputs, metadata, model, loss_criterion, device):
    """Process model outputs and compute losses."""
    labels = metadata["phase_label"]  #.to(device)
    time_to_next_phase = metadata["time_to_next_phase"].to(device)
    time_to_next_phase = torch.clamp(time_to_next_phase, 0, model.time_horizon)
    
    # Handle single tensor output (regression only)
    if isinstance(outputs, torch.Tensor):
        regression_logits = outputs
        loss = loss_criterion(regression_logits, time_to_next_phase)
        loss_phase = 0.0
        loss_anticipation = loss
        predicted_phase = torch.argmin(regression_logits, dim=1)
        regression_output = regression_logits
    else:
        # Handle other output formats if needed
        raise ValueError("Unsupported output format")
    
    return {
        'loss': loss,
        'loss_phase': loss_phase,
        'loss_anticipation': loss_anticipation,
        'predicted_phase': predicted_phase,
        'regression_output': regression_output,
        'labels': labels,
        'time_targets': time_to_next_phase
    }


def save_plot(ground_truth: List, predictions: List, filename: str, title: str):
    """Save a comparison plot."""
    plt.figure(figsize=(15, 5))
    plt.title(title)
    plt.ylabel("Time to next phase (minutes, capped at 5)")
    plt.xlabel("Sample index")
    plt.plot(ground_truth, label="Ground Truth", color="blue", alpha=0.7)
    plt.plot(predictions, label="Predicted", color="red", alpha=0.7)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {filename}")


def train_epoch(model, train_loader, optimizer, loss_criterion, device, use_rgbd, use_amp):
    """Train for one epoch."""
    model.train()
    metrics = TrainingMetrics()
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        input_data = get_input_data(batch, use_rgbd).to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        
        outputs = model(input_data)
        results = process_outputs(outputs, batch[1], model, loss_criterion, device)
        results['loss'].backward()
        optimizer.step()
    
        # Update metrics
        metrics.update(
            results['predicted_phase'],
            results['labels'],
            results['regression_output'],
            results['time_targets'],
            results['loss'],
            results['loss_phase'],
            results['loss_anticipation']
        )
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{results['loss'].item():.4f}",
            # 'anticip': f"{results['loss_anticipation']:.4f}",
            # 'phase': f"{results['loss_phase']:.4f}",
        })

        wandb.log({
            'train_step_loss': results['loss'].item(),
        })
    
    return metrics.compute_metrics()


def validate_epoch(model, val_loader, loss_criterion, device, use_rgbd, save_plots=False, plot_prefix="val"):
    """Validate for one epoch."""
    model.eval()
    metrics = TrainingMetrics()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for batch_idx, batch in enumerate(pbar):
            input_data = get_input_data(batch, use_rgbd).to(device)
            
            outputs = model(input_data)
            results = process_outputs(outputs, batch[1], model, loss_criterion, device)
            
            # Update metrics
            metrics.update(
                results['predicted_phase'],
                results['labels'],
                results['regression_output'],
                results['time_targets'],
                results['loss'],
                results['loss_phase'],
                results['loss_anticipation']
            )
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{results['loss'].item():.4f}",
                # 'anticip': f"{results['loss_anticipation']:.4f}"
            })
    
    computed_metrics = metrics.compute_metrics()
    
    # Save plots if requested
    if save_plots:
        save_plot(
            metrics.regression_ground_truth,
            metrics.regression_predictions,
            f"{plot_prefix}_time_phase.png",
            f"{plot_prefix.capitalize()}: Time to Next Phase"
        )
    
    return computed_metrics


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path, model_config):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'model_config': model_config
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(model, optimizer, scaler, checkpoint_path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scaler and checkpoint['scaler_state_dict']:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint['epoch']


def train_model(args):
    """Main training function."""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create experiment directory
    exp_name = f"{args.exp_name}_horizon={args.time_horizon}_ch={args.in_channels}"
    if args.use_rgbd:
        exp_name += "_rgbd"
    
    log_dir = f"checkpoints/{exp_name}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Experiment directory: {log_dir}")
    
    # Save configuration
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize wandb if available
    
    wandb.init(
        project=args.wandb_project,
        name=exp_name,
        config=vars(args),
        dir=log_dir
    )
    
    # Create data loaders
    print("\nCreating data loaders...")
    if args.dataset == "cholec80":
        data_dir = "./data/cholec80"
        DatasetClass = Cholec80Dataset
    elif args.dataset == "heichole":
        raise NotImplementedError("HeiChole dataset support is not implemented yet.")
        data_dir = "./data/heichole"
        DatasetClass = HeiCholeDataset
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    dataset_kwargs = {
        "root_dir": data_dir,
        "seq_len": args.seq_len,
        "fps": args.target_fps
    }
    
    train_dataset = DatasetClass(mode="train", **dataset_kwargs)
    val_dataset = DatasetClass(mode="val", **dataset_kwargs)
    test_dataset = DatasetClass(mode="test", **dataset_kwargs)
    
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print(f"\nCreating model with {args.in_channels} input channels...")

    # model_config = {
    #     "sequence_length" : args.seq_len,  # T=30 frames
    #     "num_classes" : args.num_classes,    # 7 surgical phases
    #     "time_horizon" : args.time_horizon,   # 5 minutes anticipation
    #     "in_channels" : args.in_channels,     # 3 for RGB, 4 for RGB-D
    #     "swin_model_size" : "base",           # Best balance for surgical detail recognition
    #     "temporal_processor" : "bert",         # Better for short sequences
    #     "num_encoder_layers" : 8,             # Increased for surgical complexity
    #     "num_decoder_layers" : 4,             # Lighter decoder for efficiency
    #     "nhead" : 12,                          # More attention heads for fine details
    #     "use_pretrained_bert" : True,         # Leverage pre-trained knowledge
    #     "bert_model_name" : "bert-base-uncased"  # Better performance than BERT
    # }
    # model = create_model(**model_config).to(device)

    model_config = {
        "sequence_length": args.seq_len,
        "num_classes": args.num_classes,
        "time_horizon": args.time_horizon,
        "in_channels": args.in_channels,  # RGB-D input
    }
    model = create_model(**model_config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss criterion
    if args.loss_type == "mse":
        loss_criterion = nn.MSELoss()
    elif args.loss_type == "smooth_l1":
        loss_criterion = nn.SmoothL1Loss(reduction="mean")
    elif args.loss_type == "weighted_mse":
        loss_criterion = WeightedMSELoss()
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0
    )
    
    # Training state
    start_epoch = 0
    best_val_acc = 0
    best_val_mse = float('inf')
    patience_counter = 0
        
    # Training loop
    print("\nStarting training...")
    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_criterion, device, args.use_rgbd, args.use_amp
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, loss_criterion, device, args.use_rgbd,
            save_plots=args.save_plots, plot_prefix="val"
        )
        
        # Update learning rate
        scheduler.step(val_metrics['accuracy'])
        
        # Print metrics
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1} completed in {epoch_time:.1f}s")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"MSE: {val_metrics.get('anticip_mse', 0):.4f}")
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_anticip_mse': val_metrics.get('anticip_mse', 0),
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Save checkpoint
        checkpoint_path = os.path.join(log_dir, f"checkpoint_epoch_{epoch+1}.pth")
        save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path, model_config)
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_checkpoint_path = os.path.join(log_dir, "best_model.pth")
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"New best validation accuracy: {best_val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if args.early_stopping and patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        # Keep only last N checkpoints
        if args.keep_last_n > 0:
            checkpoints = sorted([f for f in os.listdir(log_dir) if f.startswith("checkpoint_epoch_")])
            for old_checkpoint in checkpoints[:-args.keep_last_n]:
                os.remove(os.path.join(log_dir, old_checkpoint))
    
    # Final evaluation on test set
    print("\n" + "="*50)
    print("Testing with best checkpoint...")
    print("="*50)
    
    # Load best model
    best_checkpoint_path = os.path.join(log_dir, "best_model.pth")
    if os.path.exists(best_checkpoint_path):
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    
    # Test
    test_metrics = validate_epoch(
        model, test_loader, loss_criterion, device, args.use_rgbd,
        save_plots=args.save_plots, plot_prefix="test"
    )
    
    total_training_time = time.time() - training_start_time
    
    # Print final results
    print(f"\nTest Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"MSE: {test_metrics.get('anticip_mse', 0):.4f}")
    print(f"\nTotal training time: {total_training_time/3600:.1f} hours")
    
    # Save final results
    results = {
        "best_val_acc": best_val_acc,
        "test_acc": test_metrics['accuracy'],
        "test_f1": test_metrics['f1'],
        "test_anticip_mse": test_metrics.get('anticip_mse', 0),
        "total_training_time": total_training_time,
        "config": vars(args)
    }
    
    results_path = os.path.join(log_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Log final results to wandb
    wandb.log({
        "test_acc": test_metrics['accuracy'],
        "test_f1": test_metrics['f1'],
        "test_anticip_mse": test_metrics.get('anticip_mse', 0),
    })
    wandb.finish()
    
    print(f"\nTraining completed! Results saved to {log_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Temporal Anticipation Model (Pure PyTorch)")
    
    # Experiment settings
    parser.add_argument("--exp_name", type=str, default="temporal_anticipation", 
                        help="Experiment name for logging")
    parser.add_argument("--wandb_project", type=str, default="cholec80", 
                        help="Weights & Biases project name")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true", 
                        help="Use demo datasets for debugging")
    
    # Model settings
    parser.add_argument("--in_channels", type=int, default=3, choices=[3, 4],
                        help="Number of input channels (3 for RGB, 4 for RGB-D)")
    parser.add_argument("--use_rgbd", action="store_true", default=True,
                        help="Use RGB-D data (automatically sets in_channels to 4)")
    parser.add_argument("--time_horizon", type=int, default=5, 
                        help="Time horizon for anticipation in minutes")
    parser.add_argument("--num_classes", type=int, default=7, 
                        help="Number of surgical phases")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to pretrained model weights")
    
    # Dataset settings
    parser.add_argument("--dataset", type=str, default="cholec80", 
                        choices=["cholec80", "heichole"],
                        help="Dataset to use for training")
    parser.add_argument("--seq_len", type=int, default=4, 
                        help="Sequence length for temporal modeling")
    parser.add_argument("--target_fps", type=int, default=1, 
                        help="Target FPS for video subsampling")
    
    # Training settings
    parser.add_argument("--gpu", type=int, default=0, 
                        help="GPU device index to use")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=24, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use automatic mixed precision training")
    
    # Loss settings
    parser.add_argument("--loss_type", type=str, default="smooth_l1", 
                        choices=["mse", "smooth_l1", "weighted_mse"],
                        help="Loss function type")
    
    # Monitoring settings
    parser.add_argument("--early_stopping", action="store_true", 
                        help="Enable early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=10, 
                        help="Early stopping patience")
    parser.add_argument("--save_plots", type=bool, default=True,
                        help="Save validation/test plots")
    parser.add_argument("--keep_last_n", type=int, default=3,
                        help="Keep only last N checkpoints")
    
    # System settings
    parser.add_argument("--num_workers", type=int, default=8, 
                        help="Number of data loading workers")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Handle RGB-D flag
    if args.use_rgbd:
        args.in_channels = 4
    
    # Print configuration
    print("\nTraining Configuration:")
    print("-" * 50)
    for key, value in sorted(vars(args).items()):
        print(f"{key:.<30} {value}")
    print("-" * 50)
    print()
    
    # Run training
    train_model(args)


if __name__ == "__main__":
    main()