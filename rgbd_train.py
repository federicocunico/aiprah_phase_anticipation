import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Ensure CUDA errors are raised immediately
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

# Dataset imports
from datasets.cholec80 import Cholec80Dataset
from datasets.heichole import HeiCholeDataset

# from models.temporal_model_v5 import TemporalAnticipationModel  # Original model
from models.temporal_model_swin import TemporalAnticipationModel  # SwinTransformer model

# Loss imports
from losses.weighted_regr_loss import WeightedMSELoss

# Trainer import
from trainer_rgbd import create_trainer


def get_data_loaders(args):
    """Create data loaders based on dataset and configuration."""
    # Select dataset class
    if args.dataset == "cholec80":
        data_dir = "./data/cholec80"
        DatasetClass = Cholec80Dataset
    elif args.dataset == "heichole":
        data_dir = "./data/heichole"
        DatasetClass = HeiCholeDataset
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Create datasets
    dataset_kwargs = {"root_dir": data_dir, "seq_len": args.seq_len, "fps": args.target_fps}

    if args.debug:
        # Use demo subsets for debugging
        train_dataset = DatasetClass(mode="demo_train", **dataset_kwargs)
        val_dataset = DatasetClass(mode="demo_val", **dataset_kwargs)
        test_dataset = DatasetClass(mode="demo_test", **dataset_kwargs)
    else:
        # Use full datasets
        train_dataset = DatasetClass(mode="train", **dataset_kwargs)
        val_dataset = DatasetClass(mode="val", **dataset_kwargs)
        test_dataset = DatasetClass(mode="test", **dataset_kwargs)

    # Create data loaders
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def get_model(args):
    """Create model based on configuration."""
    model_kwargs = {
        "sequence_length": args.seq_len,
        "num_classes": args.num_classes,
        "time_horizon": args.time_horizon,
        "in_channels": args.in_channels,
    }

    # Add depth enhancer flag for SwinTransformer model
    if hasattr(TemporalAnticipationModel, "use_depth_enhancer"):
        model_kwargs["use_depth_enhancer"] = args.use_depth_enhancer

    model = TemporalAnticipationModel(**model_kwargs)

    # Load pretrained weights if specified
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"Loading pretrained weights from {args.pretrained_path}")
        state_dict = torch.load(args.pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    return model


def get_loss_criterion(args):
    """Create loss criterion based on configuration."""
    if args.loss_type == "mse":
        return nn.MSELoss()
    elif args.loss_type == "smooth_l1":
        return nn.SmoothL1Loss(reduction="mean")
    elif args.loss_type == "weighted_mse":
        return WeightedMSELoss()
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")


def train_model(args):
    """Main training function."""
    # Set random seed for reproducibility
    seed_everything(args.seed)

    # Set float32 matmul precision for better performance on modern GPUs
    torch.set_float32_matmul_precision(args.matmul_precision)

    # Create experiment name
    exp_name = f"{args.exp_name}_stage2_horizon={args.time_horizon}_ch={args.in_channels}"
    if args.use_rgbd:
        exp_name += "_rgbd"

    # Setup logging
    log_dir = f"checkpoints/{exp_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Initialize wandb logger
    wandb_logger = WandbLogger(name=exp_name, project=args.wandb_project, save_dir=log_dir, log_model=True)

    # Log hyperparameters
    wandb_logger.log_hyperparams(vars(args))

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = get_data_loaders(args)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    print(f"Creating model with {args.in_channels} input channels...")
    model = get_model(args)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create loss criterion
    loss_criterion = get_loss_criterion(args)

    # Create PyTorch Lightning module
    pl_model = create_trainer(
        model=model,
        loss_criterion=loss_criterion,
        use_rgbd=args.use_rgbd,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_plots=args.save_plots,
        checkpoint_dir=log_dir,
    )

    # Setup callbacks
    callbacks = []

    # Model checkpoint callback
    ckpt_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename="epoch={epoch}-val_loss={val_loss:.4f}-val_acc={val_acc:.4f}",
        monitor=args.monitor_metric,
        mode=args.monitor_mode,
        save_top_k=args.save_top_k,
        save_last=True,
        verbose=True,
    )
    callbacks.append(ckpt_callback)

    # Early stopping callback
    if args.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor=args.monitor_metric, mode=args.monitor_mode, patience=args.early_stopping_patience, verbose=True
        )
        callbacks.append(early_stop_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # Create trainer
    trainer = Trainer(
        accelerator="auto",
        devices=[args.gpu],
        precision=args.precision,
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=args.gradient_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.val_check_interval,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train the model
    print("Starting training...")
    trainer.fit(pl_model, train_loader, val_loader)

    # Test the model with best checkpoint
    print("Testing with best checkpoint...")
    trainer.test(pl_model, test_loader, ckpt_path="best")

    # Save final results
    results = {
        "best_val_loss": float(trainer.callback_metrics.get("val_loss", -1)),
        "best_val_acc": float(trainer.callback_metrics.get("val_acc", -1)),
        "test_acc": float(trainer.callback_metrics.get("test_acc", -1)),
        "test_f1": float(trainer.callback_metrics.get("test_f1", -1)),
        "test_anticip": float(trainer.callback_metrics.get("test_anticip", -1)),
    }

    # Log final results
    wandb_logger.log_metrics(results)

    print("\nTraining completed!")
    print(f"Results saved to: {log_dir}")
    print(f"Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"Test accuracy: {results['test_acc']:.4f}")
    print(f"Test F1 score: {results['test_f1']:.4f}")
    print(f"Test anticipation MSE: {results['test_anticip']:.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Temporal Anticipation Model")

    # Experiment settings
    parser.add_argument("--exp_name", type=str, default="temporal_anticipation", help="Experiment name for logging")
    parser.add_argument("--wandb_project", type=str, default="cholec80", help="Weights & Biases project name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true", help="Use demo datasets for debugging")

    # Model settings
    parser.add_argument(
        "--in_channels", type=int, default=4, choices=[3, 4], help="Number of input channels (3 for RGB, 4 for RGB-D)"
    )
    parser.add_argument("--use_rgbd", action="store_true", default=True, help="Use RGB-D data (automatically sets in_channels to 4)")
    parser.add_argument(
        "--use_depth_enhancer", type=bool, default=True, help="Use depth enhancement module (only for RGB-D)"
    )
    parser.add_argument("--time_horizon", type=int, default=5, help="Time horizon for anticipation in minutes")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of surgical phases")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to pretrained model weights")

    # Dataset settings
    parser.add_argument(
        "--dataset", type=str, default="cholec80", choices=["cholec80", "heichole"], help="Dataset to use for training"
    )
    parser.add_argument("--seq_len", type=int, default=30, help="Sequence length for temporal modeling")
    parser.add_argument("--target_fps", type=int, default=1, help="Target FPS for video subsampling")

    # Training settings
    parser.add_argument("--gpu", type=int, default=1, help="GPU device index to use")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument(
        "--accumulate_grad_batches", type=int, default=1, help="Number of batches to accumulate gradients"
    )

    # Loss settings
    parser.add_argument(
        "--loss_type",
        type=str,
        default="smooth_l1",
        choices=["mse", "smooth_l1", "weighted_mse"],
        help="Loss function type",
    )

    # Monitoring settings
    parser.add_argument("--monitor_metric", type=str, default="val_acc", help="Metric to monitor for checkpointing")
    parser.add_argument(
        "--monitor_mode", type=str, default="max", choices=["min", "max"], help="Mode for monitoring metric"
    )
    parser.add_argument("--save_top_k", type=int, default=3, help="Number of best checkpoints to save")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--val_check_interval", type=int, default=1, help="Check validation every n epochs")

    # System settings
    parser.add_argument("--num_workers", type=int, default=20, help="Number of data loading workers")
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Training precision",
    )
    parser.add_argument(
        "--matmul_precision",
        type=str,
        default="high",
        choices=["high", "medium", "highest"],
        help="Matrix multiplication precision",
    )
    parser.add_argument("--save_plots", type=bool, default=True, help="Save validation/test plots")

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
