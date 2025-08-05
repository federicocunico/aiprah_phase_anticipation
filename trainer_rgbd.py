import os
from typing import Dict, List, Tuple, Union
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score


class PhaseAnticipationTrainer(pl.LightningModule):
    """PyTorch Lightning module for training phase anticipation models."""
    
    def __init__(
        self, 
        model: nn.Module, 
        loss_criterion: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        use_rgbd: bool = False,
        save_plots: bool = True,
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model to train
            loss_criterion: Loss function to use
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            use_rgbd: Whether to use RGB-D data (4 channels) or RGB only (3 channels)
            save_plots: Whether to save validation/test plots
            checkpoint_dir: Directory to save checkpoints
        """
        super().__init__()
        self.model = model
        self.loss_criterion = loss_criterion
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_rgbd = use_rgbd
        self.save_plots = save_plots
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize plot
        if self.save_plots:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(15, 5))
        
        # Track best validation loss
        self.best_val_loss = float("inf")
        
        # Lists to store predictions and ground truth
        self.val_predictions = []
        self.val_ground_truth = []
        self.val_regression_predictions = []
        self.val_regression_ground_truth = []
        
        self.test_predictions = []
        self.test_ground_truth = []
        self.test_regression_predictions = []
        self.test_regression_ground_truth = []
    
    def configure_optimizers(self):
        """Configure the optimizer."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        return optimizer
    
    def _get_input_data(self, batch: Tuple) -> torch.Tensor:
        """
        Extract the appropriate input data based on the model configuration.
        
        Args:
            batch: Tuple of (frames, metadata)
            
        Returns:
            Input tensor with appropriate number of channels
        """
        frames, metadata = batch
        
        if self.use_rgbd:
            # Use RGB-D data if available
            if 'frames_rgbd' in metadata:
                return metadata['frames_rgbd']
            else:
                # Fallback to RGB if RGB-D not available
                print("Warning: RGB-D requested but not available in batch. Using RGB only.")
                return frames
        else:
            # Use RGB data
            return frames
    
    def _process_outputs(
        self, 
        outputs: Union[torch.Tensor, Tuple], 
        metadata: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        Process model outputs and compute losses based on output type.
        
        Args:
            outputs: Model outputs (can be tensor or tuple)
            metadata: Batch metadata containing labels and targets
            
        Returns:
            Dictionary containing losses and predictions
        """
        labels = metadata["phase_label"]
        time_to_next_phase = metadata["time_to_next_phase"]
        time_to_next_phase = torch.clamp(time_to_next_phase, 0, self.model.time_horizon)
        
        result = {}
        
        # Handle different output formats
        if isinstance(outputs, torch.Tensor):
            # Single regression output
            regression_logits = outputs
            
            # Compute loss
            result['loss'] = self.loss_criterion(regression_logits, time_to_next_phase)
            result['loss_phase'] = torch.tensor(0.0)
            result['loss_anticipation'] = result['loss']
            
            # For phase prediction, use the phase with minimum predicted time
            result['predicted_phase'] = torch.argmin(regression_logits, dim=1)
            result['regression_output'] = regression_logits
            
        elif isinstance(outputs, tuple) and len(outputs) == 2:
            # Current phase classification + regression
            current_phase, regression_output = outputs
            
            # Compute losses
            if hasattr(self.loss_criterion, '__call__'):
                loss_output = self.loss_criterion(
                    current_phase, regression_output, labels, time_to_next_phase
                )
                if isinstance(loss_output, tuple):
                    result['loss'], result['loss_phase'], result['loss_anticipation'] = loss_output
                else:
                    result['loss'] = loss_output
                    result['loss_phase'] = torch.tensor(0.0)
                    result['loss_anticipation'] = result['loss']
            
            # Get predictions
            result['predicted_phase'] = torch.argmax(current_phase, dim=1)
            result['regression_output'] = regression_output
            
        else:
            # Handle other output formats (e.g., current + future + regression)
            current_logits = outputs[0]
            regression_logits = outputs[-1] if len(outputs) > 2 else outputs[1]
            
            # Simple fallback loss computation
            result['loss'] = self.loss_criterion(regression_logits, time_to_next_phase)
            result['loss_phase'] = torch.tensor(0.0)
            result['loss_anticipation'] = result['loss']
            
            result['predicted_phase'] = torch.argmax(current_logits, dim=1)
            result['regression_output'] = regression_logits
        
        result['labels'] = labels
        result['time_targets'] = time_to_next_phase
        
        return result
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        input_data = self._get_input_data(batch)
        outputs = self.model(input_data)
        
        results = self._process_outputs(outputs, batch[1])
        
        # Log losses
        self.log("train_loss", results['loss'], prog_bar=True)
        self.log("train_loss_phase", results['loss_phase'])
        self.log("train_loss_anticipation", results['loss_anticipation'])
        
        return results['loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_data = self._get_input_data(batch)
        outputs = self.model(input_data)
        
        results = self._process_outputs(outputs, batch[1])
        
        # Store predictions for epoch-end metrics
        self.val_predictions.extend(results['predicted_phase'].detach().cpu().tolist())
        self.val_ground_truth.extend(results['labels'].detach().cpu().tolist())
        
        # Store regression outputs (per-class time predictions)
        for i in range(results['labels'].size(0)):
            phase_idx = results['labels'][i].item()
            self.val_regression_predictions.append(
                results['regression_output'][i, phase_idx].detach().cpu().item()
            )
            self.val_regression_ground_truth.append(
                results['time_targets'][i, phase_idx].detach().cpu().item()
            )
        
        # Log losses
        self.log("val_loss", results['loss'], prog_bar=True)
        self.log("val_loss_phase", results['loss_phase'])
        self.log("val_loss_anticipation", results['loss_anticipation'])
        
        return results['loss']
    
    def on_validation_epoch_start(self):
        """Reset validation metrics at epoch start."""
        self.val_predictions.clear()
        self.val_ground_truth.clear()
        self.val_regression_predictions.clear()
        self.val_regression_ground_truth.clear()
    
    def on_validation_epoch_end(self):
        """Compute and log validation metrics at epoch end."""
        # Compute classification accuracy
        acc = accuracy_score(self.val_ground_truth, self.val_predictions)
        self.log("val_acc", acc)
        
        # Compute regression MSE
        mse_loss = nn.functional.mse_loss(
            torch.tensor(self.val_regression_predictions),
            torch.tensor(self.val_regression_ground_truth)
        )
        self.log("val_anticip", mse_loss)
        
        # Save best model
        if mse_loss < self.best_val_loss:
            self.best_val_loss = mse_loss
            checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(self.model.state_dict(), checkpoint_path)
            self.log("best_val_loss", self.best_val_loss)
        
        # Save plot
        if self.save_plots:
            self._save_plot(
                self.val_regression_ground_truth,
                self.val_regression_predictions,
                "val_time_phase.png",
                "Validation: Time to Next Phase"
            )
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        input_data = self._get_input_data(batch)
        outputs = self.model(input_data)
        
        results = self._process_outputs(outputs, batch[1])
        
        # Store predictions for epoch-end metrics
        self.test_predictions.extend(results['predicted_phase'].detach().cpu().tolist())
        self.test_ground_truth.extend(results['labels'].detach().cpu().tolist())
        
        # Store regression outputs
        for i in range(results['labels'].size(0)):
            phase_idx = results['labels'][i].item()
            self.test_regression_predictions.append(
                results['regression_output'][i, phase_idx].detach().cpu().item()
            )
            self.test_regression_ground_truth.append(
                results['time_targets'][i, phase_idx].detach().cpu().item()
            )
        
        # Log losses
        self.log("test_loss", results['loss'])
        self.log("test_loss_phase", results['loss_phase'])
        self.log("test_loss_anticipation", results['loss_anticipation'])
        
        return results['loss']
    
    def on_test_epoch_start(self):
        """Reset test metrics at epoch start."""
        self.test_predictions.clear()
        self.test_ground_truth.clear()
        self.test_regression_predictions.clear()
        self.test_regression_ground_truth.clear()
    
    def on_test_epoch_end(self):
        """Compute and log test metrics at epoch end."""
        # Compute classification metrics
        acc = accuracy_score(self.test_ground_truth, self.test_predictions)
        f1 = f1_score(self.test_ground_truth, self.test_predictions, average="weighted")
        self.log("test_acc", acc)
        self.log("test_f1", f1)
        
        # Compute regression MSE
        mse_loss = nn.functional.mse_loss(
            torch.tensor(self.test_regression_predictions),
            torch.tensor(self.test_regression_ground_truth)
        )
        self.log("test_anticip", mse_loss)
        
        # Save plot
        if self.save_plots:
            self._save_plot(
                self.test_regression_ground_truth,
                self.test_regression_predictions,
                "test_time_phase.png",
                "Test: Time to Next Phase"
            )
    
    def _save_plot(self, ground_truth: List, predictions: List, filename: str, title: str):
        """Save a plot comparing ground truth and predictions."""
        self.ax.clear()
        self.ax.set_title(title)
        self.ax.set_ylabel("Time to next phase (minutes, capped at 5)")
        self.ax.set_xlabel("Sample index")
        self.ax.plot(ground_truth, label="Ground Truth", color="blue", alpha=0.7)
        self.ax.plot(predictions, label="Predicted", color="red", alpha=0.7)
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {filename}")


def create_trainer(
    model: nn.Module,
    loss_criterion: nn.Module,
    use_rgbd: bool = False,
    **kwargs
) -> PhaseAnticipationTrainer:
    """
    Factory function to create a trainer instance.
    
    Args:
        model: The model to train
        loss_criterion: Loss function to use
        use_rgbd: Whether to use RGB-D data
        **kwargs: Additional arguments passed to PhaseAnticipationTrainer
        
    Returns:
        PhaseAnticipationTrainer instance
    """
    return PhaseAnticipationTrainer(
        model=model,
        loss_criterion=loss_criterion,
        use_rgbd=use_rgbd,
        **kwargs
    )