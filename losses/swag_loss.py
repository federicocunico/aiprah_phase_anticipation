import torch
import torch.nn as nn


class SWAGLoss(nn.Module):
    def __init__(self, future_steps, time_horizon):
        super(SWAGLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()  # For current phase classification
        self.bce_loss = nn.BCEWithLogitsLoss()  # For future anticipation
        self.mae_loss = nn.L1Loss()  # For regression tasks
        self.future_steps = future_steps
        self.time_horizon = time_horizon

    def r2c_mapping(self, reg_logits):
        """
        Map regression logits to classification probabilities (R2C).
        Args:
            reg_logits (torch.Tensor): Regression logits [B, T, C].
        Returns:
            torch.Tensor: Discretized probabilities [B, T, F, C].
        """
        B, T, C = reg_logits.size()
        discrete_probs = torch.zeros((B, T, self.future_steps, C), device=reg_logits.device)

        # Map remaining time to discrete bins
        for f in range(1, self.future_steps + 1):
            threshold = f * (self.time_horizon / self.future_steps)
            discrete_probs[:, :, f - 1, :] = (reg_logits <= threshold).float()

        return discrete_probs

    def forward(
        self,
        current_logits,
        future_logits,
        regression_logits,
        GT_current_targets,
        GT_future_targets,
        GT_regression_targets,
    ):
        """
        Compute the combined SWAG loss.

        Args:
            current_logits (torch.Tensor): [B, C], logits for current phase classification.
            future_logits (torch.Tensor): [B, T, F, C], logits for future phase classification.
            regression_logits (torch.Tensor): [B, T, C], logits for time-to-next-phase regression.
            GT_current_targets (torch.Tensor): [B], ground truth for current classification.
            GT_future_targets (torch.Tensor): [B, T, F, C], ground truth for future anticipation.
            GT_regression_targets (torch.Tensor): [B, C], ground truth for remaining time regression.

        Returns:
            dict: Dictionary with individual loss components and total loss.
        """
        # Current phase classification loss
        classification_loss = self.ce_loss(current_logits, GT_current_targets)

        # Future phase anticipation loss
        future_loss = self.bce_loss(future_logits, GT_future_targets.float())

        # Expand regression targets to match [B, T, C]
        GT_regression_targets_expanded = GT_regression_targets.unsqueeze(1).expand(-1, regression_logits.size(1), -1)

        # Regression loss for remaining time
        regression_loss = self.mae_loss(regression_logits, GT_regression_targets_expanded)

        # R2C mapping and classification
        r2c_probs = self.r2c_mapping(regression_logits)  # [B, T, F, C]
        r2c_loss = self.bce_loss(r2c_probs, GT_future_targets.float())

        # Combine losses
        total_loss = classification_loss + future_loss + regression_loss + r2c_loss

        return {
            "classification_loss": classification_loss,
            "future_loss": future_loss,
            "regression_loss": regression_loss,
            "r2c_loss": r2c_loss,
            "total_loss": total_loss,
        }
