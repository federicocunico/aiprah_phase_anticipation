import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    def __init__(self, epsilon=0.1):
        super(WeightedMSELoss, self).__init__()
        self.epsilon = epsilon  # Small constant to avoid division by zero

    def forward(self, predictions, targets):
        # Compute weights: w(y) = 1 / (y + epsilon)
        weights = 1 / (targets + self.epsilon)

        # Compute squared errors
        squared_errors = (predictions - targets) ** 2

        # Apply weights
        weighted_loss = weights * squared_errors

        # Return mean loss across all phases and samples
        return weighted_loss.mean()
