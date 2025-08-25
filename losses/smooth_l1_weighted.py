import torch
from torch import nn

# ------------------------------ Interior-Weighted SmoothL1 ------------------------------
class InteriorWeightedSmoothL1(nn.Module):
    """
    Emphasize targets inside (0, T) while keeping gradients for extremes.
    weight(y) = base_weight + (1 - base_weight) * ((y/T) * (1 - y/T))^alpha, in [base_weight, 1]
      - alpha > 0 sharpens focus on interior; alpha=2..4 often works well.
      - base_weight > 0 ensures endpoints (0 or T) still contribute.
    """
    def __init__(self, time_horizon: float, alpha: float = 3.0, base_weight: float = 0.2, beta: float = 1.0):
        super().__init__()
        assert time_horizon > 0
        assert 0.0 <= base_weight < 1.0
        assert alpha > 0
        self.T = float(time_horizon)
        self.alpha = float(alpha)
        self.base_weight = float(base_weight)
        self.beta = float(beta)  # SmoothL1 delta

        # Use a reduction='none' SmoothL1 and apply custom weights
        self._huber = nn.SmoothL1Loss(reduction="none", beta=self.beta)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        preds, targets: [B, C] in [0, T]
        returns scalar loss
        """
        # Core Smooth L1 elementwise
        per_elem = self._huber(preds, targets)  # [B, C]

        # Compute interior weights from targets
        y = torch.clamp(targets / self.T, 0.0, 1.0)       # normalize to [0,1]
        w = (y * (1.0 - y)).pow(self.alpha)               # Beta-like bump centered at 0.5
        w = self.base_weight + (1.0 - self.base_weight) * w

        # Weighted mean
        loss = (per_elem * w).mean()
        return loss
