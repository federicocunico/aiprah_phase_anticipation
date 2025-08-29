from typing import Any, Dict, Tuple
from pyparsing import Optional
import torch
from torch import nn
from torch.nn import functional as F

try:
    import torchvision
    from torchvision.models import (
        resnet18,
        resnet50,
        ResNet18_Weights,
        ResNet50_Weights,
    )

    _HAS_TORCHVISION_WEIGHTS_ENUM = True
except Exception:
    import torchvision  # type: ignore

    _HAS_TORCHVISION_WEIGHTS_ENUM = False


class TemporalConvBlock(nn.Module):
    """
    Temporal convolutional block with residual connection and causal padding.
    Uses dilated convolutions for different receptive fields.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()

        self.use_residual = use_residual and (in_channels == out_channels)

        # Causal padding - only look at past and present
        self.pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # We'll handle padding manually for causality
        )

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=1  # Pointwise conv
        )

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Residual connection projection if needed
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T]
        """
        residual = x

        # Causal padding - pad only the left side
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))

        # First conv with dilation
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Pointwise conv
        x = self.conv2(x)
        x = self.norm2(x)

        # Residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            x = x + residual

        x = self.activation(x)
        return x


class MultiScaleTemporalCNN(nn.Module):
    """
    Multi-scale temporal CNN with different dilation rates to capture
    both short-term and long-term temporal dependencies.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Different dilation rates for multi-scale modeling
        dilations = [1, 2, 4, 8, 16][:num_layers]

        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(
            TemporalConvBlock(
                in_channels,
                hidden_channels,
                kernel_size=3,
                dilation=dilations[0],
                dropout=dropout,
                use_residual=False,
            )
        )

        # Subsequent layers
        for i in range(1, num_layers):
            self.layers.append(
                TemporalConvBlock(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    dilation=dilations[i],
                    dropout=dropout,
                    use_residual=True,
                )
            )

        # Final projection
        self.final_proj = nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.final_norm = nn.BatchNorm1d(hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T]
        """
        for layer in self.layers:
            x = layer(x)

        x = self.final_proj(x)
        x = self.final_norm(x)
        x = F.gelu(x)

        return x


class SpatialAttentionPool(nn.Module):
    """
    Spatial attention pooling for CNN features.
    Instead of simple average pooling, learn what spatial regions are important.
    """

    def __init__(self, in_channels: int):
        super().__init__()

        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid(),
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] -> [B, C]
        """
        # Compute spatial attention weights
        att_weights = self.spatial_att(x)  # [B, 1, H, W]

        # Apply attention
        x_attended = x * att_weights  # [B, C, H, W]

        # Global pooling
        x_pooled = self.global_pool(x_attended)  # [B, C, 1, 1]
        x_pooled = x_pooled.flatten(1)  # [B, C]

        return x_pooled


class TemporalCNNAnticipation(nn.Module):
    """
    Non-transformer approach using:
    1. Visual backbone with spatial attention pooling
    2. Multi-scale temporal CNN for sequence modeling
    3. Specialized heads for phase classification and anticipation

    Key advantages:
    - Faster than transformers (especially for longer sequences)
    - Naturally causal (no future information leakage)
    - Good for 1fps videos with clear temporal structure
    - Easier to interpret and debug
    """

    def __init__(
        self,
        sequence_length: int,
        num_classes: int = 6,
        time_horizon: float = 2.0,
        *,
        backbone: str = "resnet18",
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        hidden_channels: int = 256,
        num_temporal_layers: int = 5,
        dropout: float = 0.1,
        use_spatial_attention: bool = True,
    ):
        super().__init__()

        self.T = sequence_length
        self.C = num_classes
        self.H = time_horizon
        self.hidden_channels = hidden_channels

        # ---- Visual Backbone ----
        if backbone == "resnet18":
            if _HAS_TORCHVISION_WEIGHTS_ENUM:
                bb = resnet18(
                    weights=ResNet18_Weights.DEFAULT if pretrained_backbone else None
                )
            else:
                bb = torchvision.models.resnet18(pretrained=pretrained_backbone)
            feat_dim = 512
        elif backbone == "resnet50":
            if _HAS_TORCHVISION_WEIGHTS_ENUM:
                bb = resnet50(
                    weights=ResNet50_Weights.DEFAULT if pretrained_backbone else None
                )
            else:
                bb = torchvision.models.resnet50(pretrained=pretrained_backbone)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove the final FC layer and average pooling
        self.backbone_features = nn.Sequential(
            *list(bb.children())[:-2]
        )  # Remove avgpool and fc

        if freeze_backbone:
            for p in self.backbone_features.parameters():
                p.requires_grad = False

        # ---- Spatial pooling ----
        if use_spatial_attention:
            self.spatial_pool = SpatialAttentionPool(feat_dim)
        else:
            self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        # ---- Feature projection ----
        self.feature_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- Temporal CNN ----
        self.temporal_cnn = MultiScaleTemporalCNN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_temporal_layers,
            dropout=dropout,
        )

        # ---- Phase classification head ----
        # Use only the last timestep for current phase
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

        # ---- Anticipation head ----
        # Use all timesteps with attention pooling
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_channels, num_heads=8, dropout=dropout, batch_first=True
        )

        self.anticipation_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
            nn.Softplus(beta=1.0),  # Ensure positive outputs
        )

        # ---- Learnable query for anticipation ----
        self.anticipation_query = nn.Parameter(
            torch.randn(1, 1, hidden_channels) * 0.02
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights properly"""
        for module in [self.feature_proj, self.phase_head, self.anticipation_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        frames: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None,
        return_aux: bool = False,
    ) -> (
        Tuple[torch.Tensor, torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]
    ):
        """
        Args:
            frames: [B, T, 3, H, W] - video frames
            meta: metadata dict (unused but kept for API compatibility)
            return_aux: whether to return auxiliary outputs

        Returns:
            reg: [B, C] - time to next phase for each class (anticipation)
            logits: [B, C] - phase classification logits
            aux: (optional) auxiliary outputs for debugging
        """
        B, T, C_in, H, W = frames.shape
        assert T == self.T, f"Expected sequence length {self.T}, got {T}"

        # ---- Extract visual features ----
        x = frames.view(B * T, C_in, H, W)  # [B*T, 3, H, W]

        with torch.set_grad_enabled(self.backbone_features.training):
            # Get spatial feature maps
            spatial_features = self.backbone_features(x)  # [B*T, feat_dim, h, w]

        # Apply spatial pooling
        if hasattr(self.spatial_pool, "forward"):
            visual_features = self.spatial_pool(spatial_features)  # [B*T, feat_dim]
        else:
            visual_features = self.spatial_pool(spatial_features).flatten(
                1
            )  # [B*T, feat_dim]

        # Reshape back to sequence
        visual_features = visual_features.view(B, T, -1)  # [B, T, feat_dim]

        # Project to hidden space
        features = []
        for t in range(T):
            feat_t = self.feature_proj(visual_features[:, t])  # [B, hidden_channels]
            features.append(feat_t)

        features = torch.stack(features, dim=2)  # [B, hidden_channels, T]

        # ---- Temporal modeling ----
        temporal_features = self.temporal_cnn(features)  # [B, hidden_channels, T]

        # Convert back to [B, T, hidden_channels] for further processing
        temporal_features_seq = temporal_features.transpose(
            1, 2
        )  # [B, T, hidden_channels]

        # ---- Phase classification (current state) ----
        current_features = temporal_features_seq[
            :, -1
        ]  # [B, hidden_channels] - last timestep
        phase_logits = self.phase_head(current_features)  # [B, num_classes]

        # ---- Anticipation prediction ----
        # Use attention pooling over the entire sequence
        query = self.anticipation_query.expand(B, -1, -1)  # [B, 1, hidden_channels]

        attended_features, _ = self.temporal_attention(
            query=query, key=temporal_features_seq, value=temporal_features_seq
        )  # [B, 1, hidden_channels]

        attended_features = attended_features.squeeze(1)  # [B, hidden_channels]
        anticipation_raw = self.anticipation_head(attended_features)  # [B, num_classes]

        # Apply constraints
        anticipation = torch.clamp(anticipation_raw, 0.0, self.H)

        # Row-min constraint (at least one phase at time 0)
        anticipation = anticipation - anticipation.min(dim=1, keepdim=True)[0]
        anticipation = torch.clamp(anticipation, 0.0, self.H)

        if return_aux:
            aux = {
                "current_features": current_features.detach(),
                "attended_features": attended_features.detach(),
                "temporal_features_mean": temporal_features_seq.detach().mean(dim=1),
                "anticipation_raw": anticipation_raw.detach(),
            }
            return anticipation, phase_logits, aux

        return anticipation, phase_logits
