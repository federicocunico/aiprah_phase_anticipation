from typing import Any, Optional, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# ---- backbone (torchvision) ----
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


# -----------------------------
# Building blocks
# -----------------------------
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
        self.pad = (kernel_size - 1) * dilation  # causal left padding

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # manual pad for causality
        )
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T]
        """
        residual = x
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))  # (left, right)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)

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

        dilations = [2**i for i in range(num_layers)]  # [1,2,4,8,...]

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
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = self.spatial_att(x)  # [B,1,H,W]
        x = x * att
        x = self.global_pool(x).flatten(1)  # [B,C]
        return x


class GatedFusion(nn.Module):
    """
    Gated lateral fusion for two temporal feature streams aligned in time.
    out = σ(Conv1d([A;B])) * A + (1-σ(...)) * B
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Conv1d(2 * channels, channels, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)

        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a, b], dim=1)  # [B,2C,T]
        g = torch.sigmoid(self.gate(x))  # [B,C,T]
        y = g * a + (1.0 - g) * b
        y = self.proj(y)
        y = self.norm(y)
        return F.gelu(y)


# -----------------------------
# Causal Conformer-style temporal encoder
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        attn_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(1)
        residual = x
        x = self.ln(x)
        out, _ = self.mha(x, x, x, attn_mask=attn_mask, need_weights=False)
        out = self.dropout(out)
        return residual + out


class ConvModule(nn.Module):
    """
    Conformer convolution module (temporal).
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.dw = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
            dilation=dilation,
            padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.ln = nn.LayerNorm(d_model)

        nn.init.xavier_uniform_(self.pw1.weight)
        nn.init.zeros_(self.pw1.bias)
        nn.init.xavier_uniform_(self.dw.weight)
        nn.init.xavier_uniform_(self.pw2.weight)
        nn.init.zeros_(self.pw2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x)  # [B,T,D]
        x = x.transpose(1, 2)  # [B,D,T]

        x = self.pw1(x)  # [B,2D,T]
        a, b = x.chunk(2, dim=1)
        x = a * torch.sigmoid(b)  # [B,D,T]

        pad = (self.kernel_size - 1) * self.dilation
        if pad > 0:
            x = F.pad(x, (pad, 0))
        x = self.dw(x)
        x = self.bn(x)
        x = F.gelu(x)
        x = self.pw2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # [B,T,D]
        return residual + x


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TemporalConformerBlock(nn.Module):
    """
    FFN -> Causal Self-Attention -> Conv Module -> FFN (with residuals)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ffn1 = PositionwiseFFN(d_model, expansion=4, dropout=dropout)
        self.attn = CausalSelfAttention(d_model, num_heads=num_heads, dropout=dropout)
        self.conv = ConvModule(
            d_model, kernel_size=kernel_size, dilation=dilation, dropout=dropout
        )
        self.ffn2 = PositionwiseFFN(d_model, expansion=4, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn1(x)
        x = self.attn(x)
        x = self.conv(x)
        x = self.ffn2(x)
        return x


class TemporalConformerEncoder(nn.Module):
    """
    Stack of TemporalConformerBlock with increasing dilations for multi-scale context.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        dilations = [2**i for i in range(num_layers)]
        self.blocks = nn.ModuleList(
            [
                TemporalConformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    kernel_size=3,
                    dilation=d,
                    dropout=dropout,
                )
                for d in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# -----------------------------
# Slow-Fast Anticipation Model (differentiable constraints)
# -----------------------------
class SlowFastTemporalAnticipation(nn.Module):
    """
    Slow–Fast temporal anticipation model with completion head.

    Returns: anticipation [B,C], phase_logits [B,C], completion [B,1], {}
    """

    def __init__(
        self,
        sequence_length: int,
        num_classes: int = 6,
        time_horizon: float = 2.0,
        *,
        backbone_fast: str = "resnet18",
        backbone_slow: str = "resnet18",
        pretrained_backbone_fast: bool = True,
        pretrained_backbone_slow: bool = True,
        freeze_backbone_fast: bool = False,
        freeze_backbone_slow: bool = False,
        hidden_channels: int = 256,
        num_temporal_layers: int = 5,
        dropout: float = 0.1,
        use_spatial_attention: bool = True,
        attn_heads: int = 8,
        softmin_tau: float | None = None,
        sigmoid_scale: float = 1.0,
        floor_beta: float = 2.0,
    ):
        super().__init__()

        self.T = sequence_length
        self.C = num_classes
        self.H = time_horizon
        self.hidden_channels = hidden_channels

        self.softmin_tau = softmin_tau if softmin_tau is not None else 0.02 * self.H
        self.sigmoid_scale = sigmoid_scale
        self.floor_beta = floor_beta

        # ---- Visual backbones (independent) ----
        self.backbone_fast_features, feat_dim_fast = self._make_backbone(
            backbone_fast, pretrained_backbone_fast, freeze_backbone_fast
        )
        self.backbone_slow_features, feat_dim_slow = self._make_backbone(
            backbone_slow, pretrained_backbone_slow, freeze_backbone_slow
        )

        # ---- Spatial pooling ----
        if use_spatial_attention:
            self.spatial_pool_fast = SpatialAttentionPool(feat_dim_fast)
            self.spatial_pool_slow = SpatialAttentionPool(feat_dim_slow)
        else:
            self.spatial_pool_fast = nn.AdaptiveAvgPool2d(1)
            self.spatial_pool_slow = nn.AdaptiveAvgPool2d(1)

        # ---- Feature projections (per pathway) ----
        self.feature_proj_fast = nn.Sequential(
            nn.Linear(feat_dim_fast, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.feature_proj_slow = nn.Sequential(
            nn.Linear(feat_dim_slow, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- Pathway-specific temporal CNNs ----
        self.temporal_cnn_fast = MultiScaleTemporalCNN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_temporal_layers,
            dropout=dropout,
        )
        self.temporal_cnn_slow = MultiScaleTemporalCNN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_temporal_layers,
            dropout=dropout,
        )

        # ---- Gated fusion on the fast timeline ----
        self.fusion_gate_fast = GatedFusion(hidden_channels)

        # ---- Post-merge causal Conformer encoder ----
        self.temporal_encoder = TemporalConformerEncoder(
            d_model=hidden_channels,
            num_layers=num_temporal_layers,
            num_heads=attn_heads,
            dropout=dropout,
        )

        # ---- Heads ----
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

        # Anticipation via attention pooling
        self.anticipation_query = nn.Parameter(
            torch.randn(1, 1, hidden_channels) * 0.02
        )
        self.temporal_attention_pool = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.anticipation_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

        self.completion_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    # ----- utils -----
    def _make_backbone(self, backbone: str, pretrained: bool, freeze: bool):
        if backbone == "resnet18":
            if _HAS_TORCHVISION_WEIGHTS_ENUM:
                bb = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            else:
                bb = torchvision.models.resnet18(pretrained=pretrained)
            feat_dim = 512
        elif backbone == "resnet50":
            if _HAS_TORCHVISION_WEIGHTS_ENUM:
                bb = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            else:
                bb = torchvision.models.resnet50(pretrained=pretrained)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        features = nn.Sequential(*list(bb.children())[:-2])
        if freeze:
            for p in features.parameters():
                p.requires_grad = False
        return features, feat_dim

    def _init_weights(self):
        for module in [
            self.feature_proj_fast,
            self.feature_proj_slow,
            self.phase_head,
            self.anticipation_head,
            self.completion_head,
        ]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _pool_spatial(self, pool: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if isinstance(pool, SpatialAttentionPool):
            return pool(x)
        return pool(x).flatten(1)

    @staticmethod
    def _softmin(x: torch.Tensor, dim: int, tau: float) -> torch.Tensor:
        return -tau * torch.logsumexp(-x / tau, dim=dim, keepdim=True)

    # ----- forward -----
    def forward(
        self,
        frames: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, C_in, H, W = frames.shape
        assert T == self.T, f"Expected sequence length {self.T}, got {T}"

        # Pathway sampling
        frames_fast = frames
        idx_slow = torch.arange(0, T, 2, device=frames.device)
        frames_slow = frames.index_select(1, idx_slow)
        T_slow = frames_slow.shape[1]

        # Fast features
        xf = frames_fast.view(B * T, C_in, H, W)
        with torch.set_grad_enabled(self.backbone_fast_features.training):
            sf = self.backbone_fast_features(xf)
        vf = self._pool_spatial(self.spatial_pool_fast, sf)
        vf_seq = vf.view(B, T, -1)
        feats_fast = torch.stack(
            [self.feature_proj_fast(vf_seq[:, t]) for t in range(T)], dim=2
        )

        # Slow features
        xs = frames_slow.view(B * T_slow, C_in, H, W)
        with torch.set_grad_enabled(self.backbone_slow_features.training):
            ss = self.backbone_slow_features(xs)
        vs = self._pool_spatial(self.spatial_pool_slow, ss)
        vs_seq = vs.view(B, T_slow, -1)
        feats_slow = torch.stack(
            [self.feature_proj_slow(vs_seq[:, t]) for t in range(T_slow)], dim=2
        )

        # Temporal TCNs
        tf = self.temporal_cnn_fast(feats_fast)
        ts = self.temporal_cnn_slow(feats_slow)

        # Align + fuse
        ts_up = F.interpolate(ts, size=T, mode="linear", align_corners=False)
        fused_fast = self.fusion_gate_fast(tf, ts_up)

        # Conformer temporal encoder
        encoded = self.temporal_encoder(fused_fast.transpose(1, 2))

        # Heads
        current = encoded[:, -1, :]
        phase_logits = self.phase_head(current)
        completion = self.completion_head(current)

        # Anticipation via attention pooling
        query = self.anticipation_query.expand(B, -1, -1)
        pooled, _ = self.temporal_attention_pool(query=query, key=encoded, value=encoded)
        pooled = pooled.squeeze(1)
        raw = self.anticipation_head(pooled)

        if self.sigmoid_scale != 1.0:
            y = self.H * torch.sigmoid(self.sigmoid_scale * raw)
        else:
            y = self.H * torch.sigmoid(raw)

        m = self._softmin(y, dim=1, tau=self.softmin_tau)
        y = y - m
        anticipation = F.softplus(y, beta=self.floor_beta)

        return anticipation, phase_logits, completion, {}

