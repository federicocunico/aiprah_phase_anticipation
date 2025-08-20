"""
Causal Visual-Transformer Decoder Regressor (PyTorch, CPU-ready)
- Input:  sequence of T RGB frames, each 224x224 (sampled at 1s cadence).
- Output: [B, C] real-valued predictions, one scalar per class in [0, time_horizon].

Design (succinct justification):
1) Robust visual features: each frame is encoded by a pretrained CNN (ResNet-50, ImageNet)—strong inductive bias for local textures and shapes.
2) Temporal modeling with CAUSAL attention: frame features pass through a Transformer ENCODER with a strict causal mask (no token sees the future).
   This enforces online/streaming correctness and prevents information leakage (no access to frames at t' > t).
3) Transformer DECODER with learned class queries attends to the (causally-encoded) memory to produce per-class representations,
   which a small head maps to bounded [0, time_horizon] via scaled sigmoid.
   Using a decoder lets each class learn its own "what matters in the past" attention over time.

Notes:
- Uses torchvision pretrained weights (robust and widely available). If ResNet50_Weights isn't present (older torchvision), falls back to pretrained=True.
- Everything runs on CPU. Forward has concise inline comments explaining each step.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    import torchvision
    from torchvision.models import resnet50, ResNet50_Weights
    _HAS_TORCHVISION_WEIGHTS_ENUM = True
except Exception:
    import torchvision  # type: ignore
    _HAS_TORCHVISION_WEIGHTS_ENUM = False


class SinusoidalPositionalEncoding(nn.Module):
    """Standard transformer sinusoidal PE (sequence-first shape: [S, B, D])."""
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [S, B, D]
        S, B, D = x.shape
        return x + self.pe[:S].unsqueeze(1)  # broadcast over batch


def generate_causal_mask(S: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular mask with -inf above diagonal for causal self-attention."""
    # shape [S, S], True or -inf where masked depending on API; we use additive mask with -inf for attention scores.
    mask = torch.full((S, S), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)  # keep 0 on diagonal and below, -inf above
    return mask


class CausalVideoDecoderRegressor(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        num_classes: int = 7,
        time_horizon: int = 5,
        *,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.time_horizon = float(time_horizon)

        # --- Visual backbone (pretrained) ---
        # We use ResNet-50 to extract a per-frame global feature vector (robust, widely available).
        if _HAS_TORCHVISION_WEIGHTS_ENUM:
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained_backbone else None)
        else:
            backbone = torchvision.models.resnet50(pretrained=pretrained_backbone)  # fallback for older torchvision
        backbone.fc = nn.Identity()  # return pooled 2048-dim features
        self.backbone = backbone
        self.backbone_out_dim = 2048
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # --- Project visual features to transformer model dimension ---
        self.feat_proj = nn.Linear(self.backbone_out_dim, d_model)
        self.feat_norm = nn.LayerNorm(d_model)

        # --- Positional encoding across time steps (1s cadence is implicit in positions) ---
        self.pos_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max(1024, sequence_length + 1))

        # --- Temporal causal encoder over frame features ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # --- Transformer decoder with learned class queries ---
        self.class_queries = nn.Parameter(torch.randn(num_classes, d_model) * 0.02)  # [C, D]
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # --- Output head: per-class scalar, bounded to [0, time_horizon] via scaled sigmoid ---
        self.out_head = nn.Linear(d_model, 1)

        # Init a bit more carefully
        nn.init.xavier_uniform_(self.feat_proj.weight)
        nn.init.zeros_(self.feat_proj.bias)
        nn.init.xavier_uniform_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, 3, 224, 224]
        returns: [B, C] in [0, time_horizon]
        """
        B, T, C, H, W = x.shape
        assert T == self.sequence_length, f"Expected sequence_length={self.sequence_length}, got T={T}"
        device = x.device

        # --- 1) Tokenization/patching (visual feature extraction per frame) ---
        # Flatten time into batch, run pretrained backbone, get a single robust feature per frame.
        x_flat = x.view(B * T, C, H, W)  # [B*T, 3, 224, 224]
        with torch.set_grad_enabled(self.backbone.training):
            frame_feats = self.backbone(x_flat)  # [B*T, 2048], already globally pooled
        frame_feats = frame_feats.view(B, T, self.backbone_out_dim)

        # --- 2) Project to transformer dimension and normalize ---
        feats = self.feat_proj(frame_feats)               # [B, T, D]
        feats = self.feat_norm(feats)                     # [B, T, D]

        # --- 3) Prepare for transformer (sequence-first) and add positional encodings across time ---
        src = feats.transpose(0, 1).contiguous()          # [T, B, D]
        src = self.pos_encoding(src)                      # [T, B, D] + PE(time)

        # --- 4) Causality: apply a strict subsequent mask so each time step sees only <= current time ---
        # This prevents information leakage from future frames (theoretical requirement for online/forecasting).
        src_mask = generate_causal_mask(T, device=device)  # [T, T] with -inf above diagonal

        # --- 5) Temporal encoding with causal self-attention ---
        memory = self.encoder(src, mask=src_mask)         # [T, B, D]; memory contains only past information per t

        # --- 6) Transformer decoder with class queries (no temporal dimension here) ---
        # Create target as learned class tokens (per class) broadcast to batch.
        tgt = self.class_queries.unsqueeze(1).expand(self.num_classes, B, -1).contiguous()  # [C, B, D]

        # Decoder cross-attends to the (causal) encoded memory; no need for extra masks here since memory is already causal.
        dec_out = self.decoder(tgt=tgt, memory=memory)    # [C, B, D]

        # --- 7) Head: map each class query to a scalar, then bound to [0, time_horizon] ---
        dec_out = dec_out.transpose(0, 1)                 # [B, C, D]
        raw = self.out_head(dec_out).squeeze(-1)          # [B, C]
        y = torch.sigmoid(raw) * self.time_horizon        # [B, C] in [0, time_horizon]
        return y


# ------------------------------ Minimal runnable demo (CPU) ------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cpu")

    # Config
    B = 2
    T = 4  # example sequence length (set this to your desired N)
    C = 7
    H = W = 224
    time_horizon = 5

    # Instantiate model
    model = CausalVideoDecoderRegressor(
        sequence_length=T,
        num_classes=C,
        time_horizon=time_horizon,
        d_model=512,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        freeze_backbone=True,         # freeze for the demo; unfreeze for finetuning
        pretrained_backbone=True,
    ).to(device)
    model.eval()

    # Dummy input (simulates T RGB frames sampled at 1s)
    x = torch.randn(B, T, 3, H, W, device=device)

    # Forward pass
    with torch.no_grad():
        y = model(x)

    # Sanity checks
    print("Output shape:", tuple(y.shape))            # expect (B, C)
    print("Output sample:\n", y)
    print("Min/Max per-batch:", y.amin(dim=1), y.amax(dim=1))
    # Assertions (allow tiny numerical epsilon)
    assert y.shape == (B, C)
    assert (y >= -1e-5).all() and (y <= time_horizon + 1e-5).all(), "Outputs not bounded to [0, time_horizon]"
    print("All checks passed.")
