"""
Row-Min-0: Enforce "at least one zero per row" on sequence regression outputs

Context
- Model predicts per-sample per-class time-to-next-phase: y ∈ [0, H]^(B×C)
- Structural prior: in each row, at least one entry must be exactly 0.

This file shows 3 practical ways to enforce it during training:
1) Hard projection  (recommended): y <- clamp(y - min(y, dim=1), [0, H])
2) Soft projection               : y <- clamp(y - softmin_tau(y), [0, H]) with temperature tau
3) Gated-zero via classification : set the predicted active class to 0 using straight-through Gumbel-Softmax

We provide:
- A small attention-based temporal model (2D CNN backbone + Transformer encoder + class-query decoder).
- Projection utilities you can drop into your training loop or inside the model.
- A tiny CPU demo that runs end-to-end.

You can integrate method (1) directly into your model's forward(), or apply it just before loss.
"""

import math
from typing import Tuple, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Torchvision backbone (lightweight) ----
try:
    import torchvision
    from torchvision.models import resnet18, ResNet18_Weights
    _HAS_TORCHVISION_WEIGHTS_ENUM = True
except Exception:
    import torchvision  # type: ignore
    _HAS_TORCHVISION_WEIGHTS_ENUM = False


# --------------------------- Projection utilities ---------------------------

def project_row_min_zero_hard(y: torch.Tensor, H: float) -> torch.Tensor:
    """
    Deterministic projection: for each row, subtract the (hard) minimum and clamp.
    This guarantees min(row)=0. Subgradients flow to the argmin elements (piecewise differentiable).
    y: [B, C]
    """
    y_shift = y - y.min(dim=1, keepdim=True).values
    return torch.clamp(y_shift, 0.0, H)


def project_row_min_zero_soft(y: torch.Tensor, H: float, tau: float = 0.2) -> torch.Tensor:
    """
    Soft projection: subtract a temperature-controlled softmin (smooth approximation of min).
    Good if you want smoother gradients when multiple entries compete to be the minimum.
    y: [B, C]
    """
    # softmin_tau(a)_i = exp(-a_i/tau) / sum_j exp(-a_j/tau); softmin value = sum_i softmin_tau(a)_i * a_i
    weights = F.softmax(-y / max(tau, 1e-6), dim=1)        # [B, C]
    soft_min = (weights * y).sum(dim=1, keepdim=True)      # [B, 1]
    y_shift = y - soft_min
    return torch.clamp(y_shift, 0.0, H)


def gumbel_softmax_onehot(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    Straight-Through Gumbel-Softmax to sample a one-hot vector per row (argmax with gradient).
    logits: [B, C] -> returns onehot [B, C]
    """
    g = -torch.empty_like(logits).exponential_().log()  # Gumbel(0,1)
    y_soft = F.softmax((logits + g) / tau, dim=1)
    # Straight-through: make it one-hot in the forward pass, keep soft gradients in backward.
    index = y_soft.argmax(dim=1, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(1, index, 1.0)
    y_st = y_hard + (y_soft - y_soft.detach())  # stop-grad trick
    return y_st  # [B, C]


def project_row_min_zero_gated(y: torch.Tensor, cls_logits: torch.Tensor, H: float, tau: float = 1.0) -> torch.Tensor:
    """
    Gated projection using a classification head:
    - Choose the active class with straight-through Gumbel-Softmax (one-hot selector s).
    - Force that entry to zero by subtracting y * s per row.
      y' = y - (y ⊙ s)  where s is one-hot (so the chosen index becomes exactly 0).
    - Clamp to [0, H].

    This lets the classifier decide which class is "currently active" and guarantees one exact zero.

    y: [B, C], cls_logits: [B, C]
    """
    s = gumbel_softmax_onehot(cls_logits, tau=tau)  # [B, C], one-hot (straight-through)
    offset = (y * s).sum(dim=1, keepdim=True)       # [B, 1] value at the selected index
    y_shift = y - offset
    return torch.clamp(y_shift, 0.0, H)


# --------------------------- Attention-based temporal model ---------------------------

class TemporalTransformerLearnedPE(nn.Module):
    """
    2D CNN backbone -> temporal Transformer encoder -> class-query Transformer decoder -> dual heads.
    Adds an optional row-min-0 projection on regression head outputs.
    """
    def __init__(
        self,
        sequence_length: int,
        num_classes: int = 7,
        time_horizon: int = 5,
        *,
        backbone_name: str = "resnet18",     # "resnet18" or "resnet50"
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        causal: bool = False,
        prior_mode: Literal["none", "hard", "soft", "gated"] = "hard",
        soft_tau: float = 0.2,
        gated_tau: float = 1.0,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.time_horizon = float(time_horizon)
        self.causal = causal
        self.prior_mode = prior_mode
        self.soft_tau = soft_tau
        self.gated_tau = gated_tau

        # --- 1) Per-frame visual backbone (robust 2D features) ---
        if backbone_name == "resnet18":
            if _HAS_TORCHVISION_WEIGHTS_ENUM:
                backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained_backbone else None)
            else:
                backbone = torchvision.models.resnet18(pretrained=pretrained_backbone)  # type: ignore
            feat_dim = 512
        elif backbone_name == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=pretrained_backbone)  # type: ignore
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.backbone_out_dim = feat_dim
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # --- 2) Project to model width + norm ---
        self.feat_proj = nn.Linear(self.backbone_out_dim, d_model)
        self.feat_norm = nn.LayerNorm(d_model)

        # --- 3) Learned positional embedding over time ---
        self.pos_embedding = nn.Embedding(sequence_length, d_model)

        # --- 4) Temporal Transformer encoder ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # --- 5) Transformer decoder with learned per-class queries ---
        self.class_queries = nn.Parameter(torch.randn(num_classes, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # --- 6) Heads: regression (pre-projection) and classification ---
        # NOTE: We output raw nonnegative distances with softplus; projection will enforce min=0.
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softplus(beta=1.0),  # nonnegative, smooth near 0
        )
        self.cls_head = nn.Linear(d_model, 1)

        # Init
        nn.init.xavier_uniform_(self.feat_proj.weight); nn.init.zeros_(self.feat_proj.bias)
        for m in self.reg_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.cls_head.weight); nn.init.zeros_(self.cls_head.bias)

    def _causal_mask(self, S: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((S, S), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, 3, 224, 224]
        returns:
            reg:    [B, C] in [0, time_horizon], with at least one exact 0 per row (if prior_mode != "none")
            logits: [B, C] classification logits
        """
        B, T, C, H, W = x.shape
        assert T == self.sequence_length, f"Expected sequence_length={self.sequence_length}, got T={T}"
        device = x.device

        # --- Tokenization/patching via CNN backbone (robust per-frame features) ---
        x_flat = x.view(B * T, C, H, W)                 # [B*T, 3, 224, 224]
        with torch.set_grad_enabled(self.backbone.training):
            frame_feats = self.backbone(x_flat)         # [B*T, F]
        frame_feats = frame_feats.view(B, T, self.backbone_out_dim)

        # --- Project + norm ---
        feats = self.feat_proj(frame_feats)             # [B, T, D]
        feats = self.feat_norm(feats)

        # --- Add learned positional embeddings ---
        pos_ids = torch.arange(T, device=device).unsqueeze(0)   # [1, T]
        pos_emb = self.pos_embedding(pos_ids).transpose(0, 1)   # [T, 1, D]
        src = feats.transpose(0, 1).contiguous() + pos_emb      # [T, B, D]

        # --- Temporal encoder (optionally causal) ---
        enc_mask = self._causal_mask(T, device) if self.causal else None
        memory = self.encoder(src, mask=enc_mask)       # [T, B, D]

        # --- Class-query decoder ---
        tgt = self.class_queries.unsqueeze(1).expand(self.num_classes, B, -1).contiguous()  # [C, B, D]
        dec = self.decoder(tgt, memory)                 # [C, B, D]
        dec = dec.transpose(0, 1)                       # [B, C, D]

        # --- Heads: distances (pre-projection, nonnegative) + logits ---
        dist_pre = self.reg_head(dec).squeeze(-1)       # [B, C]  >= 0 (softplus)
        logits   = self.cls_head(dec).squeeze(-1)       # [B, C]

        # --- Structural prior: enforce at least one zero per row ---
        if self.prior_mode == "hard":
            reg = project_row_min_zero_hard(dist_pre, self.time_horizon)
        elif self.prior_mode == "soft":
            reg = project_row_min_zero_soft(dist_pre, self.time_horizon, tau=self.soft_tau)
        elif self.prior_mode == "gated":
            reg = project_row_min_zero_gated(dist_pre, logits, self.time_horizon, tau=self.gated_tau)
        elif self.prior_mode == "none":
            # fallback: clip to [0, H] only
            reg = torch.clamp(dist_pre, 0.0, self.time_horizon)
        else:
            raise ValueError(f"Unknown prior_mode: {self.prior_mode}")

        return reg, logits


# ------------------------------ Minimal runnable demo (CPU) ------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cpu")

    # Config
    B, T, C, H, W = 3, 8, 6, 224, 224
    TIME_H = 5.0

    # Try each prior mode
    for mode in ["none", "hard", "soft", "gated"]:
        print(f"\n== Prior mode: {mode} ==")
        model = TemporalTransformerLearnedPE(
            sequence_length=T,
            num_classes=C,
            time_horizon=TIME_H,
            backbone_name="resnet18",
            pretrained_backbone=False,  # set True in real training
            freeze_backbone=False,
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=1,
            dim_feedforward=512,
            prior_mode=mode,
            soft_tau=0.2,
            gated_tau=0.7,
        ).to(device)
        model.eval()

        x = torch.randn(B, T, 3, H, W, device=device)
        with torch.no_grad():
            reg, logits = model(x)

        # Checks
        print("reg shape:", tuple(reg.shape), "logits shape:", tuple(logits.shape))
        print("row mins (should be 0 when mode != 'none'):", reg.min(dim=1).values)
        assert reg.shape == (B, C) and logits.shape == (B, C)
        assert (reg >= -1e-5).all() and (reg <= TIME_H + 1e-5).all()
        if mode != "none":
            # exact zeros for 'hard' and 'gated'; approximately zero for 'soft'
            mins = reg.min(dim=1).values
            if mode in ("hard", "gated"):
                assert torch.allclose(mins, torch.zeros_like(mins), atol=1e-6)
        print("OK.")
