# models/temporal_stateless_tformer_mstcn.py

from typing import Tuple, Dict, Any, Optional, Literal, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- backbone (torchvision) ----
try:
    import torchvision
    from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
    _HAS_TORCHVISION_WEIGHTS_ENUM = True
except Exception:
    import torchvision  # type: ignore
    _HAS_TORCHVISION_WEIGHTS_ENUM = False


# ---------- helpers ----------
def rowmin_zero_hard(y: torch.Tensor, H: float) -> torch.Tensor:
    # ensure ≥ 1 zero per row, clamp to [0, H]
    return torch.clamp(y - y.min(dim=1, keepdim=True).values, 0.0, H)


class LayerNorm1d(nn.Module):
    """LayerNorm over channel dim for 1D sequences (B, C, T)."""
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x:[B,C,T]
        x = x.transpose(1, 2)     # [B,T,C]
        x = self.ln(x)
        return x.transpose(1, 2)  # [B,C,T]


class TemporalConvBlock(nn.Module):
    """
    Dilated temporal conv with residual:
      Conv1d(C,C, ks=3, dilation=d, padding=d) -> GELU -> Dropout -> Conv1d -> GELU -> Dropout -> +res -> Norm
    """
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = LayerNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x:[B,C,T]
        res = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.drop(x)
        x = x + res
        x = self.norm(x)
        return x


class MSTCNStage(nn.Module):
    """One stage of MS-TCN with a list of dilations, all channel-preserving."""
    def __init__(self, channels: int, dilations: List[int], dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([TemporalConvBlock(channels, d, dropout) for d in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,C,T]
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------- model ----------
class WindowMemoryTransformer(nn.Module):
    """
    Stateless hybrid:
      - Visual backbone per frame -> proj to D
      - MS-TCN (2 stages, dilations [1,2,4,8]) over time
      - [CLS] token + TransformerEncoder (global)
      - TransformerDecoder with C class queries
      - Heads: cls on [CLS], reg on per-class decoded vectors (+ hard row-min prior)
    """
    def __init__(
        self,
        sequence_length: int,
        num_classes: int = 6,
        time_horizon: float = 2.0,
        *,
        backbone: str = "resnet50",            # "resnet18" | "resnet50"
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
        d_model: int = 384,
        nhead: int = 6,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 1536,
        dropout: float = 0.10,
        tcn_stages: int = 2,
        tcn_dilations: Optional[List[int]] = None,  # default [1,2,4,8]
        reg_activation: Literal["softplus", "linear"] = "softplus",
        prior_mode: Literal["hard", "none"] = "hard",
    ):
        super().__init__()
        assert d_model % nhead == 0
        self.T = int(sequence_length)
        self.C = int(num_classes)
        self.H = float(time_horizon)
        self.prior_mode = prior_mode

        # ---- Backbone ----
        if backbone == "resnet18":
            if _HAS_TORCHVISION_WEIGHTS_ENUM:
                bb = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained_backbone else None)
            else:
                bb = torchvision.models.resnet18(pretrained=pretrained_backbone)  # type: ignore
            feat_dim = 512
        elif backbone == "resnet50":
            if _HAS_TORCHVISION_WEIGHTS_ENUM:
                bb = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained_backbone else None)
            else:
                bb = torchvision.models.resnet50(pretrained=pretrained_backbone)  # type: ignore
            feat_dim = 2048
        else:
            raise ValueError("Unsupported backbone")

        # global pooling head replaced by Identity
        bb.fc = nn.Identity()
        self.backbone = bb
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ---- Visual projection to D ----
        self.feat_proj = nn.Linear(feat_dim, d_model)
        self.feat_norm = nn.LayerNorm(d_model)

        # ---- MS-TCN over time (robust local structure) ----
        if tcn_dilations is None:
            tcn_dilations = [1, 2, 4, 8]
        self.to_tcn = nn.Linear(d_model, d_model)
        self.tcn_stages = nn.ModuleList(
            [MSTCNStage(d_model, tcn_dilations, dropout=dropout) for _ in range(tcn_stages)]
        )
        self.from_tcn = nn.Linear(d_model, d_model)

        # ---- Transformer Encoder with [CLS] token (global) ----
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_emb = nn.Embedding(self.T + 1, d_model)  # +1 for CLS
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # ---- Class-query Transformer Decoder (per-class aggregation) ----
        self.class_queries = nn.Parameter(torch.randn(self.C, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # ---- Heads ----
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.C),
        )
        if reg_activation == "softplus":
            self.reg_head = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Softplus(beta=1.0),
            )
        else:
            self.reg_head = nn.Linear(d_model, 1)

        # ---- init ----
        nn.init.xavier_uniform_(self.feat_proj.weight); nn.init.zeros_(self.feat_proj.bias)
        nn.init.xavier_uniform_(self.to_tcn.weight);    nn.init.zeros_(self.to_tcn.bias)
        nn.init.xavier_uniform_(self.from_tcn.weight);  nn.init.zeros_(self.from_tcn.bias)
        for m in self.cls_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        if isinstance(self.reg_head, nn.Linear):
            nn.init.xavier_uniform_(self.reg_head.weight); nn.init.zeros_(self.reg_head.bias)

    # ---- core forward ----
    def forward(self, frames: torch.Tensor, meta: Optional[Dict[str, Any]] = None, return_aux: bool = False
               ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        frames: [B, T, 3, H, W]
        meta:   unused (kept for API compatibility with your trainer)
        returns:
          reg   : [B, C]  (clamped to [0, H], row-min prior if enabled)
          logits: [B, C]
          aux   : (optional) dict with debug tensors
        """
        B, T, C3, H, W = frames.shape
        assert T == self.T, f"Expected T={self.T}, got {T}"

        # -- 1) Tokenization/patching: 2D CNN per frame --
        x = frames.view(B * T, C3, H, W)
        with torch.set_grad_enabled(self.backbone.training):
            feats = self.backbone(x)              # [B*T, F]
        feats = feats.view(B, T, -1)
        feats = self.feat_proj(feats)             # [B,T,D]
        feats = self.feat_norm(feats)             # normalization before temporal modules

        # -- 2) Local temporal modeling: MS-TCN (dilated 1D convs) --
        z = self.to_tcn(feats)                    # [B,T,D]
        z = z.transpose(1, 2)                     # [B,D,T] for Conv1d
        for stage in self.tcn_stages:
            z = stage(z)                          # residual temporal refinement
        z = z.transpose(1, 2)                     # [B,T,D]
        z = self.from_tcn(z)                      # back to token space

        # -- 3) Positional encodings (+ [CLS]) for global attention --
        cls = self.cls_token.expand(-1, B, -1)    # [1,B,D]
        pos = self.pos_emb(torch.arange(T + 1, device=z.device))  # [T+1,D]
        src = torch.cat([cls.transpose(0, 1), z], dim=1)          # [B,T+1,D]
        src = src.transpose(0, 1)                                  # [T+1,B,D]
        src = src + pos.unsqueeze(1)                               # add PEs

        # -- 4) Transformer Encoder (global, acausal) --
        memory = self.encoder(src)                                 # [T+1,B,D]
        cls_repr = memory[0]                                       # [B,D]
        mem_no_cls = memory[1:]                                    # [T,B,D]

        # -- 5) Transformer Decoder with class queries (per-class attention over time) --
        tgt = self.class_queries.unsqueeze(1).expand(self.C, B, -1).contiguous()  # [C,B,D]
        percls = self.decoder(tgt, mem_no_cls).transpose(0, 1)                    # [B,C,D]

        # -- 6) Heads --
        # Classification head (phase): from [CLS] representation
        logits = self.cls_head(cls_repr)                           # [B,C]

        # Regression head (anticipation): per-class vectors -> distances
        dist = self.reg_head(percls).squeeze(-1)                   # [B,C], >=0 if softplus

        # -- 7) Prior / clamping (keep outputs in [0, H]) --
        if self.prior_mode == "hard":
            reg = rowmin_zero_hard(dist, self.H)                   # ensure ≥1 zero per row
        else:
            reg = torch.clamp(dist, 0.0, self.H)

        if return_aux:
            aux = {
                "cls_repr": cls_repr.detach(),
                "percls_mean": percls.detach().mean(dim=1),
            }
            return reg, logits, aux
        return reg, logits


# ----------------- minimal runnable demo -----------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cpu")

    B, T, C, H, W = 2, 16, 3, 224, 224
    num_classes = 6
    model = WindowMemoryTransformer(
        sequence_length=T,
        num_classes=num_classes,
        time_horizon=2.0,
        backbone="resnet18",
        pretrained_backbone=False,   # True in real training
        freeze_backbone=False,
        d_model=256,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=1,
        dim_feedforward=512,
        dropout=0.1,
        tcn_stages=2,
        prior_mode="hard",
    ).to(device).train()

    x = torch.randn(B, T, C, H, W, device=device)
    meta = {
        "video_name": ["video01"] * B,
        "frames_indexes": torch.stack([torch.arange(0, T), torch.arange(T, 2*T)], dim=0),
        "phase_label": torch.randint(0, num_classes, (B,)),
        "time_to_next_phase": torch.rand(B, num_classes),
    }
    reg, logits = model(x, meta)
    print("reg:", reg.shape, reg.min().item(), reg.max().item())
    print("logits:", logits.shape)
    # quick loss/opt test
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    reg_tgt = torch.rand(B, num_classes)
    cls_tgt = torch.randint(0, num_classes, (B,))
    L = F.smooth_l1_loss(reg, reg_tgt) + F.cross_entropy(logits, cls_tgt)
    L.backward(); opt.step()
    print("demo OK")
