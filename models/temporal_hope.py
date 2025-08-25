"""
Multi-Task Seq Regressor: [B, C] regression in [0, time_horizon] + [B, C] classification logits
- Stable baseline that converges quickly.
- Per-frame pretrained CNN features (ResNet-18 by default).
- BiGRU temporal encoder + robust fusion (mean/max/last).
- Two heads:
    * reg_head -> bounded [0, time_horizon] (scaled sigmoid)
    * cls_head -> raw logits for classification (use CE or BCEWithLogits as needed)

Init signature includes: sequence_length: int, num_classes: int = 7, time_horizon: int = 5
Everything else has sane defaults.

Demo:
- Runs on CPU with synthetic data.
- Optimizes combined loss = SmoothL1(reg) + λ * CrossEntropy(class)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Torchvision backbone (lightweight + common) ----
try:
    import torchvision
    from torchvision.models import resnet18, ResNet18_Weights
    _HAS_TORCHVISION_WEIGHTS_ENUM = True
except Exception:
    import torchvision  # type: ignore
    _HAS_TORCHVISION_WEIGHTS_ENUM = False


class GRUSeqRegressorClassifier(nn.Module):
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
        gru_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.time_horizon = float(time_horizon)

        # --- 1) Visual backbone -> per-frame feature vector ---
        if backbone_name == "resnet18":
            if _HAS_TORCHVISION_WEIGHTS_ENUM:
                backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained_backbone else None)
            else:
                backbone = torchvision.models.resnet18(pretrained=pretrained_backbone)  # type: ignore
            feat_dim = 512
        elif backbone_name == "resnet50":
            if hasattr(torchvision.models, "resnet50"):
                if _HAS_TORCHVISION_WEIGHTS_ENUM:
                    backbone = torchvision.models.resnet50(weights="IMAGENET1K_V1" if pretrained_backbone else None)  # type: ignore
                else:
                    backbone = torchvision.models.resnet50(pretrained=pretrained_backbone)  # type: ignore
                feat_dim = 2048
            else:
                raise ValueError("resnet50 not available in this torchvision build")
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        backbone.fc = nn.Identity()  # global pooled features
        self.backbone = backbone
        self.backbone_out_dim = feat_dim
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # --- 2) Project to GRU input + norm ---
        self.in_proj = nn.Linear(self.backbone_out_dim, d_model)
        self.in_norm = nn.LayerNorm(d_model)

        # --- 3) Temporal encoder: BiGRU (stable & fast to train) ---
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=gru_layers,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,  # [B, T, D]
        )

        # --- 4) Fusion head backbone ---
        fused_dim = 6 * d_model  # mean(2d) + max(2d) + last_hidden(2d)
        self.fuse_norm = nn.LayerNorm(fused_dim)
        self.fuse_dropout = nn.Dropout(dropout)
        self.shared = nn.Sequential(
            nn.Linear(fused_dim, 2 * d_model),
            nn.GELU(),
        )

        # --- 5a) Regression head -> [0, time_horizon] ---
        self.reg_head = nn.Linear(2 * d_model, num_classes)

        # --- 5b) Classification head -> raw logits [B, C] ---
        self.cls_head = nn.Linear(2 * d_model, num_classes)

        # Init
        nn.init.xavier_uniform_(self.in_proj.weight); nn.init.zeros_(self.in_proj.bias)
        for mod in (self.shared, self.reg_head, self.cls_head):
            for m in (mod.modules() if isinstance(mod, nn.Sequential) else [mod]):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, 3, 224, 224]
        returns:
            reg: [B, C] in [0, time_horizon]
            logits: [B, C] (classification logits)
        """
        B, T, C, H, W = x.shape
        assert T == self.sequence_length, f"Expected sequence_length={self.sequence_length}, got T={T}"

        # --- Tokenization/patching: per-frame CNN features (robust visual features) ---
        x_flat = x.view(B * T, C, H, W)                         # [B*T, 3, 224, 224]
        frame_feats = self.backbone(x_flat)                     # [B*T, F], global pooled
        frame_feats = frame_feats.view(B, T, self.backbone_out_dim)

        # --- Positional projection before GRU ---
        feats = self.in_proj(frame_feats)                       # [B, T, D]
        feats = self.in_norm(feats)                             # [B, T, D]

        # --- Temporal modeling (bidirectional; no causality to maximize context) ---
        seq_out, h_n = self.gru(feats)                          # seq_out: [B, T, 2D], h_n: [2*L, B, D]

        # --- Robust sequence fusion ---
        mean_pool = seq_out.mean(dim=1)                         # [B, 2D]  -- global average
        max_pool, _ = seq_out.max(dim=1)                        # [B, 2D]  -- salient spikes
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)     # [B, 2D]  -- last states (both directions)
        fused = torch.cat([mean_pool, max_pool, last_hidden], dim=-1)  # [B, 6D]

        z = self.fuse_dropout(self.fuse_norm(fused))            # normalize & regularize
        z = self.shared(z)                                      # [B, 2D]

        # --- Heads ---
        reg_raw = self.reg_head(z)                              # [B, C]
        reg = torch.sigmoid(reg_raw) * self.time_horizon        # bound to [0, time_horizon]
        logits = self.cls_head(z)                               # [B, C] (raw logits)
        return reg, logits


# ---------------------- Minimal training demo (CPU) ----------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cpu")

    # Config
    BATCH_SIZE = 4
    T = 4                  # sequence length
    C = 7                  # num classes
    H = W = 224
    TIME_HOR = 5.0
    STEPS = 40
    LAMBDA_CLS = 0.5       # weight for classification loss

    # Build model (kept fully local for demo; set pretrained_backbone=True in real training)
    model = GRUSeqRegressorClassifier(
        sequence_length=T,
        num_classes=C,
        time_horizon=TIME_HOR,
        backbone_name="resnet18",
        pretrained_backbone=False,  # set True on your training machine to load ImageNet weights
        freeze_backbone=False,      # unfreeze to let it learn in this tiny demo
        d_model=256,
        gru_layers=1,
        dropout=0.1,
    ).to(device)
    model.train()

    # Optimizer & losses
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    reg_criterion = nn.SmoothL1Loss(reduction="mean")
    ce_criterion = nn.CrossEntropyLoss()   # multi-class; for multi-label use BCEWithLogitsLoss on logits

    # ---- Synthetic supervised task that's learnable ----
    # Use simple image statistics to generate regression targets in [0,TIME_HOR] and a class label.
    a = torch.randn(C) * 0.5
    b = torch.randn(C) * 0.5

    def make_batch(bs):
        x = torch.randn(bs, T, 3, H, W)
        with torch.no_grad():
            mean_over_time = x.mean(dim=(2, 3, 4))             # [B, T]
            max_over_time = x.amax(dim=(2, 3, 4))              # [B, T]
            s1 = mean_over_time.mean(dim=1)                    # [B]
            s2 = max_over_time.max(dim=1).values               # [B]
            # Per-class regression targets
            logits_reg = s1.unsqueeze(1) * a + s2.unsqueeze(1) * b   # [B, C]
            y_reg = torch.sigmoid(logits_reg) * TIME_HOR             # [B, C] in [0, T]
            # Classification target = argmax of (noisy) underlying scores
            class_scores = logits_reg + 0.1 * torch.randn_like(logits_reg)
            y_cls = class_scores.argmax(dim=1)                        # [B] class indices
        return x, y_reg, y_cls

    # Train a few steps
    for step in range(1, STEPS + 1):
        x, y_reg, y_cls = make_batch(BATCH_SIZE)
        x, y_reg, y_cls = x.to(device), y_reg.to(device), y_cls.to(device)

        optimizer.zero_grad()
        pred_reg, logits = model(x)
        loss_reg = reg_criterion(pred_reg, y_reg)
        loss_cls = ce_criterion(logits, y_cls)
        loss = loss_reg + LAMBDA_CLS * loss_cls
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 1 or step == STEPS:
            with torch.no_grad():
                acc = (logits.argmax(dim=1) == y_cls).float().mean()
            print(f"Step {step:02d} | total={loss.item():.4f} | reg={loss_reg.item():.4f} | cls={loss_cls.item():.4f} | acc={acc.item():.3f}")

    # Quick eval
    model.eval()
    x_test, y_reg_t, y_cls_t = make_batch(2)
    with torch.no_grad():
        pred_reg_t, logits_t = model(x_test.to(device))
    print("Pred reg shape:", tuple(pred_reg_t.shape))   # (2, C)
    print("Pred cls shape:", tuple(logits_t.shape))     # (2, C)
    # bounds check
    assert (pred_reg_t >= -1e-5).all() and (pred_reg_t <= TIME_HOR + 1e-5).all()
    print("Demo complete.")
