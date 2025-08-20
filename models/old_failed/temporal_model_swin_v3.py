import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.swin_transformer import Swin_T_Weights


# -------------------------------------------------------------------------
#  1)  Same Swin-T backbone you already wrote
# -------------------------------------------------------------------------
class AdaptiveSwinTransformer(nn.Module):
    """Swin-T that accepts an arbitrary number of input channels."""
    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        self.in_channels = in_channels

        # 1-A  Load the backbone
        self.swin = models.swin_t(
            weights=Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # 1-B  Adapt the *very first* conv if #channels ≠ 3
        if in_channels != 3:
            old_conv = self.swin.features[0][0]                       # Conv2d(3,96,…)
            new_conv = nn.Conv2d(
                in_channels,                                           # new C_in
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

            with torch.no_grad():
                if in_channels < 3:
                    # take only the first k channels of RGB weights
                    new_conv.weight.copy_(old_conv.weight[:, :in_channels])
                else:
                    # copy RGB weights, then replicate their mean for extra chans
                    new_conv.weight[:, :3].copy_(old_conv.weight)      # RGB
                    extra = in_channels - 3
                    mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:].copy_(mean_rgb.repeat(1, extra, 1, 1))
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)

            self.swin.features[0][0] = new_conv

        # 1-C  Strip the classification head
        self.swin.head = nn.Identity()
        self.output_dim = 768    # Swin-T embed size

    def forward(self, x):
        return self.swin(x)      # [B, 768]
        

# -------------------------------------------------------------------------
#  2)  Temporal model with a transformer *decoder* on the image embeddings
# -------------------------------------------------------------------------
class TemporalAnticipationModel(nn.Module):
    """
    Input  : a clip   [B, T, C, H, W]
    Output : per-sample regression   [B, num_classes]
    """
    def __init__(
        self,
        sequence_length: int,
        num_classes: int = 7,
        time_horizon: int = 5,
        in_channels: int = 3,
        num_decoder_layers: int = 4,
        num_decoder_heads: int = 8,
        num_queries: int = 1,          # 1 token → one global regression vector
    ):
        super().__init__()

        # 2-A  Visual backbone -------------------------------------------------
        self.sequence_length = sequence_length
        self.backbone = AdaptiveSwinTransformer(in_channels=in_channels, pretrained=True)
        self.embed_dim = self.backbone.output_dim                      # 768

        # 2-B  Positional embeddings for the frame sequence ---------------
        self.pos_embed = nn.Parameter(torch.randn(1, sequence_length, self.embed_dim))

        # 2-C  Query embeddings for the decoder ---------------------------
        #      Shape: [num_queries, embed_dim]
        self.query_embed = nn.Parameter(torch.randn(num_queries, self.embed_dim))

        # 2-D  Transformer decoder ----------------------------------------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=num_decoder_heads,
            dim_feedforward=4 * self.embed_dim,
            dropout=0.1,
            batch_first=False,          # decoder expects [T,B,E]
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 2-E  Final regression head  -------------------------------------
        self.regressor = nn.Sequential(
            nn.Linear(self.embed_dim * num_queries, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        self.time_horizon = time_horizon

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        x : [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape
        assert T == self.sequence_length, f"Model was built for T={self.sequence_length}, got {T}"
        # -----------------------------------------------------------------
        # 1)  Backbone over each frame  →  [B*T, C, H, W] → [B*T, E]
        x = x.reshape(B * T, C, H, W)
        feats = self.backbone(x)                            # [B*T, 768]

        # 2)  Restore temporal dim     →  [B, T, E]
        feats = feats.view(B, T, self.embed_dim)
        feats = feats + self.pos_embed[:, :T]               # add positions

        # 3)  Prepare inputs for decoder -----------------------------
        memory = feats.permute(1, 0, 2)                     # [T, B, E]
        tgt    = self.query_embed.unsqueeze(1).repeat(1, B, 1)  # [Q, B, E]

        # 4)  Transformer decoder  -------------------------------
        decoded = self.decoder(tgt, memory)                 # [Q, B, E]
        decoded = decoded.permute(1, 0, 2).contiguous()     # [B, Q, E]
        decoded = decoded.view(B, -1)                       # [B, Q*E]

        # 5)  Regression head  -----------------------------------
        out = self.regressor(decoded)                       # [B, num_classes]
        out = torch.sigmoid(out) * self.time_horizon        # clamp to [0, H]

        return out

def create_model(
    sequence_length: int,
    num_classes: int = 7,
    time_horizon: int = 5,
    in_channels: int = 3,
    num_decoder_layers: int = 4,
    num_decoder_heads: int = 8,
    num_queries: int = 1,
):
    """
    Factory function to create the TemporalAnticipationModel.
    """
    return TemporalAnticipationModel(
        sequence_length=sequence_length,
        num_classes=num_classes,
        time_horizon=time_horizon,
        in_channels=in_channels,
        num_decoder_layers=num_decoder_layers,
        num_decoder_heads=num_decoder_heads,
        num_queries=num_queries,
    )

# ---------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------
def _quick_test():
    """
    Run a forward pass to be sure everything is wired correctly.
    – Works for both RGB (C=3) and RGB-D (C=4).
    – Prints tensor shapes, run time, and #parameters.
    """
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------ choose your configuration -------------------
    B, T, C, H, W = 2, 6, 4, 224, 224      # try C=3 for RGB only
    num_classes   = 7
    time_horizon  = 5

    # ------------ build the model -----------------------------
    model = TemporalAnticipationModel(
        sequence_length=T,
        num_classes=num_classes,
        time_horizon=time_horizon,
        in_channels=C,            # 3 = RGB   | 4 = RGB-D
    ).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params/1e6:.2f} M")

    # ------------ dummy input ---------------------------------
    x = torch.randn(B, T, C, H, W, device=device)

    # ------------ warm-up & timed pass ------------------------
    _ = model(x)                      # warm-up (caches, JIT, etc.)
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()
    out = model(x)
    torch.cuda.synchronize() if device.type == "cuda" else None
    dur = time.time() - start

    # ------------ report --------------------------------------
    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Forward pass : {dur*1000:.1f} ms")

if __name__ == "__main__":
    _quick_test()
