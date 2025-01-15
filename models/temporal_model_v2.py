import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

from models.non_local import TimeConv


class Identity(nn.Module):
    def forward(self, x):
        return x


# class TemporalAnticipationModel(nn.Module):
#     def __init__(self, sequence_length: int, num_classes: int = 7, time_horizon: int = 5):
#         super(TemporalAnticipationModel, self).__init__()

#         self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
#         self.backbone.heads = Identity()  # Remove classification head
#         self.sequence_length = sequence_length
#         self.num_classes = num_classes
#         self.time_horizon = time_horizon

#         # Positional encoding for temporal ordering
#         self.positional_encoding = nn.Parameter(torch.zeros(1, sequence_length, 768))

#         # Transformer encoder for temporal modeling
#         encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
#         self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

#         # Anticipation prediction
#         self.fc = nn.Linear(768, 1)

#     def forward(self, x: torch.Tensor):
#         B, T, C, H, W = x.size()

#         # Extract spatial features
#         x = x.view(B * T, C, H, W)
#         x = self.backbone.forward(x)  # [B*T, 768]
#         x = x.view(B, T, -1)  # [B, T, 768]

#         # Add positional encoding
#         x = x + self.positional_encoding

#         # Pass through temporal transformer
#         x = self.temporal_transformer(x)  # [B, T, 768]

#         # Mean pooling across time dimension to anticipate the future
#         x = x.mean(dim=1)  # [B, 768]

#         # Classification or temporal anticipation
#         out: torch.Tensor = self.fc(x)  # [B, 1]
#         out = out.squeeze(dim=1)  # [B]

#         return out

## TOO SLOW
# class TemporalAnticipationModel(nn.Module):
#     def __init__(self, sequence_length: int, num_classes: int = 7, time_horizon: int = 5):
#         super(TemporalAnticipationModel, self).__init__()

#         self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
#         self.backbone.heads = Identity()  # Remove classification head
#         self.sequence_length = sequence_length
#         self.num_classes = num_classes
#         self.time_horizon = time_horizon

#         # Cross-attention mechanism
#         self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)

#         # Transformer encoder for refined temporal modeling
#         encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
#         self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

#         # Anticipation prediction
#         self.fc_anticipation = nn.Linear(768, 1)
#         self.fc_phase = nn.Linear(768, num_classes)

#     def forward(self, x: torch.Tensor):
#         B, T, C, H, W = x.size()

#         # Extract spatial features
#         x = x.view(B * T, C, H, W)
#         spatial_features = self.backbone.forward(x)  # [B*T, 768]
#         spatial_features = spatial_features.view(B, T, -1)  # [B, T, 768]

#         # Create learnable temporal queries
#         temporal_queries = nn.Parameter(torch.zeros(1, self.sequence_length, 768)).to(x.device)

#         # Apply cross-attention: temporal queries attend to spatial features
#         temporal_output, _ = self.cross_attention(
#             query=temporal_queries.repeat(B, 1, 1),  # [B, T, 768]
#             key=spatial_features,  # [B, T, 768]
#             value=spatial_features,  # [B, T, 768]
#         )

#         # Pass through temporal transformer
#         temporal_output = self.temporal_transformer(temporal_output)  # [B, T, 768]

#         # Mean pooling across time dimension to anticipate the future
#         x = temporal_output.mean(dim=1)  # [B, 768]

#         # Classification or temporal anticipation
#         ant: torch.Tensor = self.fc_anticipation(x)  # [B, 1]
#         ant = ant.squeeze(dim=1)  # [B]
#         ant = self.time_horizon * torch.sigmoid(ant)

#         ph: torch.Tensor = self.fc_phase(x)  # [B, num_classes]
#         return ph, ant


# class TemporalAnticipationModel(nn.Module):
#     def __init__(self, sequence_length: int, num_classes: int = 7, time_horizon: int = 5):
#         super(TemporalAnticipationModel, self).__init__()

#         self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
#         self.backbone.heads = Identity()  # Remove classification head
#         self.sequence_length = sequence_length
#         self.num_classes = num_classes
#         self.time_horizon = time_horizon

#         # Dimensionality reduction for spatial features
#         self.reduce_dim = nn.Linear(768, 256)

#         # Downsample temporal tokens to reduce sequence length
#         self.temporal_pool = nn.AdaptiveAvgPool1d(output_size=self.sequence_length // 2)  # Reduce T to 16

#         # Simplified cross-attention mechanism
#         self.cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)

#         # Lightweight temporal transformer for refined temporal modeling
#         encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True)
#         self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

#         # Anticipation prediction
#         self.fc_anticipation = nn.Linear(256, num_classes)
#         self.fc_phase = nn.Linear(256, num_classes)

#         self.temporal_queries = nn.Parameter(torch.zeros(1, self.sequence_length // 2, 256))

#     def forward(self, x: torch.Tensor):
#         B, T, C, H, W = x.size()

#         # Extract spatial features
#         x = x.view(B * T, C, H, W)
#         spatial_features = self.backbone.forward(x)  # [B*T, 768]
#         spatial_features = spatial_features.view(B, T, -1)  # [B, T, 768]

#         # Reduce dimensionality
#         spatial_features = self.reduce_dim(spatial_features)  # [B, T, 256]

#         # Downsample temporal tokens
#         spatial_features = self.temporal_pool(spatial_features.transpose(1, 2))  # [B, 256, T/2]
#         spatial_features = spatial_features.transpose(1, 2)  # [B, T/2, 256]

#         # Create learnable temporal queries
#         temporal_queries = self.temporal_queries.detach().clone()

#         # Apply cross-attention: temporal queries attend to spatial features
#         temporal_output, _ = self.cross_attention(
#             query=temporal_queries.repeat(B, 1, 1),  # [B, T/2, 256]
#             key=spatial_features,  # [B, T/2, 256]
#             value=spatial_features,  # [B, T/2, 256]
#         )

#         # Pass through temporal transformer
#         temporal_output = self.temporal_transformer(temporal_output)  # [B, T/2, 256]

#         # Mean pooling across time dimension to anticipate the future
#         x = temporal_output.mean(dim=1)  # [B, 256]

#         # Classification or temporal anticipation
#         ant: torch.Tensor = self.fc_anticipation(x)  # [B, num_classes]
#         ant = self.time_horizon * torch.sigmoid(ant)

#         ph: torch.Tensor = self.fc_phase(x)  # [B, num_classes]
#         return ph, ant


import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights


# class TemporalAnticipationModel(nn.Module):
#     def __init__(self, sequence_length: int, num_classes: int = 7, time_horizon: int = 5):
#         super(TemporalAnticipationModel, self).__init__()

#         # Load ResNet-50 backbone
#         self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
#         self.backbone.fc = nn.Identity()  # Remove the fully connected classification head

#         self.sequence_length = sequence_length
#         self.num_classes = num_classes
#         self.time_horizon = time_horizon

#         # Dimensionality reduction for spatial features
#         self.reduce_dim = nn.Linear(2048, 256)  # ResNet-50 outputs 2048-dimensional features

#         # Downsample temporal tokens to reduce sequence length
#         self.temporal_pool = nn.AdaptiveAvgPool1d(output_size=self.sequence_length // 2)  # Reduce T to T/2

#         # Simplified cross-attention mechanism
#         self.cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)

#         # Lightweight temporal transformer for refined temporal modeling
#         encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True)
#         self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

#         # Anticipation prediction
#         self.fc_anticipation = nn.Linear(256, num_classes)
#         self.fc_phase = nn.Linear(256, num_classes)

#         self.temporal_queries = nn.Parameter(torch.zeros(1, self.sequence_length // 2, 256))

#     def forward(self, x: torch.Tensor):
#         B, T, C, H, W = x.size()

#         # Extract spatial features
#         x = x.view(B * T, C, H, W)
#         spatial_features = self.backbone(x)  # [B*T, 2048]
#         spatial_features = spatial_features.view(B, T, -1)  # [B, T, 2048]

#         # Reduce dimensionality
#         spatial_features = self.reduce_dim(spatial_features)  # [B, T, 256]

#         # Downsample temporal tokens
#         spatial_features = self.temporal_pool(spatial_features.transpose(1, 2))  # [B, 256, T/2]
#         spatial_features = spatial_features.transpose(1, 2)  # [B, T/2, 256]

#         # Create learnable temporal queries
#         temporal_queries = self.temporal_queries.detach().clone()

#         # Apply cross-attention: temporal queries attend to spatial features
#         temporal_output, _ = self.cross_attention(
#             query=temporal_queries.repeat(B, 1, 1),  # [B, T/2, 256]
#             key=spatial_features,  # [B, T/2, 256]
#             value=spatial_features,  # [B, T/2, 256]
#         )

#         # Pass through temporal transformer
#         temporal_output = self.temporal_transformer(temporal_output)  # [B, T/2, 256]

#         # Mean pooling across time dimension to anticipate the future
#         x = temporal_output.mean(dim=1)  # [B, 256]

#         # Classification or temporal anticipation
#         ant: torch.Tensor = self.fc_anticipation(x)  # [B, num_classes]
#         ant = self.time_horizon * torch.sigmoid(ant)

#         ph: torch.Tensor = self.fc_phase(x)  # [B, num_classes]
#         return ph, ant


class TemporalAnticipationModel(nn.Module):
    def __init__(self, sequence_length: int, num_classes: int = 7, time_horizon: int = 5):
        super(TemporalAnticipationModel, self).__init__()

        # Lightweight ResNet-18 backbone
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()

        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.time_horizon = time_horizon

        # Dimensionality reduction
        self.reduce_dim = nn.Conv1d(512, 256, kernel_size=1)  # ResNet-18 outputs 512 features

        # Temporal modeling with lightweight TCN
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=2, stride=1, dilation=2),
            nn.ReLU(),
        )

        # Cross-attention for refined temporal correlation
        self.cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)

        # Anticipation prediction
        self.fc_anticipation = nn.Linear(256, num_classes)
        self.fc_phase = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.size()

        # Extract spatial features
        x = x.view(B * T, C, H, W)
        spatial_features = self.backbone(x)  # [B*T, 512]
        spatial_features = spatial_features.view(B, T, -1).transpose(1, 2)  # [B, 512, T]

        # Reduce dimensionality
        spatial_features = self.reduce_dim(spatial_features)  # [B, 256, T]

        # Temporal modeling with TCN
        temporal_features = self.temporal_conv(spatial_features)  # [B, 256, T]

        # Apply cross-attention
        temporal_output, _ = self.cross_attention(
            query=temporal_features.transpose(1, 2),  # [B, T, 256]
            key=temporal_features.transpose(1, 2),  # [B, T, 256]
            value=temporal_features.transpose(1, 2),  # [B, T, 256]
        )
        temporal_output = temporal_output.mean(dim=1)  # [B, 256]

        # Classification or temporal anticipation
        ant: torch.Tensor = self.fc_anticipation(temporal_output)  # [B, num_classes]
        ant = self.time_horizon * torch.sigmoid(ant)

        ph: torch.Tensor = self.fc_phase(temporal_output)  # [B, num_classes]
        return ph, ant

def __test__():
    T = 30
    xin = torch.randn(4, T, 3, 224, 224)

    model = TemporalAnticipationModel(T)

    current_phase, anticipated_phase = model(xin)

    print(current_phase.shape, anticipated_phase.shape)


if __name__ == "__main__":
    __test__()
