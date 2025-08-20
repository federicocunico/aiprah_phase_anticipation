import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

from models.non_local import TimeConv


class Identity(nn.Module):
    def forward(self, x):
        return x


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
