import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

from models.non_local import TimeConv


class Identity(nn.Module):
    def forward(self, x):
        return x


class TemporalAnticipationModel(nn.Module):
    def __init__(self, sequence_length: int, num_classes: int = 7, time_horizon: int = 5, future_steps: int = 10):
        super(TemporalAnticipationModel, self).__init__()

        # Lightweight ResNet-18 backbone
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()

        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.time_horizon = time_horizon
        self.future_steps = future_steps  # F

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

        # Future phase anticipation: Predict [F x C] for each frame
        self.fc_future = nn.Linear(256, num_classes * future_steps)  # [256 -> F*C]

        # Current phase classification
        self.fc_phase = nn.Linear(256, num_classes)  # [256 -> C]

        # Remaining time regression
        self.fc_regression = nn.Linear(256, num_classes)  # [256 -> C]

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
        temporal_output = temporal_output.transpose(1, 2)  # [B, 256, T]

        # Future phase anticipation
        future_logits = self.fc_future(temporal_output.transpose(1, 2))  # [B, T, F*C]
        future_logits = future_logits.view(B, T, self.future_steps, self.num_classes)  # [B, T, F, C]

        # Current phase classification
        current_logits = self.fc_phase(temporal_output.mean(dim=2))  # [B, C]

        # Remaining time regression
        # regression_logits = self.fc_regression(temporal_output.mean(dim=2))  # [B, C]
        # Remaining time regression (per frame)
        regression_logits = self.fc_regression(temporal_output.transpose(1, 2))  # [B, T, C]

        return current_logits, future_logits, regression_logits


def __test__():

    from losses.swag_loss import SWAGLoss
    from datasets.cholec80 import Cholec80Dataset
    from torch.utils.data import DataLoader

    B, T, C, H, W = 5, 10, 3, 224, 224  # Batch size, sequence length, channels, height, width
    num_classes = 7
    future_steps = 10
    time_horizon = 5

    # # Mock input
    # xin = torch.randn(B, T, C, H, W)

    # # Mock data loader batch
    # batch = {
    #     "frames_filepath": [f"frame_{i}.jpg" for i in range(T)],
    #     "frames_indexes": torch.arange(T),
    #     "phase_label": torch.tensor(2),
    #     "phase_label_dense": torch.tensor([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
    #     "time_to_next_phase_dense": torch.tensor([
    #         [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 5.0, 4.0, 3.0, 2.0],
    #         [3.0, 2.0, 1.0, 0.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0],
    #         [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 6.0, 5.0, 4.0],
    #         [2.0, 1.0, 0.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0],
    #     ]),
    #     "future_targets": torch.randint(0, 2, (T, future_steps, num_classes)),
    #     "time_to_next_phase": torch.tensor([2.0, 0.0, 4.0, 0.0]),
    # }

    d = Cholec80Dataset(root_dir="./data/cholec80", mode="demo_train", seq_len=T,  fps=1)
    loader = DataLoader(d, batch_size=B, shuffle=True, num_workers=0, pin_memory=False)

    xin, batch = next(iter(loader))

    # Initialize model and loss
    model = TemporalAnticipationModel(
        sequence_length=T, num_classes=num_classes, time_horizon=time_horizon, future_steps=future_steps
    )
    swag_loss = SWAGLoss(future_steps=future_steps, time_horizon=time_horizon)

    # Forward pass
    current_logits, future_logits, regression_logits = model(xin)

    # Compute loss
    current_targets = batch["phase_label"]  # [B]
    future_targets = batch["future_targets"]  # [B, T, F, C]
    regression_targets = batch["time_to_next_phase"]  # [B, C]

    loss_dict = swag_loss(
        current_logits, future_logits, regression_logits, current_targets, future_targets, regression_targets
    )

    print("Loss Components:")
    for key, value in loss_dict.items():
        print(f"{key}: {value.item()}")


if __name__ == "__main__":
    __test__()
