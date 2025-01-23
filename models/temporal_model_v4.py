import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

from models.membank_model import MemBankResNetLSTM
from models.non_local import TimeConv


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()  # [B, T, C]; batch, time, spatio-temporal features
        x = x.transpose(0, 1)  # [T, B, C]
        x, attn_w = self.attn(x, x, x)  # [T, B, C]
        x = x.transpose(0, 1)  # [B, T, C]
        return x, attn_w


class AttentionPooling(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention_weights = torch.nn.Linear(input_dim, 1)  # Learn weights for each time step

    def forward(self, features: torch.Tensor):
        B, T, E = features.size()  # [B, T, E]
        features = features.transpose(0, 1)  # [T, B, E]
        attn_scores = self.attention_weights(features)  # [T, B, 1]
        attn_scores = torch.softmax(attn_scores, dim=0)  # Normalize over time dimension (T)
        pooled_features = (features * attn_scores).sum(dim=0)  # Weighted sum: [B, E]
        return pooled_features, attn_scores


class Identity(nn.Module):
    def forward(self, x):
        return x


class TemporalAnticipationModel(nn.Module):
    def __init__(self, sequence_length: int, num_classes: int = 7, time_horizon: int = 5):
        super(TemporalAnticipationModel, self).__init__()

        # Lightweight ResNet-18 backbone
        self.backbone = MemBankResNetLSTM(num_classes=num_classes, sequence_length=sequence_length)
        # load pretrained model
        self.backbone.load_state_dict(torch.load("./wandb/run-stage1/checkpoints/membank_best.pth", weights_only=True))

        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.time_horizon = time_horizon
        # self.future_steps = future_steps  # F

        self.embedding_dim = 512  # output of backbone
        self.self_attn = SelfAttention(embed_dim=self.embedding_dim, num_heads=2)
        self.pooler_attn = AttentionPooling(input_dim=self.embedding_dim)

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, 1024),  # First layer
            torch.nn.ReLU(),  # Non-linearity
            torch.nn.Linear(1024, num_classes),  # Final output layer
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, 1024),  # First layer
            torch.nn.ReLU(),  # Non-linearity
            torch.nn.Linear(1024, num_classes),  # Final output layer
        )

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.size()

        # get spatio-temporal features
        spatio_temporal_features = self.backbone.get_features(x)  # [B*T, 512]
        spatio_temporal_features = spatio_temporal_features.view(B, T, -1)  # [B, T, 512]
        features, attn_w = self.self_attn(spatio_temporal_features)  # [B, T, 512]
        pooled_features, attn_scores = self.pooler_attn(features)  # [B, 512]

        # Apply regressor
        classification_logits = self.classifier(pooled_features)  # [B, C]

        regression_logits = self.regressor(pooled_features)  # [B, C]
        # Clamp regression logits to [0, time_horizon]
        regression_logits_clamped = torch.nn.functional.sigmoid(regression_logits) * self.time_horizon  # [B, C]

        return classification_logits, regression_logits_clamped


def __test__():


    from datasets.cholec80 import Cholec80Dataset
    from torch.utils.data import DataLoader

    B, T, C, H, W = 5, 10, 3, 224, 224  # Batch size, sequence length, channels, height, width
    num_classes = 7
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

    d = Cholec80Dataset(root_dir="./data/cholec80", mode="demo_train", seq_len=T, fps=1)
    loader = DataLoader(d, batch_size=B, shuffle=True, num_workers=0, pin_memory=False)

    xin, batch = next(iter(loader))

    # Initialize model and loss
    model = TemporalAnticipationModel(sequence_length=T, num_classes=num_classes, time_horizon=time_horizon)

    # Forward pass
    classification_logits, regression_logits = model(xin)

    # Compute loss
    current_targets = batch["phase_label"]  # [B]
    future_targets = batch["future_targets"]  # [B, T, F, C]
    regression_targets = batch["time_to_next_phase"]  # [B, C]

    ce_loss = torch.nn.functional.cross_entropy(classification_logits, current_targets)
    r_loss = torch.nn.functional.mse_loss(regression_logits, regression_targets)

    loss = ce_loss + r_loss

    print("Loss:", loss.item())
    print("Classification logits:", classification_logits)
    print("Regression logits:", regression_logits)


if __name__ == "__main__":
    __test__()
