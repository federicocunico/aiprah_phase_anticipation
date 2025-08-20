import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights, ResNet

#####################################
# Memory Bank Module
#####################################
class MemoryBank:
    def __init__(self, bank_size: int, feature_dim: int):
        """
        Initializes a memory bank with random normalized embeddings.
        :param bank_size: Total number of stored embeddings.
        :param feature_dim: Dimension of each embedding.
        """
        self.bank_size = bank_size
        self.feature_dim = feature_dim
        self.bank = torch.randn(bank_size, feature_dim)
        self.bank = F.normalize(self.bank, dim=1)

    def get_negatives(self, num_negatives: int):
        """
        Randomly sample negative embeddings from the memory bank.
        """
        indices = torch.randint(0, self.bank_size, (num_negatives,))
        return self.bank[indices]

    def update(self, indices: torch.Tensor, embeddings: torch.Tensor, momentum: float = 0.5):
        """
        Updates memory bank entries using a momentum update rule.
        :param indices: Indices of the entries to update.
        :param embeddings: New embeddings to incorporate.
        :param momentum: Update momentum.
        """
        embeddings = F.normalize(embeddings, dim=1)
        self.bank[indices] = momentum * self.bank[indices] + (1 - momentum) * embeddings
        self.bank[indices] = F.normalize(self.bank[indices], dim=1)


#####################################
# Utility: Identity Module
#####################################
class Identity(nn.Module):
    def forward(self, x):
        return x


#####################################
# Modified Backbone: MemBankResNetLSTM
#####################################
class MemBankResNetLSTM(nn.Module):
    def __init__(self, sequence_length: int, num_classes: int = 7):
        """
        Backbone combining ResNet-50 spatial feature extraction and an LSTM for temporal modeling.
        A projection head is added for contrastive representation learning.
        :param sequence_length: Number of frames in the input sequence.
        :param num_classes: Number of output classes.
        """
        super(MemBankResNetLSTM, self).__init__()
        self.sequence_length = sequence_length

        # Load a pretrained ResNet-50 and remove its final fully connected layer.
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        resnet.fc = Identity()  # remove classification head
        self.share: ResNet = resnet

        # LSTM to model temporal dynamics.
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=0.2)

        # Projection head for contrastive learning.
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Initialize LSTM and FC weights.
        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)
        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def get_spatial_features(self, x: torch.Tensor):
        """
        Extracts spatial features from each frame using ResNet-50.
        Expects input shape: (B, T, 3, 224, 224)
        Returns features reshaped to: (B, T, 2048)
        """
        B, T, C, H, W = x.size()
        x = x.view(-1, C, H, W)  # flatten batch and sequence dims
        x = self.share.forward(x)  # (B*T, 2048)
        x = x.view(-1, self.sequence_length, 2048)
        return x

    def get_spatio_temporal_features(self, x: torch.Tensor):
        """
        Applies LSTM over spatial features to capture temporal dynamics.
        Returns features of shape: (B*T, 512)
        """
        x = self.get_spatial_features(x)  # (B, T, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)  # y shape: (B, T, 512)
        y = y.contiguous().view(-1, 512)
        return y

    def forward(self, x: torch.Tensor):
        """
        Forward pass returns:
         - out: The main task output (e.g., regression or classification).
         - proj_features: Embeddings from the projection head for contrastive learning.
        """
        # Get spatio-temporal features using LSTM.
        features = self.get_spatio_temporal_features(x)  # (B*T, 512)
        # Obtain projection embeddings.
        proj_features = self.projection(features)  # (B*T, 128)
        features = self.dropout(features)
        out = self.fc(features)
        return out, proj_features


#####################################
# Self-Attention Module
#####################################
class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8):
        """
        Self-attention using PyTorch's MultiheadAttention.
        :param embed_dim: Dimensionality of input embeddings.
        :param num_heads: Number of attention heads.
        """
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor):
        # x: [B, T, C]
        x = x.transpose(0, 1)  # to shape [T, B, C] as expected by nn.MultiheadAttention
        x, attn_w = self.attn(x, x, x)
        x = x.transpose(0, 1)  # back to [B, T, C]
        return x, attn_w


#####################################
# Attention Pooling Modules
#####################################
class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        """
        Learns weights for each time step to perform weighted sum pooling.
        """
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)
        nn.init.xavier_uniform_(self.attention_weights.weight)

    def forward(self, features: torch.Tensor):
        # features: [B, T, E]
        features = features.transpose(0, 1)  # [T, B, E]
        attn_scores = self.attention_weights(features)  # [T, B, 1]
        attn_scores = torch.softmax(attn_scores, dim=0)  # normalize over time dimension
        pooled_features = (features * attn_scores).sum(dim=0)  # [B, E]
        return pooled_features, attn_scores


class MeanAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        """
        Computes the global mean and refines it with learned attention weights.
        """
        super(MeanAttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)
        nn.init.xavier_uniform_(self.attention_weights.weight)

    def forward(self, features: torch.Tensor):
        # features: [B, T, E]
        mean_features = features.mean(dim=1)  # [B, E]
        attn_scores = torch.softmax(self.attention_weights(mean_features), dim=-1)
        refined_features = mean_features * attn_scores
        return refined_features


class MaxPooler(nn.Module):
    def __init__(self, input_dim):
        """
        Uses adaptive max pooling over the temporal dimension.
        """
        super(MaxPooler, self).__init__()
        self.pooler = nn.AdaptiveMaxPool1d(1)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor):
        # x: [B, T, E]
        x = x.transpose(1, 2)  # [B, E, T]
        x = self.pooler(x)     # [B, E, 1]
        x = x.squeeze(-1)      # [B, E]
        return x


#####################################
# Combined Model: TemporalAnticipationModel
#####################################
class TemporalAnticipationModel(nn.Module):
    def __init__(self, sequence_length: int, num_classes: int = 7, time_horizon: int = 5):
        """
        Combines the backbone with attention and pooling modules.
        The model outputs a regression value (clamped by a sigmoid) scaled to the time horizon.
        :param sequence_length: Number of frames in the sequence.
        :param num_classes: Number of classes (or regression outputs).
        :param time_horizon: The maximum value for the regression output.
        """
        super(TemporalAnticipationModel, self).__init__()
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.time_horizon = time_horizon

        # Backbone: Modified ResNet + LSTM with projection head.
        self.backbone = MemBankResNetLSTM(sequence_length=sequence_length, num_classes=num_classes)
        # Optionally load pretrained weights:
        # self.backbone.load_state_dict(torch.load("path/to/checkpoint.pth", map_location='cpu'))

        # Self-attention is applied over spatial features.
        # Note: Here we use get_spatial_features (output shape: [B, T, 2048])
        self.self_attn = SelfAttention(embed_dim=2048, num_heads=2)
        # Use one of the pooling modules; here, MaxPooler is used.
        self.pooler_max = MaxPooler(input_dim=2048)

        self.regressor = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor):
        # x: [B, T, 3, 224, 224]
        B, T, C, H, W = x.size()
        # Extract spatial features from each frame.
        spatio_temporal_features = self.backbone.get_spatial_features(x)  # [B, T, 2048]
        # Apply self-attention.
        features, attn_w = self.self_attn(spatio_temporal_features)  # [B, T, 2048]
        # Pool features over time.
        pooled_features = self.pooler_max(features)  # [B, 2048]
        # Obtain regression logits.
        regression_logits = self.regressor(pooled_features)  # [B, num_classes]
        # Clamp outputs to [0, time_horizon] using sigmoid scaling.
        regression_logits_clamped = torch.sigmoid(regression_logits) * self.time_horizon
        return regression_logits_clamped


#####################################
# Testing the Models (Example Forward Pass)
#####################################
if __name__ == "__main__":
    # Create a dummy input: batch of 2 sequences, each with 10 frames of size 224x224 with 3 channels.
    B, T, C, H, W = 2, 10, 3, 224, 224
    dummy_input = torch.randn(B, T, C, H, W)

    # Test the backbone.
    backbone = MemBankResNetLSTM(sequence_length=T, num_classes=7)
    out, proj_features = backbone(dummy_input)
    print("Backbone output shape:", out.shape)
    print("Projection features shape:", proj_features.shape)

    # Test the combined temporal anticipation model.
    model = TemporalAnticipationModel(sequence_length=T, num_classes=7, time_horizon=5)
    regression_output = model(dummy_input)
    print("Temporal Anticipation Model output shape:", regression_output.shape)
