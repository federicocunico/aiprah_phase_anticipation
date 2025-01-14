import torch
from torch.nn import init
from torch import nn
import torch.nn.functional as F
from models.membank_model import MemBankResNetLSTM
from models.memory_bank import MemoryBank
from models.non_local import NLBlock, TimeConv


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class TemporalResNetLSTM(nn.Module):
    def __init__(
        self,
        backbone: MemBankResNetLSTM,
        memory_bank: MemoryBank,
        sequence_length: int,
        num_classes: int = 7,
        time_horizon: int = 5,
    ):
        super(TemporalResNetLSTM, self).__init__()

        self.sequence_length = sequence_length
        self.backbone: MemBankResNetLSTM = backbone  # Feature extractor
        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.memory_bank: MemoryBank = memory_bank
        # self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.fc_h_c = nn.Linear(1024*sequence_length, 512)  # Hidden state combination

        self.fc_c = nn.Linear(512, num_classes)  # Phase classification
        self.fc_anticipation = nn.Linear(512, 1)  # Phase anticipation
        self.nl_block = NLBlock(feature_num=512, sequence_length=sequence_length)
        self.dropout = nn.Dropout(p=0.5)
        self.time_conv = TimeConv()
        self.time_horizon = time_horizon  # minutes

        # Weight initialization
        # init.xavier_normal_(self.lstm.all_weights[0][0])
        # init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc_c.weight)
        init.xavier_uniform_(self.fc_h_c.weight)
        init.xavier_uniform_(self.fc_anticipation.weight)

    # def get_features(self, x: torch.Tensor):
    #     # Extract features from the backbone
    #     x = x.view(-1, 3, 224, 224)
    #     x = self.backbone.forward(x)
    #     x = x.view(-1, self.sequence_length, 2048)

    #     # Process with LSTM
    #     self.lstm.flatten_parameters()
    #     y, _ = self.lstm(x)
    #     y = y.contiguous().view(-1, 512)
    #     y = y[self.sequence_length - 1 :: self.sequence_length]  # Use the last hidden state for the sequence

    #     return y

    def forward(self, x: torch.Tensor):
        y: torch.Tensor = self.backbone.get_features(x)  # [batch_size*T, 512]
        # y = y.view(-1, self.sequence_length, 512)  # [batch_size, T, 512]

        # query memory bank
        mb_features, _ = self.memory_bank(y)  # [batch_size*T, 512]  # agisce su B*T

        # Long-range features via time convolution
        Lt = self.time_conv(mb_features)  # [batch_size*T, 512]  # agisce sulle features spaziali (512)

        # Non-local operation with the long-range features
        y_1 = self.nl_block(y, Lt)  # [batch_size*T, 512]  # agisce su B*T

        # Combine local and long-range features
        y = y.view(-1, self.sequence_length*512)  # [batch_size, T*512]
        y_1 = y_1.view(-1, self.sequence_length*512)  # [batch_size, T*512]
        y_combined = torch.cat([y, y_1], dim=-1)  # [B, T*1024]
        y_combined = self.dropout(self.fc_h_c(y_combined))  # [B, T, 512]
        y_combined = F.relu(y_combined)  # ReLU activation

        # Current phase prediction
        current_phase = self.fc_c(y_combined)  # [batch_size, num_classes]

        # Phase anticipation
        anticipated_phase = self.fc_anticipation(y_combined)  # [batch_size, 1]
        anticipated_phase = self.time_horizon * F.sigmoid(anticipated_phase)    
        anticipated_phase = anticipated_phase.squeeze(-1)


        # assert current_phase.shape[0] == anticipated_phase.shape[0]
        return current_phase, anticipated_phase

    # @staticmethod
    # def compute_kl_divergence(predicted_logits, target_probs):
    #     """
    #     Compute KL Divergence loss for phase anticipation.
    #     Args:
    #         predicted_logits (torch.Tensor): Predicted logits from the model (before softmax).
    #         target_probs (torch.Tensor): Ground truth probabilities for anticipation.
    #     Returns:
    #         torch.Tensor: KL divergence loss.
    #     """
    #     predicted_probs = F.log_softmax(predicted_logits, dim=-1)  # Convert logits to log-probabilities
    #     kl_loss = F.kl_div(predicted_probs, target_probs, reduction="batchmean")
    #     return kl_loss


# class TemporalResNetLSTM(torch.nn.Module):
#     def __init__(self, backbone: MemBankResNetLSTM, sequence_length: int, num_classes: int = 7):
#         super(TemporalResNetLSTM, self).__init__()

#         self.sequence_length = sequence_length
#         self.backbone: MemBankResNetLSTM = backbone.share  # get only feature extractor
#         self.lstm = nn.LSTM(2048, 512, batch_first=True)
#         self.fc_c = nn.Linear(512, num_classes)
#         self.fc_h_c = nn.Linear(1024, 512)
#         self.nl_block = NLBlock()
#         self.dropout = nn.Dropout(p=0.5)
#         self.time_conv = TimeConv()

#         init.xavier_normal_(self.lstm.all_weights[0][0])
#         init.xavier_normal_(self.lstm.all_weights[0][1])
#         init.xavier_uniform_(self.fc_c.weight)
#         init.xavier_uniform_(self.fc_h_c.weight)

#     def forward(self, x: torch.Tensor, mb_features: torch.Tensor) -> torch.Tensor:
#         x = x.view(-1, 3, 224, 224)
#         x = self.backbone.forward(x)
#         x = x.view(-1, self.sequence_length, 2048)
#         self.lstm.flatten_parameters()
#         y, _ = self.lstm(x)
#         y = y.contiguous().view(-1, 512)
#         y = y[self.sequence_length - 1::self.sequence_length]

#         Lt = self.time_conv(mb_features)

#         y_1 = self.nl_block(y, Lt)
#         y = torch.cat([y, y_1], dim=1)
#         y = self.dropout(self.fc_h_c(y))
#         y = F.relu(y)
#         y = self.fc_c(y)
#         return y


def __test__():
    from memory_bank_utils import get_long_range_feature_clip

    xin = torch.randn(4, 10, 3, 224, 224)
    backbone = MemBankResNetLSTM(10)
    mb = MemoryBank(1000, 512)

    model = TemporalResNetLSTM(backbone, mb, 10)

    # lrf = get_long_range_feature_clip(xin, [{"frames_indexes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}], mb)
    # ph, ph_ant = model(xin, lrf)

    current_phase, anticipated_phase = model(xin)

    print(current_phase.shape, anticipated_phase)


if __name__ == "__main__":
    __test__()
