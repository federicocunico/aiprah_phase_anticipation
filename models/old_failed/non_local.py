import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


class NLBlock(nn.Module):
    def __init__(self, feature_num=512, sequence_length: int = 10):
        super(NLBlock, self).__init__()
        self.linear1 = nn.Linear(feature_num, feature_num)
        self.linear2 = nn.Linear(feature_num, feature_num)
        self.linear3 = nn.Linear(feature_num, feature_num)
        self.linear4 = nn.Linear(feature_num, feature_num)
        self.layer_norm = nn.LayerNorm([1, feature_num])
        self.dropout = nn.Dropout(0.2)

        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)
        init.xavier_uniform_(self.linear3.weight)
        init.xavier_uniform_(self.linear4.weight)

    def forward(self, St: torch.Tensor, Lt: torch.Tensor) -> torch.Tensor:
        St_1 = St.view(-1, 1, 512)
        St_1 = self.linear1(St_1)
        Lt_1 = self.linear2(Lt)
        # Lt_1 = Lt_1.transpose(1, 2)
        SL = torch.matmul(St_1, Lt_1.T)
        SL = SL * ((1 / 512) ** 0.5)
        SL = F.softmax(SL, dim=1)
        Lt_2 = self.linear3(Lt)
        SLL = torch.matmul(SL, Lt_2)
        SLL = self.layer_norm(SLL)
        SLL = F.relu(SLL)
        SLL = self.linear4(SLL)
        SLL = self.dropout(SLL)
        SLL = SLL.view(-1, 512)
        return St + SLL


class TimeConv(nn.Module):
    def __init__(self, sequence_length: int = 10, feature_num: int = 512):
        super(TimeConv, self).__init__()
        self.sequence_length = sequence_length
        self.feature_num = feature_num
        self.timeconv1 = nn.Conv1d(self.feature_num, self.feature_num, kernel_size=3, padding=1)
        self.timeconv2 = nn.Conv1d(self.feature_num, self.feature_num, kernel_size=5, padding=2)
        self.timeconv3 = nn.Conv1d(self.feature_num, self.feature_num, kernel_size=7, padding=3)
        self.maxpool_m = nn.MaxPool1d(2, stride=1)
        self.maxpool = nn.AdaptiveMaxPool2d((self.feature_num, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.sequence_length, self.feature_num)
        x = x.transpose(1, 2)  # [B, T, 512] -> [B, 512, T]

        x1: torch.Tensor = self.timeconv1(x)
        y1 = x1.transpose(1, 2).unsqueeze(-1)  # [B, 512, T] -> [B, T, 512, 1]
        # y1 = y1.view(-1,30,512,1)

        x2: torch.Tensor = self.timeconv2(x)
        y2 = x2.transpose(1, 2).unsqueeze(-1)
        # y2 = y2.view(-1,30,512,1)

        x3: torch.Tensor = self.timeconv3(x)
        y3 = x3.transpose(1, 2).unsqueeze(-1)
        # y3 = y3.view(-1,30,512,1)

        x4 = F.pad(x, (1, 0), mode="constant", value=0)
        x4: torch.Tensor = self.maxpool_m(x4)
        y4 = x4.transpose(1, 2).unsqueeze(-1)
        # y4 = y4.view(-1,30,512,1)

        y0 = x.transpose(1, 2).unsqueeze(-1)
        # y0 = y0.view(-1,30,512,1)

        y = torch.cat((y0, y1, y2, y3, y4), dim=-1)
        y: torch.Tensor = self.maxpool(y)
        # y = y.view(-1,30,512)
        y = y.view(-1, self.feature_num)

        return y


# class TimeConv(nn.Module):
#     def __init__(self):
#         super(TimeConv, self).__init__()
#         # Temporal convolutions become feature-wise transformations in this case
#         self.fc1 = nn.Linear(512, 512)
#         self.fc2 = nn.Linear(512, 512)
#         self.fc3 = nn.Linear(512, 512)

#         # Max pooling equivalent for feature space
#         self.feature_pool = nn.AdaptiveMaxPool1d(1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: Input tensor of shape [B, 512] (batch, feature dimension).

#         Returns:
#             Tensor of shape [B, 512] after feature-wise transformation.
#         """
#         # Apply feature-wise transformations
#         x1 = F.relu(self.fc1(x))  # [B, 512]
#         x2 = F.relu(self.fc2(x))  # [B, 512]
#         x3 = F.relu(self.fc3(x))  # [B, 512]

#         # Combine transformed features
#         y = x + x1 + x2 + x3  # Skip connection for stability

#         # Pool features (optional, depending on task)
#         y = y.unsqueeze(2)  # [B, 512, 1]
#         y = self.feature_pool(y).squeeze(2)  # [B, 512]

#         return y
