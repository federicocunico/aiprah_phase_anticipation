import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights, ResNet


class Identity(nn.Module):
    def forward(self, x):
        return x


class MemBankResNetLSTM(torch.nn.Module):
    def __init__(self, sequence_length: int, num_classes: int = 7):
        super(MemBankResNetLSTM, self).__init__()

        self.sequence_length = sequence_length

        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        resnet.fc = Identity()
        self.share: ResNet = resnet
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=0.2)

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)

    def get_spatio_temporal_features(self, x: torch.Tensor):
        x = self.get_spatial_features(x)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        return y

    def get_spatial_features(self, x: torch.Tensor):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, self.sequence_length, 2048)
        return x

    def forward(self, x: torch.Tensor):
        y = self.get_spatio_temporal_features(x)
        y = self.dropout(y)
        y = self.fc(y)
        return y


def __test__():
    model = MemBankResNetLSTM(10)
    x = torch.randn(1, 10, 3, 224, 224)
    y = model(x)
    print(y.size())


if __name__ == "__main__":
    __test__()
