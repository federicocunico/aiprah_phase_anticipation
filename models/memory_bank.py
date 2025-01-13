import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    def __init__(self, memory_size: int, feature_dim: int):
        super(MemoryBank, self).__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.register_buffer("memory", torch.randn(memory_size, feature_dim))
        self.memory = F.normalize(self.memory, dim=1)  # Normalize features

    def forward(self, query):
        # Normalize query
        query = F.normalize(query, dim=1)

        # Compute similarity scores
        sim = torch.matmul(query, self.memory.T)  # [batch_size, memory_size]
        attention_weights = F.softmax(sim, dim=1)  # [batch_size, memory_size]

        # Retrieve memory features
        memory_features = torch.matmul(attention_weights, self.memory)  # [batch_size, feature_dim]

        return memory_features, attention_weights

    def update(self, new_embeddings):
        # Normalize new embeddings
        new_embeddings = F.normalize(new_embeddings, dim=1)

        # Update memory using FIFO or other strategies (replace oldest entries)
        self.memory = torch.cat((self.memory[len(new_embeddings) :], new_embeddings), dim=0)


def __test__():
    model = MemoryBank(100, 512)
    x = torch.randn(4, 512)
    y, _ = model(x)
    print(y.size())

    model.update(x)
    y, _ = model(x)
    print(y.size())


if __name__ == "__main__":
    __test__()
