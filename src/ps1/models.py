"""Three-layer temporal CNN: [B,L,768] -> [B,1] story point prediction."""
import torch
from torch import nn


class TokenCNN(nn.Module):
    def __init__(self, embedding_dim=768, channels=(128,64,32), kernel_size=3, dropout=0.2):
        super().__init__()
        if len(channels) != 3 or kernel_size % 2 != 1:
            raise ValueError('Exactly three convolution layers and odd kernel_size required')
        layers = []
        previous = embedding_dim
        for ch in channels:
            layers.append(nn.Conv1d(previous, ch, kernel_size, padding=kernel_size // 2))
            previous = ch
        self.convs = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(nn.Linear(channels[-1], 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, embeddings, mask):
        # [batch, sequence, embeddings] -> [batch, embeddings, sequence]
        x = embeddings.transpose(1, 2)
        mask = mask.unsqueeze(1).to(dtype=x.dtype)
        for conv in self.convs:
            x = torch.relu(conv(x)) * mask  # never pool padded tokens
        pooled = (x * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
        return self.regressor(self.dropout(pooled)).squeeze(-1)
