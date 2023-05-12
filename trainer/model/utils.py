import torch
from torch import nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Feature2Embedding(nn.Module):
    """
    Convert [B, C, H, W] image feature tensor to [B, seq_len, D] (B, 512, 512)
    (B, C, H, W) -> (B, W, H, C)
    """

    def forward(self, x):
        n, c, h, w = x.shape
        return x.permute(0, 3, 2, 1).reshape(n, -1, c)
