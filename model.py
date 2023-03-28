from functools import reduce

import torch
from torch import nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Feature2Embedding(nn.Module):
    """
    Convert [B, C, H, W] image feature tensor to [B, seq_len, D]
    """

    def forward(self, x):
        n = x.shape[0]
        return x.permute(0, 3, 2, 1).reshape(n, -1, 768)


class CNN(nn.Module):
    """
    Custom CNN
    """

    def __init__(
        self,
        stride_list,
        vocab_size,
        kernel_size=3,
    ):
        super().__init__()

        self.lm_head = nn.Linear(768, vocab_size, bias=False)
        self.image_embeddings = nn.Sequential(
            self.block(3, 64, st=(stride_list[0], 1), ks=kernel_size),
            self.block(64, 128, st=(stride_list[1], 1), ks=kernel_size),
            self.block(128, 256, st=(stride_list[2], 1), ks=kernel_size),
            self.block(256, 512, st=(stride_list[3], 1), ks=kernel_size),
            self.block(512, 768, st=(stride_list[4], 1), ks=kernel_size),
            Feature2Embedding(),
        )

    def block(self, in_channels, out_channels, st=2, ks=3):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=ks, stride=1, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            Swish(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=ks,
                stride=st,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            Swish(),
        )

    def forward(self, images, *args, **kwargs):
        embedding = self.image_embeddings(images)
        logits = self.lm_head(embedding)
        return logits
