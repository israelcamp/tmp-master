import torch
from torch import nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Feature2Embedding(nn.Module):
    """
    Convert [B, C, H, W] image feature tensor to [B, seq_len, D] (B, 512, 512)
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
    ):
        super().__init__()

        self.image_embeddings = nn.Sequential(
            self.block(3, 64, st=(2, 2)),
            self.block(64, 128, st=(2, 2)),
            self.block(128, 256, st=(2, 1)),
            self.block(256, 512, st=(2, 1)),
            self.block(512, 768, st=(2, 1)),
            Feature2Embedding(),
        )

    def block(self, in_channels, out_channels, st=2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=st, padding=1),
            nn.BatchNorm2d(out_channels),
            Swish(),
        )

    def forward(self, images, *args, **kwargs):
        embedding = self.image_embeddings(images)
        return embedding
