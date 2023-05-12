from torch import nn

from .utils import Swish, Feature2Embedding


class AbstractCNN(nn.Module):
    """
    Custom CNN
    """

    def __init__(
        self,
        vocab_size: int = 100,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.image_embeddings = None
        self.lm_head = None

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

    def lm(self, embedding):
        return self.lm_head(embedding)


class CNNBase(AbstractCNN):
    """
    Custom CNN
    """

    def __init__(
        self,
        vocab_size: int = 100,
    ):
        super().__init__(vocab_size=vocab_size)

        self.image_embeddings = nn.Sequential(
            self.block(3, 64, st=(2, 2)),
            self.block(64, 128, st=(2, 2)),
            self.block(128, 256, st=(2, 1)),
            self.block(256, 512, st=(2, 1)),
            self.block(512, 768, st=(2, 1)),
            Feature2Embedding(),
        )

        self.lm_head = nn.Linear(768, self.vocab_size)


class CNNSmall(AbstractCNN):
    """
    Custom CNN
    """

    def __init__(
        self,
        vocab_size: int = 100,
    ):
        super().__init__(vocab_size=vocab_size)

        self.image_embeddings = nn.Sequential(
            self.block(3, 64, st=(2, 2)),
            self.block(64, 128, st=(2, 2)),
            self.block(128, 256, st=(2, 1)),
            self.block(256, 512, st=(4, 1)),
            Feature2Embedding(),
        )
        self.lm_head = nn.Linear(512, self.vocab_size)
