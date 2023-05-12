import torch

from .cnn import CNNBase, CNNSmall, AbstractCNN
from .transformers import (
    TransformersEncoderBase,
    TransformersEncoderSmall,
    AbstractTransformersEncoder,
)
from .crnn import ImageFeatureExtractor


class OCRModel(torch.nn.Module):
    def __init__(
        self, visual_model: AbstractCNN, rec_model: AbstractTransformersEncoder
    ):
        super().__init__()
        self.visual_model = visual_model
        self.rec_model = rec_model

    def forward(self, images, attention_mask=None):
        features = self.visual_model(images)
        logits = self.rec_model(features, attention_mask=attention_mask)
        return logits

    def cnn_lm(self, embedding):
        return self.visual_model.lm(embedding)


__all__ = [
    "OCRModel",
    "TransformersEncoder",
    "CNNBase",
    "CNNSmall",
    "AbstractCNN",
    "ImageFeatureExtractor",
    "TransformersEncoderBase",
    "TransformersEncoderSmall",
    "AbstractTransformersEncoder",
]
