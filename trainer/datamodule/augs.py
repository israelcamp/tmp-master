from typing import Any
from PIL import Image

import numpy as np

from dataclasses import field, dataclass


@dataclass
class ImgaugBackend:
    tfms: Any = field(
        default=None, metadata="Image augmentations using imgaug"
    )

    def __call__(self, image: Image):
        image = np.array(image)
        image = self.tfms(image=image)
        image = Image.fromarray(image)
        return image


@dataclass
class AlbumentationsBackend:
    tfms: Any = field(
        default=None, metadata="Image augmentations using albumentations"
    )

    def __call__(self, image: Image):
        image = np.array(image)
        image = self.tfms(image=image)["image"]
        image = Image.fromarray(image)
        return image
