import os
from dataclasses import field, dataclass
from typing import Any

from PIL import Image
import numpy as np

from pydantic import DirectoryPath, PositiveInt

from torch.utils.data import Dataset

# Utils


def get_image(image_path):
    pil_image = Image.open(image_path).convert("RGB")
    np_image = np.array(pil_image)
    return np_image


# Dataset


@dataclass
class TextRecDataset(Dataset):
    images_dir: DirectoryPath = field(metadata="Dir of images")
    img2label: dict = field(metadata="Dict mapping images to labels")
    height: PositiveInt = field(default=32, metadata="Height of images")
    tfms: Any = field(default=None, metadata="Image augmentations")
    min_width: PositiveInt = field(default=40, metadata="Min width of images")

    def __post_init__(
        self,
    ):
        super().__init__()
        self.labeltuple = sorted(
            [(k, v) for k, v in self.img2label.items()], key=lambda x: x[0]
        )

    def __len__(
        self,
    ):
        return len(self.labeltuple)

    @staticmethod
    def expand_image(img, h, w):
        expanded = Image.new("RGB", (w, h), color=3 * (255,))  # white
        expanded.paste(img)
        return expanded

    def get_image(self, image_name: str):
        image_path = os.path.join(self.images_dir, f"{image_name}.png")
        image = Image.open(image_path).convert("RGB")

        w, h = image.size
        ratio = self.height / h  # how the height will change
        nw = round(w * ratio)

        image = image.resize((nw, self.height))

        if nw < self.min_width:
            image = self.expand_image(image, self.height, self.min_width)

        if self.tfms is not None:
            image = self.tfms(image)

        return image

    def __getitem__(self, idx):
        image_name, label = self.labeltuple[idx]
        image = self.get_image(image_name)
        outputs = (image, label)
        return outputs


@dataclass
class TestTextRecDataset(TextRecDataset):
    def __getitem__(self, idx):
        image_name, label = self.labeltuple[idx]
        image = self.get_image(image_name)
        outputs = (image, label, image_name)
        return outputs
