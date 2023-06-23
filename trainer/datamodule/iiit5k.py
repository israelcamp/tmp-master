from dataclasses import dataclass
import os
from PIL import Image

import srsly
import imgaug.augmenters as iaa

from .sroie import SROIETask2DataModule
from .augs import ImgaugBackend
from .dataset import GrayScaleTextRecDataset


@dataclass
class IIIT5KDataModule(SROIETask2DataModule):
    @property
    def dataset_class(
        self,
    ):
        return GrayScaleTextRecDataset

    @staticmethod
    def expand_image(img, h, w):
        expanded = Image.new("L", (w, h), color=(0,))  # black
        expanded.paste(img)
        return expanded

    def setup(self, stage):
        train_images_dir = os.path.join(self.root_dir, "train")

        # separate names
        image_names = sorted(list(set([k for k in self.img2label])))
        train_size = round(len(image_names) * (1.0 - self.val_pct))
        train_image_names = image_names[:train_size]
        valid_image_names = image_names[train_size:]

        # create datasets
        train_img2label = {k: self.img2label[k] for k in train_image_names}
        valid_img2label = {k: self.img2label[k] for k in valid_image_names}

        self.train_dataset = self.dataset_class(
            images_dir=train_images_dir,
            img2label=train_img2label,
            height=self.height,
        )

        self.val_dataset = self.dataset_class(
            images_dir=train_images_dir,
            img2label=valid_img2label,
            height=self.height,
        )

        test_json_path = os.path.join(self.root_dir, "test.json")
        test_images_dir = os.path.join(self.root_dir, "test")
        self.test_img2label = srsly.read_json(test_json_path)
        self.test_dataset = self.dataset_class(
            images_dir=test_images_dir,
            img2label=self.test_img2label,
            height=self.height,
        )
