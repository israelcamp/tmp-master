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
    def train_augs(
        self,
    ):
        rotate = iaa.KeepSizeByResize(
            iaa.Affine(rotate=(-5, 5), cval=255, fit_output=True)
        )
        affine = iaa.Affine(
            scale=(0.98, 1.02),
            cval=255,
        )
        pad = iaa.Pad(
            percent=((0, 0.01), (0, 0.1), (0, 0.01), (0, 0.1)),
            keep_size=False,
            pad_cval=255,
        )
        elastic = iaa.ElasticTransformation(alpha=(0.0, 10.0), sigma=2.0)
        tfms = [rotate, affine, pad, iaa.Noop(), elastic]
        augment = iaa.OneOf(
            [
                iaa.OneOf(tfms),
                iaa.SomeOf(2, tfms),
            ]
        )
        return ImgaugBackend(tfms=augment)

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
            tfms=self.train_augs(),
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
