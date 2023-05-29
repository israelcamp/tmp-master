from dataclasses import dataclass
import os

import albumentations as A
import imgaug.augmenters as iaa

from .sroie import SROIETask2DataModule
from .augs import AlbumentationsBackend, ImgaugBackend


@dataclass
class IAMDataModule(SROIETask2DataModule):
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
        tfms = [rotate, affine, pad, elastic]
        # augment = iaa.SomeOf(2, tfms)
        augment = iaa.OneOf(
            [
                iaa.OneOf(tfms),
                iaa.SomeOf(2, tfms),
                iaa.SomeOf(3, tfms),
                iaa.Sequential(tfms),
            ]
        )
        tfms = iaa.OneOf([augment, augment, augment, iaa.Noop()])
        return ImgaugBackend(tfms)

    def get_name2label(self, key: str):
        lst_data = self.img2label[key]
        return {k: v["text"] for k, v in lst_data.items()}

    def setup(self, stage=None):
        image_dir = os.path.join(self.root_dir, "lines")
        if stage == "fit":
            train_img2label = self.get_name2label("tr")
            valid_img2label = self.get_name2label("va")

            self.train_dataset = self.dataset_class(
                images_dir=image_dir,
                img2label=train_img2label,
                height=self.height,
                tfms=self.train_augs(),
            )
            self.val_dataset = self.dataset_class(
                images_dir=image_dir,
                img2label=valid_img2label,
                height=self.height,
            )

        if stage == "test":
            test_img2label = self.get_name2label("te")
            self.test_dataset = self.dataset_class(
                images_dir=image_dir,
                img2label=test_img2label,
                height=self.height,
            )
