from dataclasses import dataclass
import os

import albumentations as A

from .sroie import SROIETask2DataModule
from .augs import AlbumentationsBackend


@dataclass
class IAMDataModule(SROIETask2DataModule):
    def train_augs(
        self,
    ):
        rotate = A.Affine(
            rotate=(-1, 1),
            translate_px=(0, 10),
            fit_output=True,
            keep_ratio=True,
            cval=(255, 255, 255),
        )
        gaussian = A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 2.0))
        downscale = A.Downscale(scale_min=0.85, scale_max=0.95)
        tfms = A.Compose(
            [
                rotate,
                gaussian,
                downscale,
            ]
        )
        return AlbumentationsBackend(tfms=tfms)

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
