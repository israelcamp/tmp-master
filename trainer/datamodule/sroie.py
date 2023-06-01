from typing import Any
from dataclasses import field, dataclass

from PIL import Image
import srsly


import imgaug.augmenters as iaa
from torch.utils.data import DataLoader
import torchvision as tv
import torch

from .dataset import TextRecDataset, TestTextRecDataset
from .augs import ImgaugBackend


class MaxPoolImagePad:
    def __init__(self, pooler="mine"):
        if pooler == "crnn":
            self.pool = torch.nn.Sequential(
                torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                torch.nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
            )
        if pooler == "iam":
            self.pool = torch.nn.Sequential(
                torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            )
        else:
            self.pool = torch.nn.Sequential(
                torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            )

    def __call__(self, x):
        return self.pool(x)


# DataModule


@dataclass
class SROIETask2DataModule:
    root_dir: str = field(metadata="Dir of images")
    label_file: str = field(metadata="JSON file of labels")
    tokenizer: Any = field(metadata="tokenizer")
    height: int = field(default=32, metadata="Height of images")
    train_bs: int = field(default=16, metadata="Training batch size")
    valid_bs: int = field(default=16, metadata="Eval batch size")
    num_workers: int = field(default=2)
    val_pct: float = field(
        default=0.1, metadata="Percentage of images to validation"
    )
    pooler_mode: str = field(default="mine", metadata="Pooling method")

    max_width: int = field(default=None)
    do_pool: bool = field(default=True)

    def __post_init__(
        self,
    ):
        self.img2label = srsly.read_json(self.label_file)
        self.pooler = MaxPoolImagePad(pooler=self.pooler_mode)

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
        gaussian = iaa.imgcorruptlike.GaussianNoise(severity=(1, 3))
        jpeg = iaa.imgcorruptlike.JpegCompression(severity=(1, 5))
        pixelate = iaa.imgcorruptlike.Pixelate(severity=(1, 4))
        dropout = iaa.Dropout(p=(0, 0.05))
        tfms = [
            rotate,
            affine,
            pad,
            elastic,
            gaussian,
            jpeg,
            pixelate,
            dropout,
        ]
        augment = iaa.OneOf(
            [
                iaa.OneOf(tfms),
                iaa.SomeOf(2, tfms),
                iaa.SomeOf(3, tfms),
                iaa.SomeOf(4, tfms),
                iaa.SomeOf(5, tfms),
                iaa.Sequential(tfms),
            ]
        )
        tfms = iaa.OneOf([augment, augment, augment, iaa.Noop()])
        return ImgaugBackend(tfms=tfms)

    @property
    def dataset_class(
        self,
    ):
        return TextRecDataset

    def setup(self, stage):
        if stage == "fit":
            # separate names
            image_names = sorted(
                list(set([k.split("__")[0] for k in self.img2label]))
            )
            train_size = round(len(image_names) * (1.0 - self.val_pct))
            train_image_names = image_names[:train_size]
            valid_image_names = image_names[train_size:]

            # create datasets
            train_img2label = {
                k: v
                for k, v in self.img2label.items()
                if k.split("__")[0] in train_image_names
            }
            valid_img2label = {
                k: v
                for k, v in self.img2label.items()
                if k.split("__")[0] in valid_image_names
            }

            self.train_dataset = self.dataset_class(
                images_dir=self.root_dir,
                img2label=train_img2label,
                height=self.height,
                tfms=self.train_augs(),
            )
            self.val_dataset = self.dataset_class(
                images_dir=self.root_dir,
                img2label=valid_img2label,
                height=self.height,
            )

        if stage == "test":
            self.test_dataset = self.val_dataset

    @staticmethod
    def expand_image(img, h, w):
        expanded = Image.new("RGB", (w, h), color=3 * (255,))  # white
        expanded.paste(img)
        return expanded

    def collate_fn(self, samples):
        images = [s[0] for s in samples]
        labels = [s[1] for s in samples]

        image_widths = [im.width for im in images]
        max_width = (
            self.max_width if self.max_width is not None else max(image_widths)
        )

        attention_images = []
        for w in image_widths:
            attention_images.append([1] * w + [0] * (max_width - w))
        attention_images = (
            self.pooler(torch.tensor(attention_images).float()).long()
            if self.do_pool
            else None
        )

        h = images[0].size[1]
        to_tensor = tv.transforms.ToTensor()
        images = [
            to_tensor(self.expand_image(im, h=h, w=max_width)) for im in images
        ]

        tokens = self.tokenizer.batch_encode_plus(
            labels, padding="longest", return_tensors="pt"
        )
        input_ids = tokens.get("input_ids")
        attention_mask = tokens.get("attention_mask")

        return torch.stack(images), input_ids, attention_mask, attention_images

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_bs,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.valid_bs,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.valid_bs,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )


@dataclass
class TestSROIETask2DataModule(SROIETask2DataModule):
    def train_augs(
        self,
    ):
        return None

    @property
    def dataset_class(self):
        return TestTextRecDataset

    def setup(self, stage):
        super().setup(stage)
        self.test_dataset = torch.utils.data.ConcatDataset(
            [self.train_dataset, self.val_dataset]
        )

    def collate_fn(self, samples):
        outputs = super().collate_fn(samples)
        names = [s[2] for s in samples]
        outputs = outputs + (names,)
        return outputs

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_bs,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
