from .sroie import (
    SROIETask2DataModule,
    MaxPoolImagePad,
    TestSROIETask2DataModule,
)
from .iam import IAMDataModule
from .dataset import TextRecDataset, TestTextRecDataset


__all__ = [
    "SROIETask2DataModule",
    "MaxPoolImagePad",
    "TestSROIETask2DataModule",
    "IAMDataModule",
    "TextRecDataset",
    "TestTextRecDataset",
]
