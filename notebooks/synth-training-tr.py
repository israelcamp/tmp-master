# %%
CODE_PATH = "../trainer"

# %%
import sys

sys.path.append(CODE_PATH)

# %%
import os
from PIL import Image
import random

import torch
import torchvision as tv
from transformers import AutoTokenizer
from transformers import DebertaV2ForTokenClassification, DebertaV2Config

from ignite.engine import (
    Engine,
    Events,
)
from ignite.handlers import Checkpoint
from ignite.contrib.handlers import global_step_from_engine
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.neptune_logger import NeptuneLogger

# %%
from ctc import GreedyDecoder
from igmetrics import ExactMatch

# %%
tokenizer = AutoTokenizer.from_pretrained(
    f"{CODE_PATH}/synth-tokenizers/tokenizer-pad0"
)
decoder = GreedyDecoder(tokenizer.pad_token_id)

# %% [markdown]
# # Dataset


# %%
class SynthDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotation_file, height=32):
        self.images_dir = images_dir
        self.annotation_file = annotation_file
        self.image_files = self._load_data()
        self.height = height

    def _load_data(self):
        with open(self.annotation_file, "r") as f:
            lines = f.read().splitlines()

        image_files = [line.split(" ")[0] for line in lines]
        return image_files

    def __len__(self):
        return len(self.image_files)

    def read_image_file_and_label(self, image_file):
        label = image_file.split("_")[1]
        image_path = os.path.join(self.images_dir, image_file)

        image = Image.open(image_path).convert("L")
        w, h = image.size
        ratio = w / float(h)
        nw = round(self.height * ratio)

        image = image.resize((nw, self.height), Image.BICUBIC)

        return image, label

    def __getitem__(self, idx):
        image_file = self.image_files[idx]

        try:
            image, label = self.read_image_file_and_label(image_file)
        except:
            print(f"Error reading image {image_file} idx {idx}")
            return self.__getitem__(
                random.randint(0, len(self.image_files) - 1)
            )

        return image, label


# %%
IMAGES_DIR = "../data/synth/mnt/90kDICT32px/"
TRAIN_ANNOTATION_FILE = "../data/synth/mnt/annotation_train_good.txt"
VAL_ANNOTATION_FILE = "../data/synth/mnt/annotation_val_good.txt"

# %%
train_dataset = SynthDataset(IMAGES_DIR, TRAIN_ANNOTATION_FILE)
val_dataset = SynthDataset(IMAGES_DIR, VAL_ANNOTATION_FILE)

# %% [markdown]
# # DataModule

# %%
from dataclasses import dataclass, field
from typing import Any

from torch.utils.data import DataLoader


# %%
class MaxPoolImagePad:
    def __init__(self):
        self.pool = torch.nn.Sequential(
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

    def __call__(self, x):
        return self.pool(x)


# %%
POOLER = MaxPoolImagePad()


# %%
@dataclass
class SynthDataModule:
    train_dataset: Any = field(metadata="Training dataset")
    val_dataset: Any = field(metadata="Validation dataset")
    tokenizer: Any = field(metadata="tokenizer")
    train_bs: int = field(default=16, metadata="Training batch size")
    valid_bs: int = field(default=16, metadata="Eval batch size")
    num_workers: int = field(default=2)
    max_width: int = field(default=None)

    @staticmethod
    def expand_image(img, h, w):
        expanded = Image.new("L", (w, h), color=(0,))  # black
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
        attention_images = POOLER(
            torch.tensor(attention_images).float()
        ).long()

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


# %%
datamodule = SynthDataModule(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    tokenizer=tokenizer,
    train_bs=4,
    valid_bs=16,
    num_workers=4,
)

# %%
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()

# %%
from torch import nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Feature2Embedding(nn.Module):
    def forward(self, x):
        n, c, h, w = x.shape
        assert h == 1, "the height of out must be 1"
        x = x.squeeze(2)  # [n, c, w]
        return x.permute(0, 2, 1)  # [n, w, c]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, st=(2, 1)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            Swish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=st, padding=1
            ),
            nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=st, padding=0
            ),
            nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
        )

        self.swish = Swish()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.conv2(x) + self.downsample(input)
        x = self.swish(x)
        return x


class ResNetLike(nn.Module):
    """
    Custom CNN
    """

    def __init__(
        self,
        vocab_size: int = 100,
        p: float = 0.0,
    ):
        super().__init__()

        self.image_embeddings = nn.Sequential(
            self.block(1, 64, st=(2, 2)),
            nn.Dropout2d(p),
            self.block(64, 128, st=(2, 2)),
            nn.Dropout2d(p),
            self.block(128, 256, st=(2, 1)),
            nn.Dropout2d(p),
            self.block(256, 512, st=(4, 1)),
            nn.Dropout2d(p),
            Feature2Embedding(),
        )
        self.lm_head = nn.Linear(512, vocab_size)

    def block(self, in_channels, out_channels, st=2):
        return ResidualBlock(
            in_channels=in_channels, out_channels=out_channels, st=st
        )

    def forward(self, images, *args, **kwargs):
        embedding = self.image_embeddings(images)
        return embedding

    def lm(self, embedding):
        return self.lm_head(embedding)


class AbstractTransformersEncoder(torch.nn.Module):
    def __init__(self, vocab_size: int = 100, config_dict: dict = {}):
        super().__init__()
        self.vocab_size = vocab_size
        config_dict = self._get_config_dict(config_dict)
        config = DebertaV2Config(**config_dict)
        self.encoder = DebertaV2ForTokenClassification(config)

    def _get_config_dict(self, config_dict):
        base_config_dict = {
            "model_type": "deberta-v2",
            "architectures": ["DebertaV2ForTokenClassification"],
            "num_labels": self.vocab_size,
            "model_type": "deberta-v2",
            "attention_probs_dropout_prob": 0.15,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.15,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 768,  # 3072,
            "max_position_embeddings": 512 + 128,  # had to change from 512
            "relative_attention": True,
            "position_buckets": 64,  # TODO: Maybe less?
            "norm_rel_ebd": "layer_norm",
            "share_att_key": True,
            "pos_att_type": "p2c|c2p",
            "layer_norm_eps": 1e-7,
            "max_relative_positions": -1,
            "position_biased_input": True,
            "num_attention_heads": 8,
            "num_hidden_layers": 3,
            "type_vocab_size": 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "vocab_size": self.vocab_size,
        }
        base_config_dict.update(config_dict)
        return base_config_dict

    def forward(self, image_embeddings, attention_mask=None):
        outputs = self.encoder(
            inputs_embeds=image_embeddings, attention_mask=attention_mask
        )
        return outputs.logits


# %%
vis_model = ResNetLike(vocab_size=tokenizer.vocab_size, p=0.15)
tr_model = AbstractTransformersEncoder(vocab_size=tokenizer.vocab_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
_ = vis_model.to(device)
_ = tr_model.to(device)

# %%
MAX_EPOCHS = 2
STEPS = len(train_loader) * MAX_EPOCHS
STEPS

# %%
optimizer = torch.optim.AdamW(tr_model.parameters(), lr=1e-4, weight_decay=0)
vis_optimizer = torch.optim.AdamW(
    vis_model.parameters(), lr=1e-4, weight_decay=0
)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, STEPS, 1e-8
)
vis_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    vis_optimizer, STEPS, 1e-8
)
criterion = torch.nn.CTCLoss(blank=tokenizer.pad_token_id, zero_infinity=True)


# %%
def get_preds_from_logits(logits, attention_image, labels):
    decoded_ids = logits.argmax(-1).squeeze(0)
    if len(decoded_ids.shape) == 1:
        decoded_ids = decoded_ids.unsqueeze(0)

    decoded = [
        decoder(dec, att) for dec, att in zip(decoded_ids, attention_image)
    ]
    y_pred = tokenizer.batch_decode(decoded, skip_special_tokens=True)
    y = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return y_pred, y


# %%
def train_step(engine, batch):
    vis_model.train()
    tr_model.train()

    optimizer.zero_grad()
    tr_model.zero_grad()
    vis_model.zero_grad()

    images, labels, attention_mask, attention_image = [
        x.to(device) if x is not None else x for x in batch
    ]

    # with torch.no_grad():
    image_embeddings = vis_model(images)
    logits = tr_model(image_embeddings, attention_mask=attention_image)

    input_length = attention_image.sum(-1)
    target_length = attention_mask.sum(-1)

    logits = logits.permute(1, 0, 2)
    logits = logits.log_softmax(2)

    loss = criterion(logits, labels, input_length, target_length)

    # check if loss is nan
    isnan_loss = torch.isnan(loss).item()
    if isnan_loss:
        print("Loss is NaN")
        sys.exit(1)
        return 0

    loss.backward()

    torch.nn.utils.clip_grad_norm_(tr_model.parameters(), 1.0)

    optimizer.step()
    lr_scheduler.step()
    vis_optimizer.step()
    vis_lr_scheduler.step()
    return loss.item()


# %%
def val_step(engine, batch):
    vis_model.eval()
    tr_model.eval()
    images, labels, _, attention_image = [
        x.to(device) if x is not None else x for x in batch
    ]
    with torch.no_grad():
        image_embeddings = vis_model(images)
        logits = tr_model(image_embeddings, attention_mask=attention_image)

    y_pred, y = get_preds_from_logits(logits, attention_image, labels)
    return y_pred, y


# %%
trainer = Engine(train_step)
train_evaluator = Engine(val_step)
validation_evaluator = Engine(val_step)


# %%
def log_validation_results(engine):
    validation_evaluator.run(val_loader)
    metrics = validation_evaluator.state.metrics
    avg_accuracy = metrics["accuracy"]
    print(
        f"Validation Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.3f}"
    )


# %%
ExactMatch().attach(validation_evaluator, "accuracy")

trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

# %%
to_save = {
    "model": tr_model,
    "optimizer": optimizer,
    "lr_scheduler": lr_scheduler,
    "trainer": trainer,
}
handler = Checkpoint(
    to_save,
    "synth-tr-checkpoint-models",
    n_saved=4,
)
trainer.add_event_handler(Events.ITERATION_COMPLETED(every=10_000), handler)

# %%
vis_checkpoint = torch.load(
    "/home/israel/Mestrado/notebooks/synth-broken-ckpts/best_model_2_accuracy=0.8796.pt"
)
print("Loading vis weights", vis_model.load_state_dict(vis_checkpoint))
tr_checkpoint = torch.load(
    "/home/israel/Mestrado/notebooks/synth-broken-tr-ckpts/best_model_1_accuracy=0.9020.pt"
)
print("Loading tr weights", tr_model.load_state_dict(tr_checkpoint))


# TODO: correct path
# tr_checkpoint = torch.load(
#     "/home/israel/Mestrado/notebooks/synth-broken-tr-ckpts/checkpoint_870000.pt"
# )
# Checkpoint.load_objects(to_load=to_save, checkpoint=tr_checkpoint)


# %%
to_save = {"model": tr_model}
handler = Checkpoint(
    to_save,
    "synth-tr-checkpoint-models",
    n_saved=1,
    filename_prefix="best",
    score_name="accuracy",
    global_step_transform=global_step_from_engine(trainer),
)
validation_evaluator.add_event_handler(Events.COMPLETED, handler)

# %%
# neptune_logger = NeptuneLogger(
#     project="i155825/TRecPretrain",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhOGUyY2VlOS1hZTU5LTQ2NGQtYTY5Zi04OGJmZWM2M2NlMDAifQ==",
# )

# neptune_logger.attach_output_handler(
#     trainer,
#     event_name=Events.ITERATION_COMPLETED(every=100),
#     tag="training",
#     output_transform=lambda loss: {"loss": loss},
# )

# neptune_logger.attach_output_handler(
#     validation_evaluator,
#     event_name=Events.EPOCH_COMPLETED,
#     tag="validation",
#     metric_names=["accuracy"],
#     global_step_transform=global_step_from_engine(trainer),
# )

# neptune_logger["code"].upload_files(
#     [
#         f"{CODE_PATH}/*.py",
#         f"{CODE_PATH}/**/*.py",
#         "/home/israel/Mestrado/notebooks/synth-training-tr.py",
#     ]
# )

# %%
pbar = ProgressBar()
pbar.attach(trainer, output_transform=lambda x: {"loss": x})

# %%
trainer.run(train_loader, max_epochs=MAX_EPOCHS)

# %%
