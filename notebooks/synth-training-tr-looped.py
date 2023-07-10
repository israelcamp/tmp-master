CODE_PATH = "../trainer"

import sys

sys.path.append(CODE_PATH)

from dataclasses import dataclass, field
from typing import Any
import os
from PIL import Image
import random

import torch
from torch import nn
import torchvision as tv
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import DebertaV2ForTokenClassification, DebertaV2Config

from ignite.engine import (
    Engine,
    Events,
)
from ignite.handlers import Checkpoint
from ignite.contrib.handlers import global_step_from_engine
from ignite.contrib.handlers import ProgressBar


from ctc import GreedyDecoder
from igmetrics import ExactMatch


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


class MaxPoolImagePad:
    def __init__(self):
        self.pool = torch.nn.Sequential(
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

    def __call__(self, x):
        return self.pool(x)


@dataclass
class SynthDataModule:
    train_dataset: Any = field(metadata="Training dataset")
    val_dataset: Any = field(metadata="Validation dataset")
    tokenizer: Any = field(metadata="tokenizer")
    train_bs: int = field(default=16, metadata="Training batch size")
    valid_bs: int = field(default=16, metadata="Eval batch size")
    num_workers: int = field(default=2)
    max_width: int = field(default=None)

    pooler: Any = field(default_factory=MaxPoolImagePad)

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
        attention_images = self.pooler(
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
    def __init__(
        self, vocab_size: int = 100, config_dict: dict = {}, tokenizer=None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
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
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "vocab_size": self.vocab_size,
        }
        base_config_dict.update(config_dict)
        return base_config_dict

    def forward(self, image_embeddings, attention_mask=None):
        outputs = self.encoder(
            inputs_embeds=image_embeddings, attention_mask=attention_mask
        )
        return outputs.logits


def get_preds_from_logits(logits, attention_image, labels, decoder, tokenizer):
    decoded_ids = logits.argmax(-1).squeeze(0)
    if len(decoded_ids.shape) == 1:
        decoded_ids = decoded_ids.unsqueeze(0)

    decoded = [
        decoder(dec, att) for dec, att in zip(decoded_ids, attention_image)
    ]
    y_pred = tokenizer.batch_decode(decoded, skip_special_tokens=True)
    y = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return y_pred, y


def train_step(
    engine,
    batch,
    model,
    optimizer,
    lr_scheduler,
    criterion,
    device,
):
    model.train()

    images, labels, attention_mask, attention_image = [
        x.to(device) if x is not None else x for x in batch
    ]

    logits = model(images, attention_mask=attention_image)

    input_length = attention_image.sum(-1)
    target_length = attention_mask.sum(-1)

    logits = logits.permute(1, 0, 2)
    logits = logits.log_softmax(2)

    loss = criterion(logits, labels, input_length, target_length)

    # check if loss is nan
    isnan_loss = torch.isnan(loss).item()
    if isnan_loss:
        print("Loss is NaN")
        raise Exception("Loss is NaN")

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    model.zero_grad()

    return loss.item()


def val_step(engine, batch, model, device, decoder, tokenizer):
    model.eval()
    images, labels, _, attention_image = [
        x.to(device) if x is not None else x for x in batch
    ]
    with torch.no_grad():
        logits = model(images, attention_mask=attention_image)

    y_pred, y = get_preds_from_logits(
        logits, attention_image, labels, decoder, tokenizer
    )
    return y_pred, y


def log_validation_results(engine, validation_evaluator, val_loader):
    validation_evaluator.run(val_loader)
    metrics = validation_evaluator.state.metrics
    avg_accuracy = metrics["accuracy"]
    print(
        f"Validation Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.3f}"
    )


class OCRModel(torch.nn.Module):
    def __init__(self, visual_model, rec_model: AbstractTransformersEncoder):
        super().__init__()
        self.visual_model = visual_model
        self.rec_model = rec_model

    def forward(self, images, attention_mask=None):
        features = self.visual_model(images)
        logits = self.rec_model(features, attention_mask=attention_mask)
        return logits

    def cnn_lm(self, embedding):
        return self.visual_model.lm(embedding)


if __name__ == "__main__":
    IMAGES_DIR = "../data/synth/mnt/90kDICT32px/"
    TRAIN_ANNOTATION_FILE = "../data/synth/mnt/annotation_train_good.txt"
    VAL_ANNOTATION_FILE = "../data/synth/mnt/annotation_val_good.txt"

    def run(last_good_checkpoint: str = None):
        from functools import partial

        tokenizer = AutoTokenizer.from_pretrained(
            f"{CODE_PATH}/synth-tokenizers/tokenizer-pad0"
        )
        decoder = GreedyDecoder(tokenizer.pad_token_id)

        train_dataset = SynthDataset(IMAGES_DIR, TRAIN_ANNOTATION_FILE)
        val_dataset = SynthDataset(IMAGES_DIR, VAL_ANNOTATION_FILE)

        datamodule = SynthDataModule(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            train_bs=4,
            valid_bs=16,
            num_workers=4,
        )

        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        vis_model = ResNetLike(vocab_size=tokenizer.vocab_size, p=0.15)
        tr_model = AbstractTransformersEncoder(
            vocab_size=tokenizer.vocab_size, tokenizer=tokenizer
        )
        model = OCRModel(vis_model, tr_model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _ = model.to(device)

        MAX_EPOCHS = 2
        STEPS = len(train_loader) * MAX_EPOCHS

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=0
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, STEPS, 1e-8
        )
        criterion = torch.nn.CTCLoss(
            blank=tokenizer.pad_token_id, zero_infinity=True
        )

        partial_train_step = partial(
            train_step,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            device=device,
        )
        partial_val_step = partial(
            val_step,
            model=model,
            device=device,
            decoder=decoder,
            tokenizer=tokenizer,
        )
        trainer = Engine(partial_train_step)
        validation_evaluator = Engine(partial_val_step)

        ExactMatch().attach(validation_evaluator, "accuracy")

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            partial(
                log_validation_results,
                validation_evaluator=validation_evaluator,
                val_loader=val_loader,
            ),
        )

        to_save = {
            "model": model,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "trainer": trainer,
        }
        handler = Checkpoint(
            to_save,
            "synth-checkpoint-models",
            n_saved=4,
        )
        if last_good_checkpoint is not None:
            print("Loading last good checkpoint", last_good_checkpoint)
            checkpoint = torch.load(last_good_checkpoint)
            Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every=10_000), handler
        )

        best_to_save = {"model": model}
        best_handler = Checkpoint(
            best_to_save,
            "synth-checkpoint-models",
            n_saved=2,
            filename_prefix="best",
            score_name="accuracy",
            global_step_transform=global_step_from_engine(trainer),
        )
        validation_evaluator.add_event_handler(Events.COMPLETED, best_handler)

        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})
        try:
            trainer.run(train_loader, max_epochs=MAX_EPOCHS)
        except:
            print("Error running training")

        return handler.last_checkpoint, best_handler.last_checkpoint

    last_good_checkpoint = None
    best_checkpoint = ""

    input_path_checkpoints = (
        "/home/israel/Mestrado/notebooks/synth-checkpoint-models"
    )
    output_path_checkpoint = (
        "/home/israel/Mestrado/notebooks/synth-broken-ckpts"
    )

    finished = False
    while not finished:
        last_good_checkpoint, best_checkpoint = run(last_good_checkpoint)
        try:
            torch.load(best_checkpoint)
        except:
            print("Best checkpoint is broken, running again")
        else:
            finished = True
        finally:
            if not finished:
                print("Finished?", finished)
                # move last good checkpoint to output path
                os.rename(
                    last_good_checkpoint,
                    os.path.join(
                        output_path_checkpoint,
                        os.path.basename(last_good_checkpoint),
                    ),
                )
                # remove all other checkpoints
                for f in os.listdir(input_path_checkpoints):
                    os.remove(os.path.join(input_path_checkpoints, f))
                last_good_checkpoint = os.path.join(
                    output_path_checkpoint,
                    os.path.basename(last_good_checkpoint),
                )
            import gc

            gc.collect()
