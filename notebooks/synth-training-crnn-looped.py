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
        nw = max(nw, 100)

        image = image.resize((nw, self.height), Image.BICUBIC)

        return image, label

    def __getitem__(self, idx):
        image_file = self.image_files[idx]

        try:
            image, label = self.read_image_file_and_label(image_file)
        except:
            print(f"Error reading image {image_file} idx {idx}")
            return self.__getitem__(random.randint(0, len(self.image_files) - 1))

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
        max_width = self.max_width if self.max_width is not None else max(image_widths)

        attention_images = []
        for w in image_widths:
            attention_images.append([1] * w + [0] * (max_width - w))
        attention_images = self.pooler(torch.tensor(attention_images).float()).long()

        h = images[0].size[1]
        to_tensor = tv.transforms.ToTensor()
        images = [to_tensor(self.expand_image(im, h=h, w=max_width)) for im in images]

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


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    def __init__(self, imgH=32, nc=3, nclass=100, nh=256, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, "imgH has to be a multiple of 16"

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module(
                "conv{0}".format(i), nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i])
            )
            if batchNormalization:
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module("relu{0}".format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module("relu{0}".format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module("pooling{0}".format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module("pooling{0}".format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, False)
        convRelu(3)
        cnn.add_module(
            "pooling{0}".format(2), nn.MaxPool2d((2, 1), (2, 1))
        )  # , (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5, True)
        cnn.add_module(
            "pooling{0}".format(3), nn.MaxPool2d((2, 1), (2, 1))
        )  # , (0, 1)))  # 512x2x16
        convRelu(6, False)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh), BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, input, *args, **kwargs):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # return conv.permute(0, 2, 1)
        # rnn features
        output = self.rnn(conv)

        return output.permute(1, 0, 2)


def get_preds_from_logits(logits, attention_image, labels, decoder, tokenizer):
    decoded_ids = logits.argmax(-1).squeeze(0)
    if len(decoded_ids.shape) == 1:
        decoded_ids = decoded_ids.unsqueeze(0)

    decoded = [decoder(dec, att) for dec, att in zip(decoded_ids, attention_image)]
    y_pred = tokenizer.batch_decode(decoded, skip_special_tokens=True)
    y = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return y_pred, y


def train_step(
    engine,
    batch,
    model,
    optimizer,
    criterion,
    device,
):
    model.train()

    images, labels, attention_mask, attention_image = [
        x.to(device) if x is not None else x for x in batch
    ]

    logits = model(images)

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

    optimizer.step()
    optimizer.zero_grad()
    model.zero_grad()

    return loss.item()


def val_step(engine, batch, model, device, decoder, tokenizer):
    model.eval()
    images, labels, _, attention_image = [
        x.to(device) if x is not None else x for x in batch
    ]
    with torch.no_grad():
        logits = model(images)

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

        model = CRNN(nc=1, nclass=tokenizer.vocab_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _ = model.to(device)

        MAX_EPOCHS = 2
        STEPS = len(train_loader) * MAX_EPOCHS

        optimizer = torch.optim.Adadelta(model.parameters(), rho=0.9)
        criterion = torch.nn.CTCLoss(blank=tokenizer.pad_token_id, zero_infinity=True)

        partial_train_step = partial(
            train_step,
            model=model,
            optimizer=optimizer,
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
            "trainer": trainer,
        }
        handler = Checkpoint(
            to_save,
            "crnn-synth-checkpoint-models",
            n_saved=4,
        )
        if last_good_checkpoint is not None:
            print("Loading last good checkpoint", last_good_checkpoint)
            checkpoint = torch.load(last_good_checkpoint)
            Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=10_000), handler)

        best_to_save = {"model": model}
        best_handler = Checkpoint(
            best_to_save,
            "crnn-synth-checkpoint-models",
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
        "/home/israel/Mestrado/notebooks/crnn-synth-checkpoint-models"
    )
    output_path_checkpoint = "/home/israel/Mestrado/notebooks/crnn-synth-broken-ckpts"

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
