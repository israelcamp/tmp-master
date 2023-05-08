import collections
from PIL import Image, ImageFont, ImageDraw

import torch
from torch import nn
import torchvision as tv
import pytorch_lightning as pl
import numpy as np

from ctc import GreedyDecoder


def normalize_answer(s):
    """Lower text and remove extra whitespace."""

    def white_space_fix(text):
        return " ".join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


class CNNCTCPL(pl.LightningModule):
    def _average_key(self, outputs, key: str) -> torch.FloatTensor:
        return torch.stack([o[key] for o in outputs]).float().mean()

    def _concat_lists_by_key(self, outputs, key):
        return sum([o[key] for o in outputs], [])

    @staticmethod
    def get_ctc_logits(outputs):
        return outputs.permute(1, 0, 2).log_softmax(2)

    @staticmethod
    def get_lengths(attention_mask):
        lens = [
            row.argmin() if row.argmin() > 0 else len(row) for row in attention_mask
        ]
        return torch.tensor(lens)

    def _compute_logits(self, batch):
        images, _, _, attention_image = batch
        outputs = self.model(images, attention_image)
        return outputs

    def _handle_batch(self, batch):
        images, labels, attention_mask, attention_image = batch
        outputs = self.model(images, attention_image)

        logits = self.get_ctc_logits(outputs)
        lengths = self.get_lengths(attention_mask).type_as(images)

        if attention_image is not None:
            input_lengths = self.get_lengths(attention_image).type_as(images)
        else:
            input_lengths = torch.full(
                size=(logits.shape[1],), fill_value=logits.shape[0]
            ).type_as(images)

        loss = self.loss_fct(logits, labels, input_lengths.long(), lengths.long())
        return (loss,)

    def _handle_eval_batch(self, batch):
        outputs = self._handle_batch(batch)

        images, labels, _, attention_image = batch

        embeddings = self.model(images, attention_image)
        decoded_ids = embeddings.argmax(-1).squeeze()
        if len(decoded_ids.shape) == 1:
            decoded_ids = decoded_ids.unsqueeze(0)

        attention_image = (
            attention_image
            if attention_image is not None
            else len(decoded_ids) * [None]
        )
        decoded = [
            self.decoder(dec, att) for dec, att in zip(decoded_ids, attention_image)
        ]

        preds = self.tokenizer.batch_decode(decoded, skip_special_tokens=True)
        trues = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        outputs = outputs + (preds, trues)

        return outputs

    def _handle_eval_epoch_end(self, outputs, phase):
        loss_avg = self._average_key(outputs, f"{phase}_loss")
        originals = self._concat_lists_by_key(outputs, f"{phase}_trues")
        preds = self._concat_lists_by_key(outputs, f"{phase}_preds")

        exact_matches = []
        f1s = []
        for original, pred in zip(originals, preds):
            exact_matches.append(compute_exact(original, pred))
            f1s.append(compute_f1(original, pred))

        exact_match = np.array(exact_matches).mean()
        f1 = np.array(f1s).mean()
        return loss_avg, exact_match, f1

    ## FUNCTIONS NEEDED BY PYTORCH LIGHTNING ##

    def training_step(self, batch, batch_idx):
        outputs = self._handle_batch(batch)
        loss = outputs[0]

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        # single scheduler
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

        self.log("train_loss", loss, on_step=True, prog_bar=False, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, preds, trues = self._handle_eval_batch(batch)
        return {"val_loss": loss, "val_preds": preds, "val_trues": trues}

    def test_step(self, batch, batch_idx):
        loss, preds, trues = self._handle_eval_batch(batch)
        return {"test_loss": loss, "test_preds": preds, "test_trues": trues}

    def validation_epoch_end(self, outputs):
        loss_avg, em, f1 = self._handle_eval_epoch_end(outputs, phase="val")
        self.log("val_loss", loss_avg, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_em", em, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1", f1, on_epoch=True, prog_bar=True, logger=True)

    def test_epoch_end(self, outputs):
        loss_avg, em, f1 = self._handle_eval_epoch_end(outputs, phase="test")
        self.log("test_loss", loss_avg, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_em", em, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1", f1, on_epoch=True, prog_bar=True, logger=True)
        return {"test_loss": loss_avg}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.get_optimizer()
        return optimizer

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class TRTuner(CNNCTCPL):
    default_hparams = {
        "lr": 2e-4,
        "optimizer": "Adam",
        "optimizer_kwargs": {},
    }

    def __init__(self, model, tokenizer, decoder=None, hparams=None, **kwargs):
        self.model_kwargs = kwargs
        self.hparams.update(self._construct_hparams(hparams))

        super(CNNCTCPL, self).__init__()
        self.automatic_optimization = False

        self.model = model
        self.tokenizer = tokenizer
        self.decoder = decoder if decoder is not None else GreedyDecoder(0)
        self.loss_fct = nn.CTCLoss(blank=0, zero_infinity=True)

    def _construct_hparams(self, hparams):
        default_hparams = self.default_hparams.copy()

        if hparams is not None:
            default_hparams.update(hparams)

        default_hparams.update(self.model_kwargs)
        return default_hparams

    def get_optimizer(
        self,
    ) -> torch.optim.Optimizer:
        optimizer_name = self.hparams.optimizer
        lr = self.hparams.lr
        optimizer_hparams = self.hparams.optimizer_kwargs
        optimizer = getattr(torch.optim, optimizer_name)
        optimizer = optimizer(self.parameters(), lr=lr, **optimizer_hparams)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.hparams.steps, 1e-6
        )
        return [optimizer], [sch]

    @staticmethod
    def make_grid_image(images, texts):
        grid = tv.utils.make_grid(images, nrow=1)
        grid_gt = tv.transforms.ToPILImage()(grid)

        w, h = grid_gt.size

        font = ImageFont.truetype("Roboto-Black.ttf", size=16)
        img = Image.new(mode="RGB", size=(2 * w + 50, h), color=3 * (255,))
        draw = ImageDraw.Draw(img)
        img.paste(grid_gt)

        avgh = h // len(images)

        for i, t in enumerate(texts):
            draw.text((w + 10, i * avgh + avgh * 0.25), t, fill=3 * (0,), font=font)

        return img

    def validation_step(self, batch, batch_idx):
        loss, preds, trues = self._handle_eval_batch(batch)

        if batch_idx == 0:
            grid = self.make_grid_image(batch[0].cpu(), preds)
            try:
                self.logger.experiment.log_image("val_image", grid)
            except:
                pass

        return {"val_loss": loss, "val_preds": preds, "val_trues": trues}
