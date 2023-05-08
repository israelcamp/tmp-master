import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from transformers import AutoTokenizer

from datamodule import SROIETask2DataModule
from tuner import TRTuner
from model import TextRecognitionModel
from ctc import GreedyDecoder


if __name__ == "__main__":
    pl.seed_everything(0)

    max_epochs = 30

    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")

    dm = SROIETask2DataModule(
        root_dir="SROIETask2/data/",
        label_file="SROIETask2/data.json",
        tokenizer=tokenizer,
        height=32,
        num_workers=8,
        train_bs=16,
        valid_bs=64,
        max_width=None,
        do_pool=True,
    )

    dm.setup("fit")
    steps = len(dm.train_dataloader()) * max_epochs

    model = TRTuner(
        TextRecognitionModel(vocab_size=tokenizer.vocab_size),
        tokenizer,
        GreedyDecoder(0),
        {
            "lr": 1e-4,
            "optimizer": "Adam",
            "steps": steps,
        },
    )

    FAST_DEV_RUN = True
    trainer_params = {
        "max_epochs": max_epochs,
        # "gpus": [1],
        # "accelerator": "cpu",
        "log_every_n_steps": 1,
        "fast_dev_run": FAST_DEV_RUN,
        # "log_gpu_memory": True,
    }

    if not FAST_DEV_RUN:
        neptune_logger = NeptuneLogger(
            project_name="i155825/SROIETask2Ablation",
            experiment_name="Deberta",  # Optional,
            tags=["cnn", "ctc", "deberta", "augmentations", "attention-images"],
        )
        neptune_logger.log_artifact("dataset.py")
        neptune_logger.log_artifact("datamodule.py")
        neptune_logger.log_artifact("model.py")
        neptune_logger.log_artifact("tuner.py")
        neptune_logger.log_artifact("ctc.py")

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filename=f"{neptune_logger.version}"
            + "{epoch}-{val_loss:.4f}-{val_em:.4f}-{val_f1:.4f}",
            monitor="val_f1",
            mode="max",
            save_top_k=1,
        )

        trainer_params.update(
            {
                "logger": neptune_logger,
                "callbacks": [checkpoint_callback],
            }
        )

    trainer = pl.Trainer(**trainer_params)

    trainer.fit(model, dm)
