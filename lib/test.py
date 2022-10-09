import torch
import numpy as np
import pytorch_lightning as pl
import transformers
import datasets
import torchmetrics
import re
import os
import json
import shutil
import requests
import gzip

from lib.data import build_collate
from models.distilbert import LitModel


def test(
    name,
    dataset,
    description,
    version,
    root_dir,
    test_dataset,
    batch_size,
    num_workers,
    max_sequence_length=256,
    padding="max_length",
    truncation=True,
):
    """
    Test DistilBERT model.
    """

    model_name = "distilbert-base-uncased"

    # Build tokenizer.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name, use_fast=True
    )
    tokenize = lambda text: tokenizer.batch_encode_plus(
        text,
        max_length=max_sequence_length,
        padding=padding,
        truncation=truncation,
        return_tensors="pt",
    )

    # Build collate function.
    collate = build_collate(tokenize=tokenize)

    # Build dataloaders.
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Build trainer.
    trainer = pl.Trainer(gpus=1, enable_checkpointing=False, logger=False)

    # Get the model with the best validation accuracy.
    checkpoints_path = os.path.join(root_dir, version, "checkpoints")
    checkpoints = os.listdir(checkpoints_path)
    best_idx = (
        np.array(
            [
                re.search(r"val_acc=(?P<acc>0\.\d{2})", checkpoint).group("acc")
                for checkpoint in checkpoints
            ]
        )
        .argmax()
        .tolist()
    )
    best_checkpoint_path = os.path.join(checkpoints_path, checkpoints[best_idx])

    # Load model from checkpoint.
    model = LitModel.load_from_checkpoint(checkpoint_path=best_checkpoint_path)

    # Evaluate model.
    metrics = trainer.test(model=model, dataloaders=test_dataloader, verbose=False)[0]

    print(f"Experiment: {name}\n", f"| Test accuracy: {metrics['test_acc']:2f}")

    return metrics
