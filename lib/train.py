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


def train(
    name,
    dataset,
    root_dir,
    train_dataset,
    val_dataset,
    seed=42,
    epochs=10,
    batch_size=32,
    num_workers=4,
    pretrained=False,
    max_sequence_length=256,
    padding="max_length",
    truncation=True,
    lr=5e-5,
    lr_decay_step_size=4,
    lr_decay_gamma=0.1,
    lr_decay_last_epoch=-1,
    weight_decay=0.0,
    vocab_size=30522,
    max_position_embeddings=512,
    sinusoidal_pos_embds=False,
    n_layers=6,
    n_heads=12,
    dim=768,
    hidden_dim=4 * 768,
    dropout=0.1,
    attention_dropout=0.1,
    activation="gelu",
    initializer_range=0.02,
    qa_dropout=0.1,
    seq_classif_dropout=0.2,
    pad_token_id=0,
    num_labels=2,
):
    """
    Train DistilBERT model.
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
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Build logger.
    logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir=root_dir, name="", version=name, default_hp_metric=False
    )

    # Build trainer.
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=epochs,
        logger=logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                filename="{epoch};{step};{train_loss:.2f};{val_loss:.2f};{val_acc:.2f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
            pl.callbacks.ModelCheckpoint(
                filename="{epoch};{step};{train_loss:.2f};{val_loss:.2f};{val_acc:.2f}"
            ),
        ],
    )

    # Load model.
    model = LitModel(
        lr=lr,
        lr_decay_step_size=lr_decay_step_size,
        lr_decay_gamma=lr_decay_gamma,
        lr_decay_last_epoch=lr_decay_last_epoch,
        weight_decay=weight_decay,
        pretrained=pretrained,
        max_sequence_length=max_sequence_length,
        padding=padding,
        truncation=truncation,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        sinusoidal_pos_embds=sinusoidal_pos_embds,
        n_layers=n_layers,
        n_heads=n_heads,
        dim=dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        activation=activation,
        initializer_range=initializer_range,
        qa_dropout=qa_dropout,
        seq_classif_dropout=seq_classif_dropout,
        pad_token_id=pad_token_id,
        num_labels=num_labels,
    )

    # Seed everything.
    pl.seed_everything(seed, workers=True)

    # Train model.
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    return model
