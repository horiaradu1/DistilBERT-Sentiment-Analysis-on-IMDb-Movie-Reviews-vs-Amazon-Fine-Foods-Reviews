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


class LitModel(pl.LightningModule):
    """
    A wrapper around the DistilBERT model sourced from HuggingFace.
    """

    def __init__(
        self,
        lr,
        lr_decay_step_size,
        lr_decay_gamma,
        lr_decay_last_epoch,
        weight_decay,
        pretrained=False,
        max_sequence_length=256,
        padding="max_length",
        truncation=True,
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
        super().__init__()

        model_name = "distilbert-base-uncased"

        self.save_hyperparameters()

        if self.hparams.pretrained:
            config = transformers.AutoConfig.from_pretrained(
                pretrained_model_name_or_path=model_name,
                num_labels=self.hparams.num_labels,
            )
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=model_name, config=config
            )
        else:
            config = transformers.DistilBertConfig(
                max_sequence_length=self.hparams.max_sequence_length,
                padding=self.hparams.padding,
                truncation=self.hparams.truncation,
                vocab_size=self.hparams.vocab_size,
                max_position_embeddings=self.hparams.max_position_embeddings,
                sinusoidal_pos_embds=self.hparams.sinusoidal_pos_embds,
                n_layers=self.hparams.n_layers,
                n_heads=self.hparams.n_heads,
                dim=self.hparams.dim,
                hidden_dim=self.hparams.hidden_dim,
                dropout=self.hparams.dropout,
                attention_dropout=self.hparams.attention_dropout,
                activation=self.hparams.activation,
                initializer_range=self.hparams.initializer_range,
                qa_dropout=self.hparams.qa_dropout,
                seq_classif_dropout=self.hparams.seq_classif_dropout,
                pad_token_id=self.hparams.pad_token_id,
                num_labels=self.hparams.num_labels,
            )
            model = transformers.DistilBertForSequenceClassification(config=config)

        self.model = model
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, **inputs):
        return self.model(**inputs)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            params=optimizer_grouped_parameters, lr=self.hparams.lr
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=self.hparams.lr_decay_step_size,
            gamma=self.hparams.lr_decay_gamma,
            last_epoch=self.hparams.lr_decay_last_epoch,
        )
        return [optimizer], [scheduler]

    def _step(self, batch, batch_idx, kind):
        outputs = self(**batch)
        loss = outputs[0]
        preds = outputs[1]
        self.log(f"{kind}_loss", loss)
        return {"loss": loss, "y_hat": preds.detach(), "y": batch["labels"]}

    def training_step(self, batch, batch_idx):
        return self._step(batch=batch, batch_idx=batch_idx, kind="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch=batch, batch_idx=batch_idx, kind="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch=batch, batch_idx=batch_idx, kind="test")

    def _epoch_end(self, outputs, kind):
        y_hat = torch.cat([output["y_hat"] for output in outputs])
        y = torch.cat([output["y"] for output in outputs])

        accuracy = self.accuracy(y_hat, y)

        self.log(f"{kind}_acc", accuracy)

        return {f"{kind}_accuracy": accuracy}

    def validation_epoch_end(self, outputs):
        return self._epoch_end(outputs=outputs, kind="val")

    def test_epoch_end(self, outputs):
        return self._epoch_end(outputs=outputs, kind="test")
