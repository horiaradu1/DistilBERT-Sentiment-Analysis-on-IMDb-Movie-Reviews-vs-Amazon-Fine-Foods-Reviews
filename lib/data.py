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


def build_collate(tokenize):
    def collate(batch):
        texts = [sample["text"] for sample in batch]
        labels = [sample["label"] for sample in batch]

        tokenized = tokenize(texts)
        return {**tokenized, "labels": torch.tensor(labels)}

    return collate


def load_imdb(val_ratio, seed, directory=None):
    """Load the IMDB dataset."""
    if directory is not None:
        os.makedirs(name=directory, exist_ok=True)
        dir_contents = os.listdir(directory)
        if all(f in dir_contents for f in ["train.json", "val.json", "test.json"]):
            print(f"Loading dataset from [{directory}]")
            try:
                with open(os.path.join(directory, "train.json"), "r") as f:
                    train = json.load(f)
                with open(os.path.join(directory, "val.json"), "r") as f:
                    val = json.load(f)
                with open(os.path.join(directory, "test.json"), "r") as f:
                    test = json.load(f)
                return train, val, test
            except Exception as e:
                print(f"Error loading dataset from [{directory}], downloading...")

    print("Downloading IMDb dataset...")
    directory = directory or "data/imdb"

    rng = np.random.default_rng(seed)

    imdb = datasets.load_dataset("imdb")

    original_train = imdb["train"]

    val_size = int(val_ratio * len(original_train))
    train_size = len(original_train) - val_size
    train_positive_size = train_size // 2

    train_labels = np.array([sample["label"] for sample in original_train])
    positive_indices = np.where(train_labels == 1)[0]
    negative_indices = np.where(train_labels == 0)[0]

    rng.shuffle(positive_indices)
    rng.shuffle(negative_indices)

    train_positive_indices = positive_indices[:train_positive_size]
    train_negative_indices = negative_indices[:train_positive_size]

    val_positive_indices = positive_indices[train_positive_size:]
    val_negative_indices = negative_indices[train_positive_size:]

    train_indices = np.concatenate(
        [train_positive_indices, train_negative_indices]
    ).tolist()
    val_indices = np.concatenate([val_positive_indices, val_negative_indices]).tolist()

    # Check for any leakage of the data
    assert (
        np.intersect1d(train_indices, val_indices, assume_unique=True).size == 0
    ), "Train indices leaking into val!"

    train = [original_train[i] for i in train_indices]
    val = [original_train[i] for i in val_indices]
    test = list(imdb["test"])

    with open(os.path.join(directory, "train.json"), "w") as f:
        json.dump(train, f)
    with open(os.path.join(directory, "val.json"), "w") as f:
        json.dump(val, f)
    with open(os.path.join(directory, "test.json"), "w") as f:
        json.dump(test, f)

    return train, val, test


class AmazonDataset(torch.utils.data.Dataset):
    """Amazon dataset."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_amazon(train_size, val_size, test_size, seed, directory=None):
    """Load the Amazon dataset."""
    if directory is not None:
        os.makedirs(name=directory, exist_ok=True)
        dir_contents = os.listdir(directory)
        if all(f in dir_contents for f in ["train.json", "val.json", "test.json"]):
            print(f"Loading dataset from [{directory}]")
            try:
                with open(os.path.join(directory, "train.json"), "r") as f:
                    train = json.load(f)
                with open(os.path.join(directory, "val.json"), "r") as f:
                    val = json.load(f)
                with open(os.path.join(directory, "test.json"), "r") as f:
                    test = json.load(f)
                return train, val, test
            except Exception as e:
                print(f"Error loading dataset from [{directory}]")

    print("Downloading Amazon dataset...")
    directory = directory or "data/amazon"

    train_positive_size = train_size // 2
    val_positive_size = val_size // 2
    test_positive_size = test_size // 2
    total_positive_samples = train_positive_size + val_positive_size + test_positive_size

    rng = np.random.default_rng(seed)

    response = requests.request(
        method="GET", url="https://snap.stanford.edu/data/finefoods.txt.gz"
    )

    gzip_raw = response.content

    decompressed = gzip.decompress(gzip_raw).decode("iso-8859-1")

    columns = [
        "product/productId",
        "review/userId",
        "review/profileName",
        "review/helpfulness",
        "review/score",
        "review/time",
        "review/summary",
        "review/text",
    ]

    reviews = []
    for record in decompressed.split("\n\n")[:-1]:
        values = [
            value.strip()
            for value in re.split(pattern=": |".join(columns), string=record)[1:]
        ]

        label = round(float(values[4]) / 5)
        text = values[-1]

        reviews.append({"label": label, "text": text})

    label_pool = np.array([review["label"] for review in reviews])

    positive_indices = np.where(label_pool == 1)[0]
    negative_indices = np.where(label_pool == 0)[0]

    positive_indices = rng.choice(
        positive_indices, size=total_positive_samples, replace=False
    )
    negative_indices = rng.choice(
        negative_indices, size=total_positive_samples, replace=False
    )

    train_positive_indices = positive_indices[:train_positive_size]
    train_negative_indices = negative_indices[:train_positive_size]

    val_positive_indices = positive_indices[
        train_positive_size : train_positive_size + val_positive_size
    ]
    val_negative_indices = negative_indices[
        train_positive_size : train_positive_size + val_positive_size
    ]

    test_positive_indices = positive_indices[
        train_positive_size
        + val_positive_size : train_positive_size
        + val_positive_size
        + test_positive_size
    ]
    test_negative_indices = negative_indices[
        train_positive_size
        + val_positive_size : train_positive_size
        + val_positive_size
        + test_positive_size
    ]

    train_indices = np.concatenate([train_positive_indices, train_negative_indices])
    val_indices = np.concatenate([val_positive_indices, val_negative_indices])
    test_indices = np.concatenate([test_positive_indices, test_negative_indices])

    # Check for any leakage of data
    assert (
        np.intersect1d(train_indices, test_indices, assume_unique=True).size == 0
    ), "Train leaking into test!"
    assert (
        np.intersect1d(train_indices, val_indices, assume_unique=True).size == 0
    ), "Train leaking into val!"
    assert (
        np.intersect1d(val_indices, test_indices, assume_unique=True).size == 0
    ), "Val leaking into test!"

    train_reviews = [reviews[idx] for idx in train_indices]
    val_reviews = [reviews[idx] for idx in val_indices]
    test_reviews = [reviews[idx] for idx in test_indices]

    with open(os.path.join(directory, "train.json"), "w") as f:
        json.dump(train_reviews, f)
    with open(os.path.join(directory, "val.json"), "w") as f:
        json.dump(val_reviews, f)
    with open(os.path.join(directory, "test.json"), "w") as f:
        json.dump(test_reviews, f)

    train = AmazonDataset(data=train_reviews)
    val = AmazonDataset(data=val_reviews)
    test = AmazonDataset(data=test_reviews)

    return train, val, test


def stats(train, val, test):
    """Prints the statistics of the dataset, positive and negative split plus train, test and val split."""
    train_pos, train_neg = np.bincount([sample["label"] for sample in train])
    val_pos, val_neg = np.bincount([sample["label"] for sample in val])
    test_pos, test_neg = np.bincount([sample["label"] for sample in test])

    print(
        f"Train ({len(train)})\n"
        f"| Positive: {train_pos} ({train_pos / len(train) * 100 :.2f}%)\n"
        f"| Negative: {train_neg} ({train_neg / len(train) * 100 :.2f}%)\n"
        f"Val ({len(val)})\n"
        f"| Positive: {val_pos} ({val_pos / len(val) * 100 :.2f}%)\n"
        f"| Negative: {val_neg} ({val_neg / len(val) * 100 :.2f}%)\n"
        f"Test ({len(test)})\n"
        f"| Positive: {test_pos} ({test_pos / len(test) * 100 :.2f}%)\n"
        f"| Negative: {test_neg} ({test_neg / len(test) * 100 :.2f}%)"
    )
