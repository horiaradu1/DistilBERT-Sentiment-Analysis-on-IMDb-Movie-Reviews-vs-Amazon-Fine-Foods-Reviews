import sys
import argparse
import torch
import transformers

from models.distilbert import LitModel


def main():
    parser = argparse.ArgumentParser(description="NLU Sentiment Classification demo.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to model checkpoint.",
        required=True,
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=256,
        help="Maximum sequence length.",
        required=False,
    )
    parser.add_argument(
        "--padding",
        type=str,
        default="max_length",
        help="Padding type.",
        required=False,
    )
    parser.add_argument(
        "--truncation",
        type=bool,
        default=True,
        help="Truncation.",
        required=False,
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model.
    model = LitModel.load_from_checkpoint(checkpoint_path=args.checkpoint).to(device)

    # Build tokenizer.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="distilbert-base-uncased", use_fast=True
    )
    tokenize = lambda text: tokenizer.encode_plus(
        text,
        max_length=args.max_sequence_length,
        padding=args.padding,
        truncation=args.truncation,
        return_tensors="pt",
    ).to(device)

    while True:
        text = input("Enter text: ")
        tokenized = tokenize(text)
        outputs = model(**tokenized)
        logits = outputs[0]
        probs = torch.softmax(logits, dim=1)
        label = torch.argmax(probs).item()
        sentiment = "positive" if label == 1 else "negative"
        print(f"Sentiment: {sentiment}\n")


if __name__ == "__main__":
    main()
