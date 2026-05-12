"""
Fine-tunes MarkupLMForTokenClassification on the labeled recipe JSONL.
Designed to run on Kaggle Notebooks (single GPU, ~30 GB disk).

Usage:
    python -m training.train \
        --labeled data/labeled.jsonl \
        --output models/markuplm-recipe \
        --epochs 5 \
        --batch-size 8
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from transformers import (
    MarkupLMForTokenClassification,
    MarkupLMProcessor,
    Trainer,
    TrainingArguments,
)

from training.constants import ID2LABEL, LABEL2ID, MODEL_NAME
from training.dataset import load_splits
from training.evaluate import compute_metrics

logger = logging.getLogger(__name__)


def train(
    labeled_path: Path,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 4,
    grad_accum: int = 2,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
) -> None:
    logger.info("Loading and splitting dataset from %s", labeled_path)
    train_ds, val_ds = load_splits(labeled_path, val_fraction=0.1)
    logger.info("Train: %d examples | Val: %d examples", len(train_ds), len(val_ds))

    logger.info("Loading %s", MODEL_NAME)
    model = MarkupLMForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=True,          # free perf on Kaggle T4/P100
        report_to="none",   # no wandb on Kaggle by default
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving best model to %s", output_dir)
    trainer.save_model(str(output_dir))

    processor = MarkupLMProcessor.from_pretrained(MODEL_NAME, only_label_first_subword=True)
    processor.save_pretrained(str(output_dir))

    logger.info("Training complete.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled", default=Path("data/labeled.jsonl"), type=Path)
    parser.add_argument("--output", default=Path("models/markuplm-recipe"), type=Path)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--grad-accum", default=2, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    args = parser.parse_args()
    train(args.labeled, args.output, args.epochs, args.batch_size, args.grad_accum, args.lr)


if __name__ == "__main__":
    main()
