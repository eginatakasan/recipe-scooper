"""
PyTorch Dataset that converts labeled JSONL → MarkupLM model inputs.

Each document becomes one (or more, if truncated) training examples.
Label alignment: the first subword token of each node gets the node label;
all continuation subwords and special tokens get -100 (ignored in loss).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import MarkupLMProcessor

from training.constants import LABEL2ID, MAX_LENGTH, MODEL_NAME


class RecipeDataset(Dataset):
    def __init__(
        self,
        jsonl_path: Path,
        processor: Optional[MarkupLMProcessor] = None,
        max_length: int = MAX_LENGTH,
    ) -> None:
        self.processor = processor or MarkupLMProcessor.from_pretrained(
            MODEL_NAME, only_label_first_subword=True
        )
        self.processor.parse_html = False
        self.max_length = max_length
        self.examples: list[dict] = []
        self._load(jsonl_path)

    def _load(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                nodes = record["nodes"]
                xpaths = record["xpaths"]
                str_labels = record["labels"]

                if not nodes:
                    continue

                node_labels = [LABEL2ID.get(lbl, 0) for lbl in str_labels]

                encoding = self.processor(
                    nodes=nodes,
                    xpaths=xpaths,
                    node_labels=node_labels,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                self.examples.append({
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "token_type_ids": encoding["token_type_ids"].squeeze(0),
                    "xpath_tags_seq": encoding["xpath_tags_seq"].squeeze(0),
                    "xpath_subs_seq": encoding["xpath_subs_seq"].squeeze(0),
                    "labels": encoding["labels"].squeeze(0),
                })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def load_splits(
    jsonl_path: Path,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[RecipeDataset, RecipeDataset]:
    """Split a single JSONL into train / val datasets."""
    import random

    random.seed(seed)

    lines = Path(jsonl_path).read_text(encoding="utf-8").splitlines()
    lines = [l for l in lines if l.strip()]
    random.shuffle(lines)

    split = max(1, int(len(lines) * val_fraction))
    val_lines, train_lines = lines[:split], lines[split:]

    def _write_tmp(lines_: list[str], suffix: str) -> Path:
        tmp = jsonl_path.parent / f"_tmp_{suffix}.jsonl"
        tmp.write_text("\n".join(lines_), encoding="utf-8")
        return tmp

    processor = MarkupLMProcessor.from_pretrained(MODEL_NAME, only_label_first_subword=True)
    processor.parse_html = False

    train_path = _write_tmp(train_lines, "train")
    val_path = _write_tmp(val_lines, "val")

    train_ds = RecipeDataset(train_path, processor=processor)
    val_ds = RecipeDataset(val_path, processor=processor)

    train_path.unlink()
    val_path.unlink()

    return train_ds, val_ds
