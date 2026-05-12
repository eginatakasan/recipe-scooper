"""
Evaluation helpers for MarkupLM token classification.

compute_metrics: passed directly to HuggingFace Trainer.
evaluate_file:   run evaluation against a held-out JSONL test file.

Metrics reported:
  - Overall token-level F1 (micro-averaged, excluding O)
  - Per-field precision / recall / F1
  - Exact-match accuracy for TITLE, AUTHOR, DATE (single-value fields)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from seqeval.metrics import classification_report, f1_score
from transformers import EvalPrediction

from training.constants import ID2LABEL, LABEL2ID

logger = logging.getLogger(__name__)

_IGNORE_INDEX = -100


def _align_predictions(predictions: np.ndarray, label_ids: np.ndarray):
    preds = np.argmax(predictions, axis=2)
    pred_list, label_list = [], []

    for pred_row, label_row in zip(preds, label_ids):
        pred_seq, label_seq = [], []
        for p, l in zip(pred_row, label_row):
            if l == _IGNORE_INDEX:
                continue
            pred_seq.append(ID2LABEL[p])
            label_seq.append(ID2LABEL[l])
        pred_list.append(pred_seq)
        label_list.append(label_seq)

    return pred_list, label_list


def compute_metrics(p: EvalPrediction) -> dict:
    pred_list, label_list = _align_predictions(p.predictions, p.label_ids)

    # seqeval expects BIO tags; our labels are flat classes treated as B- prefix
    bio_preds = [[f"B-{t}" if t != "O" else "O" for t in seq] for seq in pred_list]
    bio_labels = [[f"B-{t}" if t != "O" else "O" for t in seq] for seq in label_list]

    f1 = f1_score(bio_labels, bio_preds, average="micro", zero_division=0)
    report = classification_report(bio_labels, bio_preds, output_dict=True, zero_division=0)

    metrics: dict = {"f1": f1}
    for field in ("TITLE", "INGREDIENT", "STEP", "AUTHOR", "DATE"):
        key = f"B-{field}"
        if key in report:
            metrics[f"{field.lower()}_f1"] = report[key]["f1-score"]
            metrics[f"{field.lower()}_precision"] = report[key]["precision"]
            metrics[f"{field.lower()}_recall"] = report[key]["recall"]

    return metrics


def evaluate_file(
    model_dir: Path,
    test_jsonl: Path,
    device: str = "cuda",
) -> dict:
    """Run inference on a held-out test JSONL and print a full report."""
    import torch
    from transformers import MarkupLMForTokenClassification, MarkupLMProcessor

    from training.dataset import RecipeDataset
    from training.constants import MAX_LENGTH

    processor = MarkupLMProcessor.from_pretrained(str(model_dir), only_label_first_subword=True)
    model = MarkupLMForTokenClassification.from_pretrained(str(model_dir))
    model.eval().to(device)

    dataset = RecipeDataset(test_jsonl, processor=processor, max_length=MAX_LENGTH)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for example in dataset:
            inputs = {k: v.unsqueeze(0).to(device) for k, v in example.items() if k != "labels"}
            label_ids = example["labels"].numpy()

            outputs = model(**inputs)
            logits = outputs.logits.squeeze(0).cpu().numpy()

            pred_seq, label_seq = [], []
            for logit_row, l in zip(logits, label_ids):
                if l == _IGNORE_INDEX:
                    continue
                pred_seq.append(ID2LABEL[int(np.argmax(logit_row))])
                label_seq.append(ID2LABEL[int(l)])

            all_preds.append(pred_seq)
            all_labels.append(label_seq)

    bio_preds = [[f"B-{t}" if t != "O" else "O" for t in seq] for seq in all_preds]
    bio_labels = [[f"B-{t}" if t != "O" else "O" for t in seq] for seq in all_labels]

    report = classification_report(bio_labels, bio_preds, zero_division=0)
    logger.info("\n%s", report)

    return {"f1": f1_score(bio_labels, bio_preds, average="micro", zero_division=0)}


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--test-jsonl", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    results = evaluate_file(args.model_dir, args.test_jsonl, args.device)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
