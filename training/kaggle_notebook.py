# ============================================================
# Recipe Scooper — MarkupLM Fine-tuning (Kaggle Notebook)
# Run each cell sequentially. GPU accelerator must be ON.
# Upload this file + data/labeled.jsonl to your Kaggle dataset.
# ============================================================

# ── Cell 1: Install dependencies ────────────────────────────
# %pip install -q transformers==4.41.2 datasets==2.20.0 seqeval==1.2.2 accelerate==0.31.0 rapidfuzz==3.9.3

# ── Cell 2: Imports & constants ─────────────────────────────
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score
from transformers import (
    MarkupLMForTokenClassification,
    MarkupLMProcessor,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

LABEL2ID = {"O": 0, "TITLE": 1, "INGREDIENT": 2, "STEP": 3, "AUTHOR": 4, "DATE": 5}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MODEL_NAME = "microsoft/markuplm-base"
MAX_LENGTH = 512
LABELED_JSONL = Path("/kaggle/input/recipe-scooper-labeled/labeled.jsonl")
OUTPUT_DIR = Path("/kaggle/working/markuplm-recipe")

# ── Cell 3: Dataset class ────────────────────────────────────
class RecipeDataset(Dataset):
    def __init__(self, lines, processor):
        self.processor = processor
        self.examples = []
        for line in lines:
            record = json.loads(line)
            nodes, xpaths = record["nodes"], record["xpaths"]
            node_labels = [LABEL2ID.get(lbl, 0) for lbl in record["labels"]]
            if not nodes:
                continue
            enc = processor(
                nodes=nodes,
                xpaths=xpaths,
                node_labels=node_labels,
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.examples.append({k: v.squeeze(0) for k, v in enc.items()})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# ── Cell 4: Load data & split ────────────────────────────────
random.seed(42)
lines = [l for l in LABELED_JSONL.read_text().splitlines() if l.strip()]
random.shuffle(lines)

split = max(1, int(len(lines) * 0.1))
val_lines, train_lines = lines[:split], lines[split:]
print(f"Train docs: {len(train_lines)} | Val docs: {len(val_lines)}")

processor = MarkupLMProcessor.from_pretrained(MODEL_NAME, only_label_first_subword=True)
train_ds = RecipeDataset(train_lines, processor)
val_ds   = RecipeDataset(val_lines,   processor)
print(f"Train examples: {len(train_ds)} | Val examples: {len(val_ds)}")

# ── Cell 5: Load model ───────────────────────────────────────
model = MarkupLMForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL2ID),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

# ── Cell 6: Metrics ──────────────────────────────────────────
IGNORE = -100

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2)
    pred_list, label_list = [], []
    for pred_row, label_row in zip(preds, p.label_ids):
        ps, ls = [], []
        for pred, label in zip(pred_row, label_row):
            if label == IGNORE:
                continue
            ps.append(f"B-{ID2LABEL[pred]}" if ID2LABEL[pred] != "O" else "O")
            ls.append(f"B-{ID2LABEL[label]}" if ID2LABEL[label] != "O" else "O")
        pred_list.append(ps)
        label_list.append(ls)
    f1 = f1_score(label_list, pred_list, average="micro")
    report = classification_report(label_list, pred_list, output_dict=True)
    metrics = {"f1": f1}
    for field in ("TITLE", "INGREDIENT", "STEP", "AUTHOR", "DATE"):
        key = f"B-{field}"
        if key in report:
            metrics[f"{field.lower()}_f1"] = report[key]["f1-score"]
    return metrics

# ── Cell 7: Train ────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to="none",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

# ── Cell 8: Save ─────────────────────────────────────────────
trainer.save_model(str(OUTPUT_DIR))
processor.save_pretrained(str(OUTPUT_DIR))
print(f"Model saved to {OUTPUT_DIR}")

# ── Cell 9: Final eval report ────────────────────────────────
results = trainer.evaluate()
print(json.dumps(results, indent=2))
