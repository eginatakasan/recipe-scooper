"""
Runs the full weak supervision pipeline:
  1. Load crawl log + Kaggle CSV to get (html_path, labels) pairs
  2. Parse each HTML into HtmlNodes
  3. Apply Snorkel LFs to every node across all documents
  4. Fit a Snorkel LabelModel to combine LF votes
  5. Export high-confidence labeled nodes to JSONL for MarkupLM

Output JSONL format (one line per document):
    {
      "url": "...",
      "nodes": ["2 cups flour", ...],
      "xpaths": ["/html/body/...", ...],
      "tags": ["li", ...],
      "labels": ["INGREDIENT", "O", ...]
    }

Usage:
    python -m labeling.run_labeling \
        --crawl-log data/raw_html/crawl_log.csv \
        --kaggle-csv data/kaggle/sampled_urls.csv \
        --out data/labeled.jsonl \
        --confidence 0.7
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
from tqdm import tqdm

from labeling.html_parser import parse as parse_html
from labeling.labeling_functions import (
    ALL_LFS,
    ABSTAIN,
    LABEL_NAMES,
    O,
)
from labeling.schema_org import extract as extract_schema

logger = logging.getLogger(__name__)


import re as _re
_CONTROL_RE = _re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize(text: str) -> str:
    return _CONTROL_RE.sub("", text)


def _build_node_df(html: str, kaggle_row: pd.Series) -> pd.DataFrame:
    """Parse one HTML file into a node-level DataFrame ready for LF application."""
    schema = extract_schema(html)
    nodes = parse_html(html)
    if not nodes:
        return pd.DataFrame()

    rows = []
    for node in nodes:
        rows.append({
            "text": _sanitize(node.text),
            "xpath": node.xpath,
            "tag": node.tag,
            "class_name": node.class_name or "",
            "link_href": node.link_href or "",
            # Schema.org signals (join multi-value fields with || for fuzzy matching)
            "schema_name": schema.name or "",
            "schema_author": schema.author or "",
            "schema_date": schema.date_published or "",
            "schema_ingredients": "||".join(schema.ingredients),
            "schema_instructions": "||".join(schema.instructions),
            # Kaggle distant supervision signals
            "kaggle_title": str(kaggle_row.get("title", "") or ""),
            "kaggle_ingredients": str(kaggle_row.get("ingredients", "") or ""),
            "kaggle_directions": str(kaggle_row.get("directions", "") or ""),
        })

    return pd.DataFrame(rows)


def run(
    crawl_log: Path,
    kaggle_csv: Path,
    out_path: Path,
    confidence: float = 0.7,
) -> None:
    log_df = pd.read_csv(crawl_log)
    log_df = log_df[log_df["path"].notna() & (log_df["path"] != "")]  # skip failed crawls
    logger.info("%d successfully crawled pages", len(log_df))

    kaggle_df = pd.read_csv(kaggle_csv)
    kaggle_df = kaggle_df.set_index("url")

    applier = PandasLFApplier(lfs=ALL_LFS)

    # ── Pass 1: collect all node rows across all documents ────────────────────
    all_node_dfs: list[pd.DataFrame] = []
    doc_boundaries: list[tuple[int, int, str]] = []  # (start_idx, end_idx, url)
    cursor = 0

    for _, log_row in tqdm(log_df.iterrows(), total=len(log_df), desc="Parsing HTML"):
        url = log_row["url"]
        html_path = Path(log_row["path"])

        try:
            html = html_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        kaggle_row = kaggle_df.loc[url] if url in kaggle_df.index else pd.Series()
        node_df = _build_node_df(html, kaggle_row)
        if node_df.empty:
            continue

        all_node_dfs.append(node_df)
        doc_boundaries.append((cursor, cursor + len(node_df), url))
        cursor += len(node_df)

    if not all_node_dfs:
        logger.error("No nodes extracted — check crawl_log paths")
        return

    full_df = pd.concat(all_node_dfs, ignore_index=True)
    logger.info("Total nodes across all documents: %d", len(full_df))

    # ── Pass 2: apply all LFs at once ─────────────────────────────────────────
    logger.info("Applying %d labeling functions...", len(ALL_LFS))
    L = applier.apply(full_df)

    # ── Pass 3: fit label model ───────────────────────────────────────────────
    logger.info("Fitting LabelModel...")
    n_classes = max(LABEL_NAMES.keys()) + 1  # O=0 … DATE=5 → 6 classes
    label_model = LabelModel(cardinality=n_classes, verbose=True)
    label_model.fit(L, n_epochs=300, lr=0.01, log_freq=100, seed=42)

    probs = label_model.predict_proba(L)       # shape: (n_nodes, n_classes)
    preds = label_model.predict(L, tie_break_policy="abstain")

    # ── Pass 4: export per-document JSONL ─────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for start, end, url in doc_boundaries:
            doc_preds = preds[start:end]
            doc_probs = probs[start:end]
            doc_nodes = full_df.iloc[start:end]

            # Keep nodes where the label model is confident or voted O
            labels = []
            for i, (pred, prob_row) in enumerate(zip(doc_preds, doc_probs)):
                if pred == ABSTAIN:
                    labels.append("O")
                elif max(prob_row) >= confidence:
                    labels.append(LABEL_NAMES[pred])
                else:
                    labels.append("O")

            record = {
                "url": url,
                "nodes": doc_nodes["text"].tolist(),
                "xpaths": doc_nodes["xpath"].tolist(),
                "tags": doc_nodes["tag"].tolist(),
                "labels": labels,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Wrote %d labeled documents to %s", written, out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--crawl-log", default=Path("data/raw_html/crawl_log.csv"), type=Path)
    parser.add_argument("--kaggle-csv", default=Path("data/kaggle/sampled_urls.csv"), type=Path)
    parser.add_argument("--out", default=Path("data/labeled.jsonl"), type=Path)
    parser.add_argument("--confidence", default=0.7, type=float)
    args = parser.parse_args()
    run(args.crawl_log, args.kaggle_csv, args.out, args.confidence)


if __name__ == "__main__":
    main()
