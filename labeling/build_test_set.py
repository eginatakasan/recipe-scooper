"""
Builds a gold test.jsonl using schema.org as automatic ground truth.
Only keeps pages where schema.org data is present and has at least
ingredients + instructions — these are high-confidence labels requiring
no manual annotation.

Label logic (deterministic, no Snorkel):
  - Node text exactly/fuzzy matches schema.org name        → TITLE
  - Node text fuzzy matches a schema.org recipeIngredient  → INGREDIENT
  - Node text fuzzy matches a schema.org recipeInstruction → STEP
  - Node text matches schema.org author                    → AUTHOR
  - Node tag is <time> or text matches datePublished       → DATE
  - Everything else                                        → O

Usage:
    python -m labeling.build_test_set \
        --crawl-log data/raw_html/test_crawl_log.csv \
        --out data/test.jsonl \
        --min-fields 2
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

from labeling.html_parser import parse as parse_html
from labeling.schema_org import extract as extract_schema

logger = logging.getLogger(__name__)

_FUZZY_THRESHOLD = 80


def _fuzzy_in(text: str, candidates: list[str]) -> bool:
    t = text.lower().strip()
    for c in candidates:
        if fuzz.partial_ratio(t, c.lower().strip()) >= _FUZZY_THRESHOLD:
            return True
    return False


def _label_node(text: str, tag: str, schema) -> str:
    if schema.name and text.lower().strip() == schema.name.lower().strip():
        return "TITLE"
    if schema.ingredients and _fuzzy_in(text, schema.ingredients):
        return "INGREDIENT"
    if schema.instructions and _fuzzy_in(text, schema.instructions):
        return "STEP"
    if schema.author and text.lower().strip() == schema.author.lower().strip():
        return "AUTHOR"
    if tag == "time":
        return "DATE"
    if schema.date_published and schema.date_published in text:
        return "DATE"
    return "O"


def build(
    crawl_log: Path,
    out_path: Path,
    min_fields: int = 2,
) -> None:
    log_df = pd.read_csv(crawl_log)
    log_df = log_df[log_df["path"].notna() & (log_df["path"] != "")]
    logger.info("%d crawled pages to process", len(log_df))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(log_df.iterrows(), total=len(log_df), desc="Building test set"):
            url = row["url"]
            try:
                html = Path(row["path"]).read_text(encoding="utf-8", errors="replace")
            except OSError:
                skipped += 1
                continue

            schema = extract_schema(html)

            # Count how many fields schema.org provides
            fields_present = sum([
                bool(schema.name),
                bool(schema.ingredients),
                bool(schema.instructions),
                bool(schema.author),
                bool(schema.date_published),
            ])
            if fields_present < min_fields:
                skipped += 1
                continue

            nodes = parse_html(html)
            if not nodes:
                skipped += 1
                continue

            labels = [_label_node(n.text, n.tag, schema) for n in nodes]

            # Skip pages where nothing was labeled (schema.org present but no matches)
            if all(l == "O" for l in labels):
                skipped += 1
                continue

            record = {
                "url": url,
                "nodes": [n.text for n in nodes],
                "xpaths": [n.xpath for n in nodes],
                "tags": [n.tag for n in nodes],
                "labels": labels,
                "schema_fields": fields_present,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Written: %d | Skipped (no schema.org): %d", written, skipped)
    logger.info("Test set saved to %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--crawl-log", default=Path("data/raw_html/test_crawl_log.csv"), type=Path)
    parser.add_argument("--out", default=Path("data/test.jsonl"), type=Path)
    parser.add_argument("--min-fields", default=2, type=int,
                        help="Minimum schema.org fields required to include a page")
    args = parser.parse_args()
    build(args.crawl_log, args.out, args.min_fields)


if __name__ == "__main__":
    main()
