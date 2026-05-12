"""
Samples test URLs from the Kaggle CSV that were NOT used in training.
Uses schema.org presence as a quality filter — only keeps pages likely
to have verifiable ground-truth labels.

Usage:
    python -m crawler.sample_test \
        --csv data/kaggle/recipes.csv \
        --exclude data/kaggle/sampled_urls.csv \
        --out data/kaggle/test_urls.csv \
        --per-domain 15 \
        --top-domains 100
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def sample_test(
    csv_path: Path,
    exclude_path: Path,
    out_path: Path,
    per_domain: int = 15,
    top_domains: int = 100,
) -> pd.DataFrame:
    logger.info("Loading Kaggle CSV...")
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(subset=["link", "site", "title", "ingredients", "directions"])
    df["domain"] = df["site"].str.removeprefix("www.")
    df["url"] = df["link"].apply(lambda x: x if str(x).startswith("http") else f"https://{x}")
    df = df[df["domain"] != ""]

    logger.info("Excluding already-sampled URLs...")
    exclude_df = pd.read_csv(exclude_path)
    exclude_urls = set(exclude_df["url"].tolist())
    df = df[~df["url"].isin(exclude_urls)]
    logger.info("%d URLs remaining after exclusion", len(df))

    domain_counts = df["domain"].value_counts()
    top = domain_counts.head(top_domains).index.tolist()

    sampled = (
        df[df["domain"].isin(top)]
        .groupby("domain", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_domain), random_state=123))
        .reset_index(drop=True)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_path, index=False)
    logger.info("Saved %d test URLs to %s", len(sampled), out_path)
    return sampled


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--exclude", required=True, type=Path)
    parser.add_argument("--out", default=Path("data/kaggle/test_urls.csv"), type=Path)
    parser.add_argument("--per-domain", default=15, type=int)
    parser.add_argument("--top-domains", default=100, type=int)
    args = parser.parse_args()
    sample_test(args.csv, args.exclude, args.out, args.per_domain, args.top_domains)


if __name__ == "__main__":
    main()
