"""
Samples URLs from the Kaggle recipe dataset.
Stratifies by domain so both train and test sets cover diverse site structures.

Expected Kaggle CSV columns (from wilmerarltstrmberg/recipe-dataset-over-2m):
    link, site, title, ingredients, directions, ...

Usage:
    python -m crawler.sampler \
        --csv data/kaggle/recipes.csv \
        --out data/kaggle/sampled_urls.csv \
        --test-out data/kaggle/test_urls.csv \
        --per-domain 100 \
        --top-domains 100 \
        --test-fraction 0.15
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"link", "site", "title", "ingredients", "directions"}


def sample(
    csv_path: Path,
    out_path: Path,
    per_domain: int = 100,
    top_domains: int = 100,
    test_fraction: float = 0.15,
    test_out_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, test_df). Splits are stratified by domain so every
    domain contributes proportionally to both sets.
    """
    logger.info("Loading %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info("Loaded %d rows. Columns: %s", len(df), list(df.columns))

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")

    before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    logger.info("After dropna: %d rows (dropped %d)", len(df), before - len(df))

    df["domain"] = df["site"].str.removeprefix("www.")
    before = len(df)
    df = df[df["domain"] != ""]
    logger.info("After domain filter: %d rows (dropped %d with empty domain)", len(df), before - len(df))

    df["url"] = df["link"].apply(
        lambda x: x if str(x).startswith("http") else f"https://{x}"
    )

    if df.empty:
        logger.error("No rows remain after filtering")
        return df, df

    domain_counts = df["domain"].value_counts()
    logger.info("Found %d unique domains. Top 5: %s", len(domain_counts), domain_counts.head(5).to_dict())
    top = domain_counts.head(top_domains).index.tolist()

    # Per domain: sample total budget, then split into train / test
    test_per_domain = max(1, int(per_domain * test_fraction))
    train_per_domain = per_domain - test_per_domain

    train_rows, test_rows = [], []

    for _, group in df[df["domain"].isin(top)].groupby("domain"):
        shuffled = group.sample(frac=1, random_state=42)
        n = len(shuffled)
        n_test = min(test_per_domain, n)
        n_train = min(train_per_domain, n - n_test)
        test_rows.append(shuffled.iloc[:n_test])
        train_rows.append(shuffled.iloc[n_test: n_test + n_train])

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)

    # Sanity check: no URL overlap between splits
    overlap = set(train_df["url"]) & set(test_df["url"])
    assert not overlap, f"Train/test overlap detected: {len(overlap)} URLs"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_path, index=False)
    logger.info("Train: %d URLs → %s", len(train_df), out_path)

    if test_out_path is not None:
        test_out_path.parent.mkdir(parents=True, exist_ok=True)
        test_df.to_csv(test_out_path, index=False)
        logger.info("Test:  %d URLs → %s", len(test_df), test_out_path)

    return train_df, test_df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", default=Path("data/kaggle/sampled_urls.csv"), type=Path)
    parser.add_argument("--test-out", default=Path("data/kaggle/test_urls.csv"), type=Path)
    parser.add_argument("--per-domain", default=100, type=int)
    parser.add_argument("--top-domains", default=100, type=int)
    parser.add_argument("--test-fraction", default=0.15, type=float,
                        help="Fraction of per-domain sample reserved for test (default: 0.15)")
    args = parser.parse_args()
    sample(args.csv, args.out, args.per_domain, args.top_domains, args.test_fraction, args.test_out)


if __name__ == "__main__":
    main()
