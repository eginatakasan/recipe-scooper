"""
Test labeling functions by category against real crawled HTML files or a live URL.

Usage:
    python -m labeling.test_lfs --category author
    python -m labeling.test_lfs --category ingredients --n 20
    python -m labeling.test_lfs --category steps --index 5
    python -m labeling.test_lfs --category all --url https://www.allrecipes.com/recipe/12345/
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path

import pandas as pd

from labeling.html_parser import parse as parse_html
from labeling.labeling_functions import (
    ABSTAIN,
    ALL_LFS,
    LABEL_NAMES,
    lf_author_a,
    lf_author_by_prefix,
    lf_author_classname,
    lf_author_itemprop,
    lf_author_rel,
    lf_author_schema,
    lf_author_schema_fuzzy,
    lf_author_xpath,
    lf_ingredient_kaggle,
    lf_ingredient_li_in_ul,
    lf_ingredient_measure,
    lf_ingredient_schema,
    lf_ingredient_xpath,
    lf_step_kaggle,
    lf_step_ol_li,
    lf_step_long_p,
    lf_step_schema,
    lf_step_xpath,
)
from labeling.schema_org import extract as extract_schema

CATEGORY_LFS = {
    "author": [
        lf_author_schema,
        lf_author_schema_fuzzy,
        lf_author_xpath,
        lf_author_classname,
        lf_author_a,
        lf_author_rel,
        lf_author_itemprop,
        lf_author_by_prefix,
    ],
    "ingredients": [
        lf_ingredient_measure,
        lf_ingredient_xpath,
        lf_ingredient_schema,
        lf_ingredient_kaggle,
        lf_ingredient_li_in_ul,
    ],
    "steps": [
        lf_step_xpath,
        lf_step_schema,
        lf_step_kaggle,
        lf_step_ol_li,
        lf_step_long_p,
    ],
    "all": ALL_LFS,
}

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_USER_AGENT = "Mozilla/5.0 (compatible; recipe-scooper-test/1.0)"


def _sanitize(text: str) -> str:
    return _CONTROL_RE.sub("", text)


def _fetch_url(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _build_node_df(html: str, kaggle_row: pd.Series) -> pd.DataFrame:
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
            "link_rel": node.link_rel or "",
            "itemprop": node.itemprop or "",
            "schema_name": schema.name or "",
            "schema_author": schema.author or "",
            "schema_date": schema.date_published or "",
            "schema_ingredients": "||".join(schema.ingredients),
            "schema_instructions": "||".join(schema.instructions),
            "kaggle_title": str(kaggle_row.get("title", "") or ""),
            "kaggle_ingredients": str(kaggle_row.get("ingredients", "") or ""),
            "kaggle_directions": str(kaggle_row.get("directions", "") or ""),
        })
    return pd.DataFrame(rows)


def _print_results(url: str, html: str, lfs: list) -> None:
    node_df = _build_node_df(html, pd.Series())
    if node_df.empty:
        print(f"--- {url} ---")
        print("  [SKIP] No nodes extracted\n")
        return

    fired_any = False
    for _, row in node_df.iterrows():
        results = [(lf.name, LABEL_NAMES[lf(row)]) for lf in lfs if lf(row) != ABSTAIN]
        if not results:
            continue
        if not fired_any:
            print(f"--- {url} ---")
            fired_any = True
        print(f"  xpath : {row['xpath']}")
        print(f"  tag   : <{row['tag']}>")
        print(f"  text  : {row['text'][:200]}")
        for lf_name, label in results:
            print(f"    -> {lf_name}: {label}")
        print()

    if not fired_any:
        print(f"--- {url} ---")
        print("  [INFO] No nodes fired for this category\n")


def run(
    category: str,
    crawl_log: Path,
    kaggle_csv: Path,
    n: int,
    index: int | None,
    url: str | None,
    file: Path | None,
) -> None:
    lfs = CATEGORY_LFS[category]

    if file is not None:
        print(f"=== LF Category: {category.upper()} | {len(lfs)} LFs | local file ===\n")
        try:
            html = file.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"[ERROR] Could not read file: {e}")
            return
        _print_results(str(file), html, lfs)
        return

    if url is not None:
        print(f"=== LF Category: {category.upper()} | {len(lfs)} LFs | fetching live URL ===\n")
        print(f"Fetching {url} ...")
        try:
            html = _fetch_url(url)
        except Exception as e:
            print(f"[ERROR] Could not fetch URL: {e}")
            return
        _print_results(url, html, lfs)
        return

    log_df = pd.read_csv(crawl_log)
    log_df = log_df[log_df["path"].notna() & (log_df["path"] != "")].reset_index(drop=True)

    if kaggle_csv.exists():
        kaggle_df = pd.read_csv(kaggle_csv).set_index("url")
    else:
        kaggle_df = pd.DataFrame()

    if index is not None:
        if index >= len(log_df):
            print(f"Error: --index {index} out of range (0–{len(log_df) - 1})")
            return
        sample = log_df.iloc[[index]]
    else:
        sample = log_df.sample(min(n, len(log_df)))

    print(f"=== LF Category: {category.upper()} | {len(lfs)} LFs | {len(sample)} file(s) ===\n")

    for _, log_row in sample.iterrows():
        row_url = log_row["url"]
        html_path = Path(log_row["path"])
        kaggle_row = kaggle_df.loc[row_url] if row_url in kaggle_df.index else pd.Series()
        try:
            html = html_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"--- {row_url} ---\n  [ERROR] {e}\n")
            continue
        node_df = _build_node_df(html, kaggle_row)
        if node_df.empty:
            print(f"--- {row_url} ---\n  [SKIP] No nodes extracted\n")
            continue

        fired_any = False
        for _, row in node_df.iterrows():
            results = [(lf.name, LABEL_NAMES[lf(row)]) for lf in lfs if lf(row) != ABSTAIN]
            if not results:
                continue
            if not fired_any:
                print(f"--- {row_url} ---")
                fired_any = True
            print(f"  xpath : {row['xpath']}")
            print(f"  tag   : <{row['tag']}>")
            print(f"  text  : {row['text'][:200]}")
            for lf_name, label in results:
                print(f"    -> {lf_name}: {label}")
            print()

        if not fired_any:
            print(f"--- {row_url} ---")
            print("  [INFO] No nodes fired for this category\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test LFs by category against crawled HTML files or a live URL")
    parser.add_argument(
        "--category",
        choices=list(CATEGORY_LFS.keys()),
        required=True,
        help="Which LF category to test",
    )
    parser.add_argument(
        "--crawl-log",
        default=Path("data/raw_html/crawl_log.csv"),
        type=Path,
    )
    parser.add_argument(
        "--kaggle-csv",
        default=Path("data/kaggle/sampled_urls.csv"),
        type=Path,
    )
    parser.add_argument(
        "--n",
        default=10,
        type=int,
        help="Number of random files to test (default: 10)",
    )
    parser.add_argument(
        "--index",
        default=None,
        type=int,
        help="Test a single file by its index in the crawl log (overrides --n)",
    )
    parser.add_argument(
        "--url",
        default=None,
        type=str,
        help="Fetch and test a live URL directly (overrides --n and --index)",
    )
    parser.add_argument(
        "--file",
        default=None,
        type=Path,
        help="Test a local .htm/.html file by path (overrides --url, --n, and --index)",
    )
    args = parser.parse_args()
    run(args.category, args.crawl_log, args.kaggle_csv, args.n, args.index, args.url, args.file)


if __name__ == "__main__":
    main()
