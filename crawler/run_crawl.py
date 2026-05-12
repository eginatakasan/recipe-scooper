"""
Crawls URLs from a sampled CSV and saves raw HTML to disk.
Optionally crawls a test CSV into a separate output directory in the same run.

Output layout:
    <out>/crawl_log.csv
    <out>/<domain>/<url_hash>.html
    <test-out>/crawl_log.csv          (only if --test-csv is provided)
    <test-out>/<domain>/<url_hash>.html

Usage:
    python -m crawler.run_crawl \
        --csv data/kaggle/sampled_urls.csv \
        --out data/raw_html \
        --test-csv data/kaggle/test_urls.csv \
        --test-out data/raw_html_test \
        --concurrency 5 \
        --delay 1.0
"""

import argparse
import asyncio
import csv
import hashlib
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm.asyncio import tqdm

from crawler.fetch import FetchResult, fetch

logger = logging.getLogger(__name__)


def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


async def _crawl_one(
    url: str,
    domain: str,
    out_dir: Path,
    delay: float,
) -> dict:
    await asyncio.sleep(delay)
    result: FetchResult = await fetch(url)

    row = {
        "url": url,
        "domain": domain,
        "path": "",
        "used_playwright": result.used_playwright,
        "error": result.error or "",
        "status_code": result.status_code or "",
    }

    if result.ok:
        domain_dir = out_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        html_path = domain_dir / f"{_url_hash(url)}.html"
        html_path.write_text(result.html, encoding="utf-8", errors="replace")
        row["path"] = str(html_path)

    return row


def _load_pending(csv_path: Path, log_path: Path) -> list[tuple[str, str]]:
    df = pd.read_csv(csv_path)
    urls = list(zip(df["url"], df["domain"]))

    crawled: set[str] = set()
    if log_path.exists():
        existing = pd.read_csv(log_path)
        crawled = set(existing["url"].tolist())
        logger.info("Resuming %s — %d already crawled", csv_path.name, len(crawled))

    return [(url, domain) for url, domain in urls if url not in crawled]


async def _crawl_batch(
    pending: list[tuple[str, str]],
    out_dir: Path,
    log_path: Path,
    concurrency: int,
    delay: float,
    label: str,
) -> None:
    if not pending:
        logger.info("%s: nothing to crawl", label)
        return

    logger.info("%s: %d URLs to crawl → %s", label, len(pending), out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)

    async def bounded(url: str, domain: str) -> dict:
        async with sem:
            return await _crawl_one(url, domain, out_dir, delay)

    write_header = not log_path.exists()
    fieldnames = ["url", "domain", "path", "used_playwright", "status_code", "error"]

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        tasks = [bounded(url, domain) for url, domain in pending]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=label):
            row = await coro
            writer.writerow(row)
            f.flush()

    logger.info("%s: done. Log → %s", label, log_path)


async def run(
    csv_path: Path,
    out_dir: Path,
    concurrency: int,
    delay: float,
    test_csv_path: Optional[Path] = None,
    test_out_dir: Optional[Path] = None,
) -> None:
    train_log = out_dir / "crawl_log.csv"
    train_pending = _load_pending(csv_path, train_log)
    await _crawl_batch(train_pending, out_dir, train_log, concurrency, delay, label="Train")

    if test_csv_path and test_out_dir:
        test_log = test_out_dir / "crawl_log.csv"
        test_pending = _load_pending(test_csv_path, test_log)
        await _crawl_batch(test_pending, test_out_dir, test_log, concurrency, delay, label="Test")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=Path("data/kaggle/sampled_urls.csv"), type=Path)
    parser.add_argument("--out", default=Path("data/raw_html"), type=Path)
    parser.add_argument("--test-csv", default=Path("data/kaggle/test_urls.csv"), type=Path)
    parser.add_argument("--test-out", default=Path("data/raw_html_test"), type=Path)
    parser.add_argument("--concurrency", default=5, type=int)
    parser.add_argument("--delay", default=1.0, type=float)
    args = parser.parse_args()
    asyncio.run(run(
        args.csv, args.out,
        args.concurrency, args.delay,
        args.test_csv, args.test_out,
    ))


if __name__ == "__main__":
    main()
