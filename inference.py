"""
Run recipe extraction on a URL using the fine-tuned MarkupLM model.

Usage:
    python inference.py --url https://www.allrecipes.com/recipe/... --model models/markuplm-recipe
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
import trafilatura
from transformers import MarkupLMForTokenClassification, MarkupLMProcessor

from labeling.html_parser import og_image, parse as parse_html
from labeling.schema_org import extract as extract_schema
from training.constants import ID2LABEL, MAX_LENGTH

logger = logging.getLogger(__name__)


def fetch_html(url: str) -> str | None:
    html = trafilatura.fetch_url(url)
    if html and len(html) >= 500:
        return html
    logger.warning("trafilatura returned short content, trying with longer timeout")
    html = trafilatura.fetch_url(url, config=trafilatura.settings.use_config())
    return html


def predict(
    html: str,
    model: MarkupLMForTokenClassification,
    processor: MarkupLMProcessor,
    device: str,
) -> list[tuple[str, str]]:
    """Returns list of (node_text, predicted_label) pairs."""
    nodes = parse_html(html)
    if not nodes:
        return []

    node_texts = [n.text for n in nodes]
    node_xpaths = [n.xpath for n in nodes]

    encoding = processor(
        nodes=node_texts,
        xpaths=node_xpaths,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Map each token back to its source node using word_ids
    word_ids = encoding.word_ids(batch_index=0)

    inputs = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(0)  # (seq_len, num_labels)

    preds = logits.argmax(dim=-1).cpu().tolist()

    # Take the prediction of the first subword token for each node
    node_preds: dict[int, str] = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in node_preds:
            continue
        node_preds[word_idx] = ID2LABEL[preds[token_idx]]

    return [(node_texts[i], node_preds.get(i, "O")) for i in range(len(node_texts))]


def build_output(url: str, html: str, node_predictions: list[tuple[str, str]]) -> dict:
    grouped: dict[str, list[str]] = defaultdict(list)
    for text, label in node_predictions:
        if label != "O":
            grouped[label].append(text)

    schema = extract_schema(html)

    return {
        "url": url,
        "title": grouped["TITLE"][0] if grouped["TITLE"] else (schema.name or None),
        "author": grouped["AUTHOR"][0] if grouped["AUTHOR"] else (schema.author or None),
        "date": grouped["DATE"][0] if grouped["DATE"] else (schema.date_published or None),
        "ingredients": grouped["INGREDIENT"] or schema.ingredients or [],
        "steps": grouped["STEP"] or schema.instructions or [],
        "images": {
            "primary": og_image(html),
            "body": [],
        },
        "confidence": {
            "title": 1.0 if grouped["TITLE"] else 0.0,
            "ingredients": min(1.0, len(grouped["INGREDIENT"]) / 3) if grouped["INGREDIENT"] else 0.0,
            "steps": min(1.0, len(grouped["STEP"]) / 3) if grouped["STEP"] else 0.0,
            "author": 1.0 if grouped["AUTHOR"] else 0.0,
            "date": 1.0 if grouped["DATE"] else 0.0,
        },
        "extraction_status": (
            "complete" if all([grouped["TITLE"], grouped["INGREDIENT"], grouped["STEP"]])
            else "partial" if any([grouped["TITLE"], grouped["INGREDIENT"], grouped["STEP"]])
            else "failed"
        ),
    }


def run(url: str, model_dir: Path, device: str) -> dict:
    logger.info("Fetching %s", url)
    html = fetch_html(url)
    if not html:
        return {"url": url, "extraction_status": "failed", "error": "Could not fetch page"}

    logger.info("Loading model from %s", model_dir)
    processor = MarkupLMProcessor.from_pretrained(str(model_dir), only_label_first_subword=True)
    processor.parse_html = False
    model = MarkupLMForTokenClassification.from_pretrained(str(model_dir))
    model.eval().to(device)

    logger.info("Running inference...")
    node_predictions = predict(html, model, processor, device)

    result = build_output(url, html, node_predictions)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="models/markuplm-recipe", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    result = run(args.url, args.model, args.device)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
