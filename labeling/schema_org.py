"""
Extracts Recipe fields from schema.org JSON-LD metadata embedded in a page.
These are the strongest signals for labeling functions — treat them as ground truth
when present, since sites embed them for SEO and they're almost always accurate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class SchemaRecipe:
    name: Optional[str] = None
    author: Optional[str] = None
    date_published: Optional[str] = None
    ingredients: list[str] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    image: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not any([self.name, self.author, self.ingredients, self.instructions])


def _flatten_instructions(raw) -> list[str]:
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        steps = []
        for item in raw:
            if isinstance(item, str):
                steps.append(item)
            elif isinstance(item, dict):
                # HowToStep or HowToSection
                if item.get("@type") == "HowToSection":
                    for step in item.get("itemListElement", []):
                        if isinstance(step, dict):
                            steps.append(step.get("text", ""))
                else:
                    steps.append(item.get("text", item.get("name", "")))
        return [s for s in steps if s]
    return []


def _extract_author_name(raw) -> Optional[str]:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return raw.get("name")
    if isinstance(raw, list) and raw:
        return _extract_author_name(raw[0])
    return None


def _extract_image_url(raw) -> Optional[str]:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return raw.get("url")
    if isinstance(raw, list) and raw:
        return _extract_image_url(raw[0])
    return None


def _parse_graph(data: dict) -> Optional[SchemaRecipe]:
    """Handle @graph arrays (common in Yoast SEO)."""
    graph = data.get("@graph", [])
    for node in graph:
        if isinstance(node, dict) and "Recipe" in str(node.get("@type", "")):
            return _parse_recipe_object(node)
    return None


def _parse_recipe_object(obj: dict) -> SchemaRecipe:
    return SchemaRecipe(
        name=obj.get("name"),
        author=_extract_author_name(obj.get("author")),
        date_published=obj.get("datePublished"),
        ingredients=obj.get("recipeIngredient", []),
        instructions=_flatten_instructions(obj.get("recipeInstructions", [])),
        image=_extract_image_url(obj.get("image")),
    )


def extract(html: str) -> SchemaRecipe:
    soup = BeautifulSoup(html, "lxml")
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue

        if isinstance(data, list):
            data = next((d for d in data if isinstance(d, dict)), {})

        type_ = str(data.get("@type", ""))
        if "Recipe" in type_:
            return _parse_recipe_object(data)

        result = _parse_graph(data)
        if result:
            return result

    return SchemaRecipe()
