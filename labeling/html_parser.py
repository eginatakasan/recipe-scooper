"""
Parses an HTML page into a flat list of (text, xpath) pairs — one per visible text node.
This is the input format MarkupLM expects.

Skips: script, style, noscript, head, meta, link tags (invisible/non-content nodes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from bs4 import BeautifulSoup, NavigableString, Tag

_SKIP_TAGS = {"script", "style", "noscript", "head", "meta", "link", "svg", "iframe"}
_MAX_NODES = 512  # MarkupLM practical limit per document


@dataclass
class HtmlNode:
    text: str
    xpath: str
    tag: str
    class_name: str = ""
    link_href: str = ""
    link_rel: str = ""
    itemprop: str = ""


def _build_xpath(element: Tag) -> str:
    parts: list[str] = []
    for ancestor in reversed(list(element.parents)):
        if not isinstance(ancestor, Tag) or ancestor.name in ("[document]", "html"):
            continue
        siblings = [s for s in ancestor.parent.children if isinstance(s, Tag) and s.name == ancestor.name]
        idx = siblings.index(ancestor) + 1
        parts.append(f"{ancestor.name}[{idx}]")
    parts.append(f"{element.name}[1]")
    return "/" + "/".join(parts)


def parse(html: str, max_nodes: int = _MAX_NODES) -> list[HtmlNode]:
    soup = BeautifulSoup(html, "lxml")

    # Remove invisible/boilerplate subtrees entirely
    for tag in soup.find_all(_SKIP_TAGS):
        tag.decompose()

    nodes: list[HtmlNode] = []

    for element in soup.find_all(True):
        if element.name in _SKIP_TAGS:
            continue

        class_attr = element.get("class")
        if isinstance(class_attr, list):
            class_name = " ".join([c for c in class_attr if isinstance(c, str)]).strip()
        elif isinstance(class_attr, str):
            class_name = class_attr.strip()
        else:
            class_name = ""

        # Capture the closest enclosing link href and rel, if any.
        href = ""
        link_rel = ""
        link = element if element.name == "a" else element.find_parent("a")
        if link is not None:
            raw_href = link.get("href")
            if isinstance(raw_href, str):
                href = raw_href.strip()
            raw_rel = link.get("rel", [])
            link_rel = " ".join(raw_rel) if isinstance(raw_rel, list) else str(raw_rel or "")

        # Capture itemprop for microdata author/name signals.
        raw_itemprop = element.get("itemprop", "")
        itemprop = " ".join(raw_itemprop) if isinstance(raw_itemprop, list) else str(raw_itemprop or "")

        for child in element.children:
            if not isinstance(child, NavigableString):
                continue
            text = child.strip()
            if not text:
                continue
            xpath = _build_xpath(element)
            nodes.append(
                HtmlNode(
                    text=text,
                    xpath=xpath,
                    tag=element.name,
                    class_name=class_name,
                    link_href=href,
                    link_rel=link_rel,
                    itemprop=itemprop,
                )
            )

            if len(nodes) >= max_nodes:
                return nodes

    return nodes


def og_image(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")
    tag = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "og:image"})
    if tag and tag.get("content"):
        return tag["content"]
    return None
