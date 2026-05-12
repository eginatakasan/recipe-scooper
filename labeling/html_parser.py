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

        # Capture the closest enclosing link href, if any.
        href = ""
        link = element if element.name == "a" else element.find_parent("a")
        if link is not None:
            link_href = link.get("href")
            if isinstance(link_href, str):
                href = link_href.strip()

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
