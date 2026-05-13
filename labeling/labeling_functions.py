"""
Snorkel labeling functions (LFs) for recipe field extraction.

Each LF receives a pandas Series row with:
    text       str   — visible text of the HTML node
    xpath      str   — XPath of the node's element
    tag        str   — HTML tag name
    schema_*   str   — pre-joined schema.org values for fast lookup
    kaggle_*   str   — pre-joined Kaggle label values for distant supervision

Returns one of: TITLE, INGREDIENT, STEP, AUTHOR, DATE, O, ABSTAIN
"""

from __future__ import annotations

import re

from snorkel.labeling import LabelingFunction, labeling_function

# ── Label constants ────────────────────────────────────────────────────────────
ABSTAIN = -1
O = 0
TITLE = 1
INGREDIENT = 2
STEP = 3
AUTHOR = 4
DATE = 5

# ── Min Length constants ────────────────────────────────────────────────────────────
MIN_LENGTH_INGREDIENTS = 3
MIN_LENGTH_STEP = 20
MAX_LENGTH_AUTHOR = 30


# ── Helpers ───────────────────────────────────────────────────────────────────

_MEASURE_RE = re.compile(
    r"\b(\d+[\./]?\d*)\s*(cup|tbsp|tsp|tablespoon|teaspoon|oz|ounce|lb|pound|g|gram|kg|ml|liter|litre|pinch|dash|clove|slice|can|pkg|package)\b",
    re.IGNORECASE,
)

_DATE_RE = re.compile(
    r"\b(\d{4}[-/]\d{2}[-/]\d{2}|\w+ \d{1,2},? \d{4})\b",
    re.IGNORECASE,
)

_AUTHOR_XPATH_RE = re.compile(r"(author|byline|contributor)", re.IGNORECASE)
_AUTHOR_CLASS_RE = re.compile(
    r"(?:^|[\s_-])(author|contributor|byline|entry-author|recipe-author|author-name|author_name)(?:$|[\s_-])",
    re.IGNORECASE,
)
_AUTHOR_CLASS_NEG_RE = re.compile(r"(?:^|[\s_-])(comment|feedback|review)(?:$|[\s_-])", re.IGNORECASE)
_AUTHOR_ABOUT_HREF_RE = re.compile(
    r"(?:^|/)(about(?:[-_/]?me)?)(?:/|$)|/author(?:s)?(?:/|$)|/profile(?:/|$)",
    re.IGNORECASE,
)
_INGREDIENT_XPATH_RE = re.compile(r"ingredient", re.IGNORECASE)
_INSTRUCTION_XPATH_RE = re.compile(r"instruction|direction|step|method|preparation", re.IGNORECASE)


def _fuzzy_match(text: str, candidates: str, threshold: int = 80) -> bool:
    from rapidfuzz import fuzz
    if not candidates:
        return False
    for candidate in candidates.split("||"):
        candidate = candidate.strip()
        if not candidate:
            continue
        if fuzz.partial_ratio(text.lower(), candidate.lower()) >= threshold:
            return True
    return False


def _is_button_in_ol_li(x) -> bool:
    """True when the node is a `<button>` nested under an ordered-list item (`ol > li`)."""
    if getattr(x, "tag", None) != "button":
        return False
    xpath = (getattr(x, "xpath", "") or "").lower()
    if "ol" not in xpath or "li" not in xpath:
        return False
    # Best-effort "ol before li" ordering check on the XPath string.
    return xpath.find("ol") != -1 and xpath.find("li") != -1 and xpath.find("ol") < xpath.find("li")


_NON_NAME_WORDS = frozenset({
    "read", "more", "view", "see", "all", "show", "hide", "share", "print",
    "save", "jump", "click", "here", "back", "next", "prev", "previous",
    "home", "menu", "close", "open", "search", "login", "signup", "subscribe",
})


def _looks_like_author_name(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if t.lower().startswith("by "):
        t = t[3:].strip()
    if len(t) < 3 or len(t) > MAX_LENGTH_AUTHOR:
        return False
    if any(ch.isdigit() for ch in t):
        return False
    # reject label-style strings like "Author:" or "Written by:"
    if ":" in t:
        return False
    if not re.search(r"[A-Za-z]", t):
        return False
    # reject common UI action phrases like "Read More", "View All"
    words = t.lower().split()
    if all(w in _NON_NAME_WORDS for w in words):
        return False
    return True


# ── TITLE LFs ─────────────────────────────────────────────────────────────────

@labeling_function()
def lf_title_h1(x):
    """Label page titles; reject non-`<h1>` nodes.

    - Filters out: all nodes that are not the primary heading.
    - How: checks `x.tag == "h1"`.
    - Parameters: none (binary tag check).
    """
    return TITLE if x.tag == "h1" else ABSTAIN


@labeling_function()
def lf_title_schema(x):
    """Label titles that exactly match schema.org `name`; reject near-misses/other text.

    - Filters out: nodes whose visible text differs from `schema_name` (case/whitespace aside).
    - How: exact normalized string equality between `x.text` and `x.schema_name`.
    - Parameters: none (no fuzzy matching; intentionally strict).
    """
    if x.schema_name and x.text.strip().lower() == x.schema_name.strip().lower():
        return TITLE
    return ABSTAIN


@labeling_function()
def lf_title_kaggle(x):
    """Label titles that exactly match the Kaggle distant label; reject everything else.

    - Filters out: nodes that aren't exactly the Kaggle-provided title string.
    - How: exact normalized string equality between `x.text` and `x.kaggle_title`.
    - Parameters: none (no fuzzy matching; intentionally strict).
    """
    if x.kaggle_title and x.text.strip().lower() == x.kaggle_title.strip().lower():
        return TITLE
    return ABSTAIN


@labeling_function()
def lf_title_page_title(x):
    """Label the document `<title>` element as TITLE; reject other tags.

    - Filters out: non-`<title>` nodes.
    - How: checks `x.tag == "title"`.
    - Parameters: none.
    """
    return TITLE if x.tag == "title" else ABSTAIN


# ── INGREDIENT LFs ────────────────────────────────────────────────────────────

@labeling_function()
def lf_ingredient_measure(x):
    """Label ingredient-like lines containing measurement units; reject others.

    - Filters out: text without an amount + unit pattern (e.g. navigation, headings, ads).
    - How: regex `_MEASURE_RE` searches for a number plus a known unit token.
    - Parameters: `_MEASURE_RE` vocabulary (units) and numeric pattern.
    """
    if _is_button_in_ol_li(x):
        return ABSTAIN
    return INGREDIENT if len(x.text) > MIN_LENGTH_INGREDIENTS and _MEASURE_RE.search(x.text) else ABSTAIN


@labeling_function()
def lf_ingredient_xpath(x):
    """Label nodes whose XPath contains ingredient keywords; reject other paths.

    - Filters out: nodes outside ingredient containers/sections.
    - How: regex `_INGREDIENT_XPATH_RE` searches `x.xpath` for "ingredient".
    - Parameters: `_INGREDIENT_XPATH_RE` pattern.
    """
    if _is_button_in_ol_li(x):
        return ABSTAIN
    return INGREDIENT if len(x.text) > MIN_LENGTH_INGREDIENTS and _INGREDIENT_XPATH_RE.search(x.xpath) else ABSTAIN


@labeling_function()
def lf_ingredient_schema(x):
    """Label nodes that fuzzy-match schema.org `recipeIngredient`; reject non-matches.

    - Filters out: nodes not similar enough to any schema ingredient string.
    - How: `_fuzzy_match(x.text, x.schema_ingredients)` compares against `||`-joined candidates
      using RapidFuzz `partial_ratio`.
    - Parameters: `_fuzzy_match` default `threshold=80` (match must be >= 80).
    """
    if _is_button_in_ol_li(x):
        return ABSTAIN
    return INGREDIENT if len(x.text) > MIN_LENGTH_INGREDIENTS and _fuzzy_match(x.text, x.schema_ingredients) else ABSTAIN


@labeling_function()
def lf_ingredient_kaggle(x):
    """Label nodes that fuzzy-match Kaggle ingredient strings; reject non-matches.

    - Filters out: nodes not similar enough to any Kaggle ingredient candidate.
    - How: `_fuzzy_match(x.text, x.kaggle_ingredients)` over `||`-separated candidates.
    - Parameters: `_fuzzy_match` default `threshold=80`.
    """
    if _is_button_in_ol_li(x):
        return ABSTAIN
    return INGREDIENT if len(x.text) > MIN_LENGTH_INGREDIENTS and _fuzzy_match(x.text, x.kaggle_ingredients) else ABSTAIN


@labeling_function()
def lf_ingredient_li_in_ul(x):
    """Label list items in unordered lists that look like measured ingredients.

    - Filters out: list items outside `<ul>` containers and items without measurement tokens.
    - How: requires `x.tag == "li"`, `"ul" in x.xpath`, and `_MEASURE_RE` match in `x.text`.
    - Parameters: `_MEASURE_RE` (units/number pattern).
    """
    if _is_button_in_ol_li(x):
        return ABSTAIN
    if x.tag == "li" and "ul" in x.xpath and len(x.text) > MIN_LENGTH_INGREDIENTS and _MEASURE_RE.search(x.text):
        return INGREDIENT
    return ABSTAIN


# ── STEP LFs ──────────────────────────────────────────────────────────────────

@labeling_function()
def lf_step_xpath(x):
    """Label nodes located in instruction/direction containers; reject other sections.

    - Filters out: nodes whose XPath doesn't mention instruction-like keywords.
    - How: regex `_INSTRUCTION_XPATH_RE` searches `x.xpath` for terms like instruction/step.
    - Parameters: `_INSTRUCTION_XPATH_RE` pattern.
    """
    if _is_button_in_ol_li(x):
        return ABSTAIN
    return len(x.text) > MIN_LENGTH_STEP and STEP if _INSTRUCTION_XPATH_RE.search(x.xpath) else ABSTAIN


@labeling_function()
def lf_step_schema(x):
    """Label nodes that fuzzy-match schema.org instructions and are long enough to be steps.

    - Filters out: short fragments (e.g. "Step 1") and text that doesn't match schema steps.
    - How: `_fuzzy_match(x.text, x.schema_instructions, threshold=75)` AND
      `len(x.text) > MIN_LENGTH_STEP`.
    - Parameters: `threshold=75` (looser than ingredients), `MIN_LENGTH_STEP`.
    """
    if _is_button_in_ol_li(x):
        return ABSTAIN
    return STEP if len(x.text) > MIN_LENGTH_STEP and _fuzzy_match(x.text, x.schema_instructions, threshold=75)  else ABSTAIN


@labeling_function()
def lf_step_kaggle(x):
    """Label nodes that fuzzy-match Kaggle directions and are long enough to be steps.

    - Filters out: short fragments and non-matching nodes.
    - How: `_fuzzy_match(x.text, x.kaggle_directions, threshold=75)` AND
      `len(x.text) > MIN_LENGTH_STEP`.
    - Parameters: `threshold=75`, `MIN_LENGTH_STEP`.
    """
    if _is_button_in_ol_li(x):
        return ABSTAIN
    return STEP if len(x.text) > MIN_LENGTH_STEP and _fuzzy_match(x.text, x.kaggle_directions, threshold=75) > MIN_LENGTH_STEP else ABSTAIN


@labeling_function()
def lf_step_ol_li(x):
    """Label ordered-list items as steps when they look like full instructions.

    - Filters out: `<li>` nodes outside ordered lists and short list items.
    - How: requires `x.tag == "li"`, `"ol" in x.xpath`, and `len(x.text) > MIN_LENGTH_STEP`.
    - Parameters: `MIN_LENGTH_STEP`.
    """
    if _is_button_in_ol_li(x):
        return ABSTAIN
    if x.tag == "li" and "ol" in x.xpath and len(x.text) > MIN_LENGTH_STEP:
        return STEP
    return ABSTAIN


@labeling_function()
def lf_step_long_p(x):
    """Label long paragraphs as steps inside instruction-like containers.

    - Filters out: short paragraphs and paragraphs outside instruction/direction sections.
    - How: requires `x.tag == "p"`, `len(x.text) > 80`, and `_INSTRUCTION_XPATH_RE` match in
      `x.xpath`.
    - Parameters: length cutoff `80`, `_INSTRUCTION_XPATH_RE` pattern.
    """
    if _is_button_in_ol_li(x):
        return ABSTAIN
    if x.tag == "p" and len(x.text) > 80 and _INSTRUCTION_XPATH_RE.search(x.xpath):
        return STEP
    return ABSTAIN


# ── AUTHOR LFs ────────────────────────────────────────────────────────────────

@labeling_function()
def lf_author_schema(x):
    """Label authors that exactly match schema.org author string; reject other text.

    - Filters out: nodes whose text differs from `schema_author`.
    - How: exact normalized string equality between `x.text` and `x.schema_author`.
    - Parameters: none (strict match to reduce false positives).
    """
    if x.schema_author and x.text.strip().lower() == x.schema_author.strip().lower():
        return AUTHOR
    return ABSTAIN


@labeling_function()
def lf_author_xpath(x):
    """Label likely author/byline nodes based on XPath keywords and short text length.

    - Filters out: non-author sections and long content blocks that contain "author" tokens.
    - How: `_AUTHOR_XPATH_RE` match in `x.xpath` AND `_looks_like_author_name(x.text)` AND
      `len(x.text) <= MAX_LENGTH_AUTHOR`.
    - Parameters: `_AUTHOR_XPATH_RE` pattern, `_looks_like_author_name` heuristics,
      `MAX_LENGTH_AUTHOR`.
    """
    text = getattr(x, "text", "") or ""
    xpath = getattr(x, "xpath", "") or ""
    return AUTHOR if _AUTHOR_XPATH_RE.search(xpath) and _looks_like_author_name(text) and len(text) <= MAX_LENGTH_AUTHOR else ABSTAIN


@labeling_function()
def lf_author_by_prefix(x):
    """Label nodes whose visible text is a 'By Name' byline pattern.

    - Filters out: nodes not starting with 'by ', single-word suffixes (e.g. 'By Email'),
      and text that doesn't look like a name.
    - How: strips 'by ' prefix, requires >= 2 tokens, then `_looks_like_author_name`.
    - Parameters: `_looks_like_author_name` heuristics.
    """
    text = (getattr(x, "text", "") or "").strip().replace(" ", " ")
    if not text.lower().startswith("by "):
        return ABSTAIN
    name_part = text[3:].strip()
    if len(name_part.split()) < 2:
        return ABSTAIN
    return AUTHOR if _looks_like_author_name(name_part) else ABSTAIN


@labeling_function()
def lf_author_schema_fuzzy(x):
    """Label nodes that fuzzy-match the schema.org author name.

    - Filters out: nodes longer than MAX_LENGTH_AUTHOR and non-matches.
    - How: `rapidfuzz.partial_ratio` >= 85 between node text and `schema_author`.
    - Parameters: threshold 85, `MAX_LENGTH_AUTHOR` length cap.
    """
    from rapidfuzz import fuzz
    schema_author = (getattr(x, "schema_author", "") or "").strip()
    if not schema_author:
        return ABSTAIN
    text = (getattr(x, "text", "") or "").strip()
    if not text or len(text) > MAX_LENGTH_AUTHOR:
        return ABSTAIN
    return AUTHOR if fuzz.partial_ratio(text.lower(), schema_author.lower()) >= 85 else ABSTAIN


@labeling_function()
def lf_author_rel(x):
    """Label nodes inside an <a rel='author'> link.

    - Filters out: nodes not enclosed in a link with rel=author, and text > 60 chars.
    - How: checks 'author' in `x.link_rel` AND 2 <= len(text) <= 60.
    - Parameters: none (rel=author is a strong semantic signal; no name heuristic needed).
    """
    if "author" not in (getattr(x, "link_rel", "") or "").lower():
        return ABSTAIN
    text = (getattr(x, "text", "") or "").strip()
    return AUTHOR if 2 <= len(text) <= 60 else ABSTAIN


@labeling_function()
def lf_author_itemprop(x):
    """Label nodes with itemprop='author' microdata markup.

    - Filters out: nodes without itemprop=author and text > 60 chars.
    - How: checks 'author' in `x.itemprop` AND 2 <= len(text) <= 60.
    - Parameters: none (itemprop=author is a strong semantic signal; no name heuristic needed).
    """
    if "author" not in (getattr(x, "itemprop", "") or "").lower():
        return ABSTAIN
    text = (getattr(x, "text", "") or "").strip()
    return AUTHOR if 2 <= len(text) <= 60 else ABSTAIN


@labeling_function()
def lf_author_classname(x):
    """Label author nodes when the element class mentions author/contributor.

    - Filters out: nodes without an author-ish class and nodes with non-name-like text.
    - How: `_AUTHOR_CLASS_RE` match in `x.class_name` AND `_looks_like_author_name(x.text)`.
    - Parameters: `_AUTHOR_CLASS_RE` pattern, `_looks_like_author_name` heuristics.
    """
    class_name = (getattr(x, "class_name", "") or "")
    if _AUTHOR_CLASS_RE.search(class_name) and _AUTHOR_CLASS_NEG_RE.search(class_name):
        return ABSTAIN
    return AUTHOR if _AUTHOR_CLASS_RE.search(class_name) and _looks_like_author_name(getattr(x, "text", "")) else ABSTAIN

@labeling_function()
def lf_author_a(x):
    """Label author nodes when the element is a link to about pages.

    - Filters out: nodes not inside an `<a href="...">` and links that don't look like author/about pages.
    - How: `_AUTHOR_ABOUT_HREF_RE` match in `x.link_href` AND `_looks_like_author_name(x.text)`.
    - Parameters: `_AUTHOR_ABOUT_HREF_RE` pattern, `_looks_like_author_name` heuristics.
    """
    href = (getattr(x, "link_href", "") or "").strip()
    text = getattr(x, "text", "") or ""
    if not href:
        return ABSTAIN
    if not _AUTHOR_ABOUT_HREF_RE.search(href):
        return ABSTAIN
    return AUTHOR if _looks_like_author_name(text) else ABSTAIN

# ── DATE LFs ──────────────────────────────────────────────────────────────────

@labeling_function()
def lf_date_schema(x):
    """Label dates that contain the schema.org date string; reject other text.

    - Filters out: nodes that don't include `schema_date` (or have no schema date).
    - How: substring check `x.schema_date in x.text`.
    - Parameters: none (relies on schema-provided date value).
    """
    if x.schema_date and x.schema_date in x.text:
        return DATE
    return ABSTAIN


@labeling_function()
def lf_date_time_tag(x):
    """Label `<time>` elements as DATE; reject other tags.

    - Filters out: non-`<time>` nodes.
    - How: checks `x.tag == "time"`.
    - Parameters: none.
    """
    return DATE if x.tag == "time" else ABSTAIN


@labeling_function()
def lf_date_pattern(x):
    """Label short strings that look like dates; reject long/ambiguous matches.

    - Filters out: long text blocks (where a date-like substring may appear incidentally) and
      text that doesn't match a date pattern.
    - How: regex `_DATE_RE` match in `x.text` AND `len(x.text) < 60`.
    - Parameters: `_DATE_RE` pattern, max length cutoff `60`.
    """
    return DATE if _DATE_RE.search(x.text) and len(x.text) < 60 else ABSTAIN


# ── All LFs (order matters for the label matrix columns) ─────────────────────

ALL_LFS: list[LabelingFunction] = [
    lf_title_h1,
    lf_title_schema,
    lf_title_kaggle,
    lf_title_page_title,
    lf_ingredient_measure,
    lf_ingredient_xpath,
    lf_ingredient_schema,
    lf_ingredient_kaggle,
    lf_ingredient_li_in_ul,
    lf_step_xpath,
    lf_step_schema,
    lf_step_kaggle,
    lf_step_ol_li,
    lf_step_long_p,
    lf_author_schema,
    lf_author_schema_fuzzy,
    lf_author_xpath,
    lf_author_classname,
    lf_author_a,
    lf_author_rel,
    lf_author_itemprop,
    lf_author_by_prefix,
    lf_date_schema,
    lf_date_time_tag,
    lf_date_pattern,
]

LABEL_NAMES = {ABSTAIN: "ABSTAIN", O: "O", TITLE: "TITLE", INGREDIENT: "INGREDIENT", STEP: "STEP", AUTHOR: "AUTHOR", DATE: "DATE"}
