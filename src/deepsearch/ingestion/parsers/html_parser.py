"""HTML document parser — strips markup, extracts clean text."""
from __future__ import annotations

import re
from pathlib import Path

from .base import BaseParser, ParsedDocument

# Tags whose entire content (tag + children) should be removed
_DROP_TAGS = re.compile(
    r"<(script|style|noscript|svg|iframe|head)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)

# All remaining HTML tags
_ALL_TAGS = re.compile(r"<[^>]+>")

# Collapse whitespace runs (spaces, tabs, multiple blank lines)
_WHITESPACE = re.compile(r" {2,}|\t+")
_BLANK_LINES = re.compile(r"\n{3,}")

# Decode common HTML entities
_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&quot;": '"', "&#39;": "'", "&apos;": "'",
    "&nbsp;": " ", "&mdash;": "—", "&ndash;": "–",
    "&hellip;": "…", "&copy;": "©", "&reg;": "®",
}

# Extract page title
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _decode_entities(text: str) -> str:
    for entity, char in _ENTITIES.items():
        text = text.replace(entity, char)
    # Numeric entities &#NNN; and &#xHHH;
    text = re.sub(r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)), text)
    text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
    return text


def _html_to_text(html: str) -> str:
    # 1. Remove script/style/etc. blocks
    html = _DROP_TAGS.sub(" ", html)
    # 2. Replace block-level tags with newlines for natural paragraph breaks
    html = re.sub(
        r"</(p|div|article|section|h[1-6]|li|tr|blockquote|pre)[^>]*>",
        "\n", html, flags=re.IGNORECASE,
    )
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    # 3. Strip all remaining tags
    html = _ALL_TAGS.sub("", html)
    # 4. Decode entities
    html = _decode_entities(html)
    # 5. Normalise whitespace
    html = _WHITESPACE.sub(" ", html)
    html = _BLANK_LINES.sub("\n\n", html)
    return html.strip()


class HTMLParser(BaseParser):
    """Parses HTML/HTM files into plain text."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".html", ".htm", ".xhtml"]

    def parse(self, path: Path, file_id: str) -> ParsedDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")

        # Try to extract a title
        title_match = _TITLE_RE.search(raw)
        title = _html_to_text(title_match.group(1)) if title_match else path.stem

        text = _html_to_text(raw)

        return ParsedDocument(
            file_id=file_id,
            source_path=str(path),
            text=text,
            metadata={
                "title": title,
                "format": "html",
                "file_name": path.name,
            },
        )
