"""Markdown rendering helpers for frontend-controlled HTML output."""

from __future__ import annotations

import bleach
import markdown as md_lib
from markupsafe import Markup

_ALLOWED_TAGS: list[str] = [
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "ul",
    "ol",
    "li",
    "strong",
    "em",
    "code",
    "pre",
    "blockquote",
    "hr",
    "a",
    "br",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
]

_ALLOWED_ATTRS: dict[str, list[str]] = {
    "a": ["href", "title", "rel"],
    "code": ["class"],
    "pre": ["class"],
    "th": ["scope"],
    "td": ["colspan", "rowspan"],
}

_ALLOWED_PROTOCOLS: list[str] = ["http", "https", "mailto"]


def markdown_to_safe_html(text: object) -> Markup:
    """Convert markdown text to sanitized HTML safe for Jinja rendering."""
    if not text:
        return Markup("")

    raw_html = md_lib.markdown(
        str(text),
        extensions=["tables", "fenced_code", "nl2br"],
    )
    clean_html = bleach.clean(
        raw_html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRS,
        protocols=_ALLOWED_PROTOCOLS,
        strip=True,
    )
    return Markup(clean_html)
