"""Report renderers package for the Data Forecaster backend.

Renderers consume a populated :class:`ExecutiveReport` model and produce
output in a specific format.  The LLM never generates final output —
renderers iterate the structured model section-by-section.
"""

from __future__ import annotations

from report.renderers.html_renderer import HTMLRenderer
from report.renderers.markdown_renderer import MarkdownRenderer

__all__ = [
    "HTMLRenderer",
    "MarkdownRenderer",
]
