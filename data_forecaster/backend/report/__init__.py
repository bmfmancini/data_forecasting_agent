"""Structured executive report package for the Data Forecaster backend.

This package implements a two-stage reporting architecture:

1. **Stage 1 (Python)**: :class:`ExecutiveReportBuilder` computes all
   deterministic metrics (confidence score, dashboard, health indicators,
   data quality, prediction intervals, recommendations, evidence) and
   populates an :class:`ExecutiveReport` model.
2. **Stage 2 (LLM)**: :func:`generate_narratives` fills the ``narrative``
   text fields on each section using the LLM, receiving the structured
   model as context.  The LLM never computes scores or business metrics.

Dedicated renderers (:class:`MarkdownRenderer`, :class:`HTMLRenderer`,
:class:`PDFRenderer`) consume the populated :class:`ExecutiveReport` to
produce output formats — the LLM never generates final Markdown.
"""

from __future__ import annotations

from report.builder import ExecutiveReportBuilder
from report.models import ExecutiveReport
from report.narrative import generate_narratives

__all__ = [
    "ExecutiveReport",
    "ExecutiveReportBuilder",
    "generate_narratives",
]
