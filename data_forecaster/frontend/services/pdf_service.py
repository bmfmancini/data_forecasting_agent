"""
PDF generation service for the Flask forecaster frontend.

Converts a markdown-formatted forecast report to PDF bytes using fpdf2.
``[VISUAL:TAG]`` placeholder tokens are replaced with chart images
embedded as base64 PNG strings from the analysis result data — no
network calls to the backend are required.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import tempfile
from binascii import Error as BinasciiError
from pathlib import Path
from typing import Any

from fpdf import FPDF

logger = logging.getLogger(__name__)

_VISUAL_TAG_LINE_RE: re.Pattern[str] = re.compile(r"^\s*\[VISUAL:([A-Z_]+)\]\s*$")
_PDF_FONT_FAMILY = "DejaVuSans"
_DEFAULT_PDF_FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
_PDF_FONT_FILES = {
    "": "DejaVuSans.ttf",
    "B": "DejaVuSans-Bold.ttf",
}
_PDF_SYMBOL_FALLBACKS = str.maketrans(
    {
        "📈": "↗",
        "📊": "▥",
        "🎯": "◎",
        "🔍": "◉",
        "🤖": "◆",
        "✅": "✓",
    }
)

# Maps visual tags to base64 PNG fields in the analysis result.
_CHART_PNG_FIELD_BY_TAG: dict[str, str] = {
    "HISTORICAL": "chart_historical_png",
    "STL": "chart_stl_png",
    "ACF_PACF": "chart_acf_pacf",
    "FORECAST": "chart_forecast_png",
    "COMPARISON": "chart_model_comparison_png",
}


def _sanitize(text: str) -> str:
    """Preserve Unicode and map unsupported dashboard emoji to text glyphs.

    Args:
        text: Arbitrary Unicode string.

    Returns:
        Unicode text supported by the embedded font.
    """
    return text.translate(_PDF_SYMBOL_FALLBACKS)


def _register_pdf_fonts(pdf: FPDF) -> None:
    """Register regular and bold Unicode fonts required by the report.

    ``PDF_FONT_DIR`` is resolved at call time so values loaded after module import
    (for example from a Flask ``.env`` file) are honored.
    """
    font_dir = Path(os.getenv("PDF_FONT_DIR", str(_DEFAULT_PDF_FONT_DIR)))
    for style, filename in _PDF_FONT_FILES.items():
        font_path = font_dir / filename
        if not font_path.is_file():
            raise RuntimeError(
                f"Required PDF font is unavailable: {font_path}. "
                "Install fonts-dejavu-core or set PDF_FONT_DIR."
            )
        pdf.add_font(_PDF_FONT_FAMILY, style=style, fname=font_path)


def _strip_inline(text: str) -> str:
    """Remove markdown bold, italic, and code markers, preserving text.

    Args:
        text: Markdown-formatted inline text.

    Returns:
        Plain text without ``**``, ``*``, or backtick markers.
    """
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    return text


def _fetch_chart_png(
    tag: str,
    result: dict[str, Any],
) -> bytes | None:
    """Fetch a chart as PNG bytes from the analysis result.

    All chart PNGs are pre-computed on the backend during the pipeline
    run and stored as base64-encoded strings in the analysis result.
    This function decodes them — no network calls are needed.

    Args:
        tag:    Visual tag name (e.g. ``"HISTORICAL"``).
        result: Analysis result dict containing chart PNG data.

    Returns:
        PNG bytes if the chart was successfully decoded, otherwise ``None``.
    """
    field = _CHART_PNG_FIELD_BY_TAG.get(tag)
    if not field:
        return None
    b64_data = result.get(field)
    if not b64_data or not isinstance(b64_data, str):
        return None
    try:
        return base64.b64decode(b64_data)
    except BinasciiError as exc:
        logger.warning("Failed to decode base64 for chart tag '%s': %s", tag, exc)
        return None


def _embed_image(pdf: FPDF, png_bytes: bytes, max_width: float) -> None:
    """Embed a PNG image in the PDF, scaling to fit the page width.

    Args:
        pdf:       FPDF instance.
        png_bytes:  PNG image bytes.
        max_width:  Maximum width in mm for the image.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(png_bytes)
            tmp_path = tmp.name
        pdf.image(tmp_path, w=max_width)
    except (RuntimeError, OSError) as exc:
        logger.warning("Failed to embed image in PDF (path: %s): %s", tmp_path, exc)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError as exc:
                logger.warning(
                    "Failed to clean up temp image file %s: %s", tmp_path, exc
                )


def _process_line(
    pdf: FPDF,
    line: str,
    result: dict[str, Any],
    max_img_width: float,
) -> None:
    """Process a single markdown line and emit PDF content.

    Args:
        pdf:            FPDF instance.
        line:           Stripped markdown line.
        result:         Analysis result dict for chart data.
        max_img_width:  Max image width in mm.
    """

    def _cell(height: int, text: str) -> None:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, height, text)

    tag_match = _VISUAL_TAG_LINE_RE.match(line)
    if tag_match:
        tag = tag_match.group(1)
        png_bytes = _fetch_chart_png(tag, result)
        if png_bytes:
            pdf.ln(3)
            _embed_image(pdf, png_bytes, max_img_width)
            pdf.ln(3)
        return
    if line.startswith("### "):
        pdf.ln(3)
        pdf.set_font(_PDF_FONT_FAMILY, "B", 13)
        _cell(7, _sanitize(_strip_inline(line[4:])))
        pdf.ln(1)
    elif line.startswith("## "):
        pdf.ln(4)
        pdf.set_font(_PDF_FONT_FAMILY, "B", 15)
        _cell(8, _sanitize(_strip_inline(line[3:])))
        pdf.ln(2)
    elif line.startswith("# "):
        pdf.ln(5)
        pdf.set_font(_PDF_FONT_FAMILY, "B", 17)
        _cell(9, _sanitize(_strip_inline(line[2:])))
        pdf.ln(2)
    elif re.match(r"^[-*] ", line):
        pdf.set_font(_PDF_FONT_FAMILY, "", 11)
        _cell(6, _sanitize("  - " + _strip_inline(line[2:])))
    elif re.match(r"^\d+\. ", line):
        pdf.set_font(_PDF_FONT_FAMILY, "", 11)
        _cell(6, _sanitize("  " + _strip_inline(line)))
    elif re.match(r"^-{3,}$", line) or re.match(r"^\*{3,}$", line):
        pdf.ln(2)
        pdf.set_draw_color(180, 180, 180)
        y = pdf.get_y()
        pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
        pdf.ln(4)
    elif line == "":
        pdf.ln(3)
    else:
        pdf.set_font(_PDF_FONT_FAMILY, "", 11)
        _cell(6, _sanitize(_strip_inline(line)))


def report_to_pdf(
    report_md: str,
    title: str = "Forecast Report",
    result: dict[str, Any] | None = None,
) -> bytes:
    """Convert a markdown forecast report to PDF bytes with embedded charts.

    ``[VISUAL:TAG]`` tokens in the report markdown are replaced with chart
    images.  All chart PNGs are pre-computed on the backend during the
    pipeline run and stored as base64-encoded strings in the analysis
    result — no network calls to the backend are needed.

    Args:
        report_md:  Full markdown report text, possibly containing visual tags.
        title:      Title printed at the top of the first PDF page.
        result:     Analysis result dict containing chart PNG data.

    Returns:
        PDF document as raw bytes suitable for sending as a file download.
    """
    result = result or {}
    pdf = FPDF()
    _register_pdf_fonts(pdf)
    pdf.set_margins(20, 20, 20)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)
    max_img_width = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.set_font(_PDF_FONT_FAMILY, "B", 20)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 12, _sanitize(title), align="C")
    pdf.ln(6)

    for raw_line in report_md.splitlines():
        _process_line(
            pdf,
            raw_line.rstrip(),
            result,
            max_img_width,
        )

    return bytes(pdf.output())
