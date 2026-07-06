"""
PDF generation service for the Flask forecaster frontend.

Converts a markdown-formatted forecast report to PDF bytes using fpdf2.
The ``[VISUAL:TAG]`` placeholder tokens emitted by the backend report agent
are stripped before rendering since charts cannot be embedded in the
text-only PDF output.
"""

from __future__ import annotations

import re

from fpdf import FPDF

_VISUAL_TAG_LINE_RE: re.Pattern[str] = re.compile(r"^\s*\[VISUAL:[A-Z_]+\]\s*$")


def _strip_visual_tags(report_md: str) -> str:
    """Remove ``[VISUAL:TAG]`` placeholder lines from *report_md*.

    Adjacent blank lines produced by the removed lines are also collapsed
    so the surrounding prose is not broken by spurious whitespace.

    Args:
        report_md: Raw markdown string that may contain visual tag lines.

    Returns:
        Cleaned markdown string with visual tag lines removed.
    """
    cleaned: list[str] = []
    for line in report_md.splitlines():
        if _VISUAL_TAG_LINE_RE.match(line):
            while cleaned and cleaned[-1].strip() == "":
                cleaned.pop()
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _sanitize(text: str) -> str:
    """Replace characters outside Latin-1 so core fpdf2 fonts do not crash.

    Args:
        text: Arbitrary Unicode string.

    Returns:
        String with non-Latin-1 characters replaced by ``?``.
    """
    return text.encode("latin-1", errors="replace").decode("latin-1")


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


def report_to_pdf(
    report_md: str,
    title: str = "Forecast Report",
) -> bytes:
    """Convert a markdown forecast report to PDF bytes.

    All ``[VISUAL:TAG]`` tokens are stripped from the output because charts
    cannot be embedded in the text-only PDF.  The resulting document contains
    headings, paragraphs, bullet and numbered lists, and horizontal rules.

    Args:
        report_md: Full markdown report text, possibly containing visual tags.
        title:     Title printed at the top of the first PDF page.

    Returns:
        PDF document as raw bytes suitable for sending as a file download.
    """

    def _cell(pdf: FPDF, height: int, text: str) -> None:
        """Reset x to left margin then emit a multi-cell text block."""
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, height, text)

    pdf = FPDF()
    pdf.set_margins(20, 20, 20)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 12, _sanitize(title), align="C")
    pdf.ln(6)

    cleaned = _strip_visual_tags(report_md)
    for raw_line in cleaned.splitlines():
        line = raw_line.rstrip()
        if line.startswith("### "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 13)
            _cell(pdf, 7, _sanitize(_strip_inline(line[4:])))
            pdf.ln(1)
        elif line.startswith("## "):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 15)
            _cell(pdf, 8, _sanitize(_strip_inline(line[3:])))
            pdf.ln(2)
        elif line.startswith("# "):
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 17)
            _cell(pdf, 9, _sanitize(_strip_inline(line[2:])))
            pdf.ln(2)
        elif re.match(r"^[-*] ", line):
            pdf.set_font("Helvetica", "", 11)
            _cell(pdf, 6, _sanitize("  - " + _strip_inline(line[2:])))
        elif re.match(r"^\d+\. ", line):
            pdf.set_font("Helvetica", "", 11)
            _cell(pdf, 6, _sanitize("  " + _strip_inline(line)))
        elif re.match(r"^-{3,}$", line) or re.match(r"^\*{3,}$", line):
            pdf.ln(2)
            pdf.set_draw_color(180, 180, 180)
            y = pdf.get_y()
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.ln(4)
        elif line == "":
            pdf.ln(3)
        else:
            pdf.set_font("Helvetica", "", 11)
            _cell(pdf, 6, _sanitize(_strip_inline(line)))

    return bytes(pdf.output())
