"""
PDF utility functions for the Time Series Data Forecaster Agent.
Provides functionality to convert markdown reports to PDF format.
"""

import re
from fpdf import FPDF

# Strip [VISUAL:TAG] placeholders that the LLM report emits — those are
# rendered as charts in the Streamlit UI and are not meaningful in a
# text-only PDF. Removing the whole line (and any blank lines around it)
# keeps the surrounding prose from being broken by dangling whitespace.
_VISUAL_TAG_LINE_REGEX = re.compile(r"^\s*\[VISUAL:[A-Z_]+\]\s*$")


def _strip_visual_tags(report_md: str) -> str:
    """Remove [VISUAL:TAG] placeholder lines and their surrounding blanks."""
    cleaned_lines: list[str] = []
    for line in report_md.splitlines():
        if _VISUAL_TAG_LINE_REGEX.match(line):
            # Drop the tag line itself plus the trailing blank line that
            # the prompt requires the LLM to leave after each tag.
            while cleaned_lines and cleaned_lines[-1].strip() == "":
                cleaned_lines.pop()
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def report_to_pdf(report_md: str, title: str = "Forecast Report") -> bytes:
    """
    Render a markdown report string to PDF bytes using fpdf2.

    Args:
        report_md: Markdown formatted report text
        title: Title for the PDF document

    Returns:
        PDF document as bytes
    """

    def _sanitize(text: str) -> str:
        """Drop chars outside Latin-1 so core fonts don't crash."""
        return text.encode("latin-1", errors="replace").decode("latin-1")

    def _strip_inline(text: str) -> str:
        """Remove bold/italic/code markers, keep text content."""
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"\*(.*?)\*", r"\1", text)
        text = re.sub(r"`(.*?)`", r"\1", text)
        return text

    def _cell(pdf: FPDF, h: int, text: str) -> None:
        """Reset x to left margin then render a multi_cell."""
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, h, text)

    pdf = FPDF()
    pdf.set_margins(20, 20, 20)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Cover title
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 12, _sanitize(title), align="C")
    pdf.ln(6)

    cleaned_report = _strip_visual_tags(report_md)
    for raw_line in cleaned_report.splitlines():
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

    try:
        pdf_output_result = pdf.output(dest="S")
    except TypeError:
        pdf_output_result = pdf.output()

    if isinstance(pdf_output_result, str):
        # If fpdf.output() unexpectedly returns a string, encode it to bytes
        return pdf_output_result.encode("latin-1", errors="replace")
    return pdf_output_result
