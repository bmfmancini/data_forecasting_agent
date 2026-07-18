"""Tests for PDF report image embedding."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from data_forecaster.frontend.services.pdf_service import (
    _embed_image,
    _sanitize,
    report_to_pdf,
)


class _Pdf:
    def image(self, _path: str, w: float) -> None:
        raise OSError("bad image")


def test_embed_image_skips_file_errors(caplog: Any) -> None:
    """Invalid chart images should be logged and skipped."""
    _embed_image(_Pdf(), b"not-a-png", max_width=100.0)

    assert "Failed to embed image in PDF" in caplog.text


def test_pdf_embeds_unicode_font_without_lossy_sanitization(tmp_path: Any) -> None:
    """Report punctuation must survive the PDF font and encoding path."""
    unicode_text = "Unicode — × ✓ “smart quotes” and Holt-Winters’ range"
    dashboard_icons = "📈 📊 🎯 🔍 🤖 ✅"
    dashboard_fallbacks = "↗ ▥ ◎ ◉ ◆ ✓"

    assert _sanitize(unicode_text) == unicode_text
    assert _sanitize(dashboard_icons) == dashboard_fallbacks
    pdf_bytes = report_to_pdf(f"{unicode_text}\n{dashboard_icons}")

    assert b"/ToUnicode" in pdf_bytes
    assert b"/FontFile2" in pdf_bytes
    assert b"Helvetica" not in pdf_bytes

    pdf_path = tmp_path / "unicode-report.pdf"
    pdf_path.write_bytes(pdf_bytes)
    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        extracted = subprocess.run(
            [pdftotext, str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert unicode_text in extracted
        for glyph in dashboard_fallbacks.split():
            assert glyph in extracted


def test_pdf_font_directory_is_resolved_at_generation_time(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """A late PDF_FONT_DIR update must be honored after module import."""
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    if not (font_dir / "DejaVuSans.ttf").is_file():
        pytest.skip("System DejaVu font is unavailable")

    monkeypatch.setenv("PDF_FONT_DIR", str(tmp_path / "missing-fonts"))
    with pytest.raises(RuntimeError, match="Required PDF font is unavailable"):
        report_to_pdf("first attempt")

    monkeypatch.setenv("PDF_FONT_DIR", str(font_dir))
    assert report_to_pdf("second attempt — ✓").startswith(b"%PDF")


def test_pdf_title_block_contains_report_identity(tmp_path: Any) -> None:
    """The exported first page carries the same identity shown on the web."""
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        pytest.skip("pdftotext is unavailable")

    pdf_bytes = report_to_pdf(
        "## Report body",
        title="Q4 Montréal report",
        prepared_by="alice",
        creation_date="July 18, 2026 at 01:02 UTC",
    )
    pdf_path = tmp_path / "identity-report.pdf"
    pdf_path.write_bytes(pdf_bytes)
    extracted = subprocess.run(
        [pdftotext, str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert "Q4 Montréal report" in extracted
    assert "Prepared by: alice" in extracted
    assert "Forecast created: July 18, 2026 at 01:02 UTC" in extracted
