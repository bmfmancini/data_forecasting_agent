"""Tests for PDF report image embedding."""

from __future__ import annotations

from typing import Any

from data_forecaster.frontend.services.pdf_service import _embed_image


class _Pdf:
    def image(self, _path: str, w: float) -> None:
        raise OSError("bad image")


def test_embed_image_skips_file_errors(caplog: Any) -> None:
    """Invalid chart images should be logged and skipped."""
    _embed_image(_Pdf(), b"not-a-png", max_width=100.0)

    assert "Failed to embed image in PDF" in caplog.text
