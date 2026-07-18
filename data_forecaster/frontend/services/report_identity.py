"""Shared report identity normalization and presentation helpers."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from typing import Any

MAX_REPORT_TITLE_LENGTH = 200


class ReportTitleValidationError(ValueError):
    """Raised when a custom report title violates submission rules."""


def normalize_report_title(raw_title: Any, source_filename: str) -> str:
    """Return a validated custom title or the canonical dataset-based default."""
    title = str(raw_title or "").strip()
    if len(title) > MAX_REPORT_TITLE_LENGTH:
        raise ReportTitleValidationError(
            f"Report title must be {MAX_REPORT_TITLE_LENGTH} characters or fewer."
        )
    if title:
        return title
    return default_report_title(source_filename)


def default_report_title(source_filename: str) -> str:
    """Return the canonical dataset-based default report title."""
    source_stem = (
        source_filename.rsplit(".", 1)[0] if "." in source_filename else source_filename
    )
    prefix = "Forecast Report — "
    usable_stem = source_stem.strip() or "data"
    return f"{prefix}{usable_stem[: MAX_REPORT_TITLE_LENGTH - len(prefix)].rstrip()}"


def resolve_report_identity(
    result: dict[str, Any],
    source_filename: str,
    prepared_by_fallback: str | None = None,
) -> dict[str, str]:
    """Resolve title, author, and UTC creation date for report presentation."""
    executive_report = result.get("executive_report")
    metadata: dict[str, Any] = {}
    if isinstance(executive_report, dict):
        metadata_candidate = executive_report.get("metadata")
        if isinstance(metadata_candidate, dict):
            metadata = metadata_candidate
    saved_title = str(result.get("title") or "").strip()
    metadata_title = str(metadata.get("title") or "").strip()
    title = saved_title or metadata_title or default_report_title(source_filename)

    prepared_by = str(metadata.get("prepared_by") or "").strip()
    if not prepared_by or prepared_by == "Unknown":
        prepared_by = str(prepared_by_fallback or "").strip() or "Unknown"

    generated_at = metadata.get("generated_at") or result.get("created_at")
    return {
        "title": title,
        "prepared_by": prepared_by,
        "creation_date": format_utc_timestamp(generated_at),
    }


def report_download_filename(title: str) -> str:
    """Return a bounded filesystem-safe PDF filename derived from a title."""
    normalized = unicodedata.normalize("NFKC", title)
    basename = re.sub(r"[^\w.-]+", "_", normalized, flags=re.UNICODE)
    basename = basename.strip("._")[:100].rstrip("._")
    return f"{basename or 'forecast_report'}.pdf"


def format_utc_timestamp(value: Any) -> str:
    """Format an ISO/SQLite timestamp as a human-readable UTC value."""
    raw = str(value or "").strip()
    if not raw:
        return "Unknown"
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return "Unknown"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.strftime("%B %d, %Y at %H:%M UTC")
