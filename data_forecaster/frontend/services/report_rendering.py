"""Report presentation helpers for frontend routes."""

from __future__ import annotations

import re
from typing import Any

from flask import render_template

from services.markdown_service import markdown_to_safe_html
from services.report_identity import report_download_filename, resolve_report_identity

_VISUAL_TAG_RE: re.Pattern[str] = re.compile(r"\[VISUAL:([A-Z_]+)\]")

_CHART_FIELD_BY_TAG: dict[str, str] = {
    "HISTORICAL": "chart_historical",
    "STL": "chart_stl",
    "ACF_PACF": "chart_acf_pacf",
    "FORECAST": "chart_forecast",
    "COMPARISON": "chart_model_comparison",
}


def render_analysis_report(
    result: dict[str, Any],
    source_filename: str,
    export_url: str,
    custom_settings: list[dict[str, str]] | None = None,
    prepared_by_fallback: str | None = None,
) -> str:
    """Render a current or persisted final report using shared presentation."""
    executive_report: dict[str, Any] | None = result.get("executive_report")
    report_md: str = result.get("report", "Report not available.")
    web_report_md = _remove_web_dashboard_section(report_md)
    report_identity = resolve_report_identity(
        result, source_filename, prepared_by_fallback
    )
    pdf_filename = report_download_filename(report_identity["title"])
    return render_template(
        "main/report.html",
        segments=_parse_report_segments(web_report_md, result),
        pdf_filename=pdf_filename,
        er=executive_report,
        llm_fallback=bool(result.get("llm_fallback", False)),
        export_url=export_url,
        custom_settings=custom_settings or [],
        report_identity=report_identity,
    )


def _build_chart_segment(tag: str, result: dict[str, Any]) -> dict[str, Any]:
    """Return a chart segment descriptor for the given visual tag."""
    field = _CHART_FIELD_BY_TAG.get(tag)
    chart_data = result.get(field) if field else None
    if tag == "ACF_PACF":
        return {"type": "chart", "tag": tag, "acf_b64": chart_data}
    return {
        "type": "chart",
        "tag": tag,
        "chart_json": chart_data if chart_data else None,
    }


def _parse_report_segments(
    report_text: str,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Split a report into alternating text and chart segments."""
    parts = _VISUAL_TAG_RE.split(report_text)
    segments: list[dict[str, Any]] = []

    for idx, segment in enumerate(parts):
        if idx % 2 == 0:
            if segment.strip():
                segments.append(
                    {"type": "text", "html": markdown_to_safe_html(segment)}
                )
        else:
            segments.append(_build_chart_segment(segment, result))

    return segments


def _remove_web_dashboard_section(report_text: str) -> str:
    """Remove the markdown dashboard section from the web report body."""
    sections = report_text.split("\n\n---\n\n")
    filtered = [
        section
        for section in sections
        if not section.lstrip().startswith("## 1. Executive Dashboard")
    ]
    return "\n\n---\n\n".join(filtered)
