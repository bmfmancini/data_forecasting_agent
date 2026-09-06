"""Report presentation helpers for frontend routes."""

from __future__ import annotations

import re
from typing import Any

from flask import render_template

from services.markdown_service import markdown_to_safe_html
from services.report_editing import report_sections

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
    edit_url: str | None = None,
) -> str:
    """Render a current or persisted final report using shared presentation."""
    executive_report: dict[str, Any] | None = result.get("executive_report")
    sections = report_sections(result)
    for section in sections:
        body = section["body"]
        if section["is_dashboard"]:
            section["tiles"], body = _dashboard_tiles(body)
            widgets = (executive_report or {}).get("dashboard", {}).get("widgets", [])
            for tile in section["tiles"]:
                # Descriptions apply only while the corresponding fact is unchanged.
                tile["description"] = next((w.get("description", "") for w in widgets
                    if f"{w.get('icon', '')} {w.get('title', '')}".strip() == tile["label"]
                    and str(w.get("value", "")) == tile["value"]), "")
        section["segments"] = _parse_report_segments(body, result)
    return render_template(
        "main/report.html",
        sections=sections,
        edit_url=edit_url,
        er=executive_report,
        llm_fallback=bool(result.get("llm_fallback", False)),
        export_url=export_url,
        custom_settings=custom_settings or [],
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


def _dashboard_tiles(body: str) -> tuple[list[dict[str, str]], str]:
    """Render the editable dashboard table as tiles without losing added prose."""
    lines = body.splitlines()
    for start, line in enumerate(lines):
        if line.strip().lower() != "| metric | value | status |":
            continue
        end = start + 1
        while end < len(lines) and lines[end].strip().startswith("|"):
            end += 1
        rows = lines[start + 1:end]
        if not rows or not re.fullmatch(r"[\s|:\-]+", rows[0]):
            continue
        tiles = []
        for row in rows[1:]:
            cells = [cell.strip().replace(r"\|", "|") for cell in re.split(r"(?<!\\)\|", row.strip().strip("|"))]
            if len(cells) != 3:
                break
            label, value, status = cells
            tiles.append({"label": label, "value": value,
                          "status": status if status in {"positive", "negative", "warning", "info", "neutral"} else "neutral"})
        else:
            if tiles:
                return tiles, "\n".join(lines[:start] + lines[end:])
    return [], body
