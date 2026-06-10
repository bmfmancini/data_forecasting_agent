"""
Report tab module for the Time Series Data Forecaster Agent.
Handles the display of forecast reports and PDF export functionality.

The backend report-generation agent emits [VISUAL:TAG] placeholders
(e.g. [VISUAL:HISTORICAL], [VISUAL:STL], [VISUAL:ACF_PACF],
[VISUAL:FORECAST], [VISUAL:COMPARISON]). This tab replaces those
tokens with the actual plotly/PNG charts carried on the analysis
result dict.
"""

from __future__ import annotations

import base64
import re
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from utils.pdf_utils import report_to_pdf

# Map the [VISUAL:TAG] tokens the LLM emits to the chart fields populated
# by backend/orchestrator.py. Unknown tags are silently dropped.
_VISUAL_TAG_REGEX = re.compile(r"\[VISUAL:([A-Z_]+)\]")
_CHART_FIELD_BY_TAG: dict[str, str] = {
    "HISTORICAL": "chart_historical",
    "STL": "chart_stl",
    "ACF_PACF": "chart_acf_pacf",
    "FORECAST": "chart_forecast",
    "COMPARISON": "chart_model_comparison",
}


def _render_chart(tag: str, result: dict[str, Any], occurrence: int) -> None:
    """Render a single chart for a [VISUAL:TAG] token.

    ``occurrence`` is the 0-based index of this tag's appearance in the
    report. We embed it in the streamlit ``key`` so that if the same
    chart (e.g. two ``[VISUAL:STL]`` tags) ever appears more than once,
    each call gets a unique element ID.
    """
    field = _CHART_FIELD_BY_TAG.get(tag)
    if field is None:
        return

    chart = result.get(field)
    chart_key = f"report_chart_{tag.lower()}_{occurrence}"

    if tag == "ACF_PACF":
        # base64-encoded PNG from backend.utils.visualization.plot_acf_pacf
        if chart:
            try:
                st.image(
                    base64.b64decode(chart),
                    width="stretch",
                    caption="ACF / PACF",
                )
            except Exception as exc:  # pragma: no cover - defensive
                st.warning(f"Could not render ACF/PACF chart: {exc}")
        else:
            st.info("ACF/PACF chart unavailable.")
        return

    # Plotly figures are serialised as dicts in the result payload.
    if chart:
        try:
            fig = go.Figure(chart, skip_invalid=True)
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=chart_key,
            )
        except Exception as exc:  # pragma: no cover - defensive
            st.warning(f"Could not render {tag} chart: {exc}")
    else:
        st.info(f"{tag.replace('_', ' ').title()} chart unavailable.")


def _render_report_with_visuals(report_text: str, result: dict[str, Any]) -> None:
    """Render the markdown report, replacing [VISUAL:TAG] tokens with charts."""
    # Split the report by visual tag occurrences, preserving the surrounding text.
    parts = _VISUAL_TAG_REGEX.split(report_text)

    # `re.split` with a capturing group yields alternating text/match segments:
    # [text_before, TAG, text_between_TAG1_TAG2, TAG, ..., text_after]
    chart_occurrence = 0
    for idx, segment in enumerate(parts):
        if idx % 2 == 0:
            # Text segment — render as markdown. Skip empty fragments to avoid
            # the spurious blank lines you'd otherwise get around each tag.
            if segment.strip():
                st.markdown(segment)
        else:
            # Tag segment — render the corresponding chart.
            _render_chart(segment, result, occurrence=chart_occurrence)
            chart_occurrence += 1


def render_report_tab(result: dict, info: dict) -> None:
    """
    Render the report tab with forecast results and PDF export.

    Args:
        result: Dictionary containing analysis results (must include the
            ``report`` string and any of the ``chart_*`` fields used by
            the [VISUAL:TAG] tokens).
        info: Dictionary containing file information.
    """
    report_text = result.get("report", "Report not available.")

    # PDF export — pdf_utils strips the [VISUAL:*] tokens so the PDF
    # remains a clean text document.
    try:
        pdf_bytes = report_to_pdf(report_text)
        fname = (
            f"forecast_report_{info['filename'].rsplit('.', 1)[0]}.pdf"
            if info and info.get("filename")
            else "forecast_report.pdf"
        )
        st.download_button(
            label="⬇️ Download Report as PDF",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as pdf_error:
        st.warning(f"PDF export unavailable: {pdf_error}")

    st.markdown("---")

    # If the report contains no visual tags, fall back to a single
    # markdown render for performance and to preserve the original
    # behaviour for fallback reports.
    if _VISUAL_TAG_REGEX.search(report_text):
        _render_report_with_visuals(report_text, result)
    else:
        st.markdown(report_text)
