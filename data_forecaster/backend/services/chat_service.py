"""Chat service for the Data Forecaster backend.

Contains :func:`chat_with_data` and :func:`chat_general` for the
data-explorer chat functionality.  Extracted from ``orchestrator.py``
to separate chat concerns from pipeline orchestration.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from core.llm_factory import get_llm
from core.logging_config import get_logger
from prompts.general_chat_prompt import GENERAL_CHAT_PROMPT
from prompts.orchestrator_prompt import ORCHESTRATOR_CHAT_PROMPT
from services.rag_service import get_rag_kb

logger = get_logger(__name__)

# ── Visualization config validation (SEC-007) ───────────────────────────────
# Whitelists for LLM-generated visualization configs to prevent injection
# via malicious or hallucinated Plotly configurations.

# Allowed Plotly trace types in the ``data`` array of a figure.
_ALLOWED_PLOTLY_TYPES: set[str] = {
    "scatter",
    "bar",
    "line",
    "pie",
    "heatmap",
    "histogram",
    "box",
    "violin",
    "area",
}

# Allowed keys in each trace dict of a Plotly ``data`` array.
_ALLOWED_TRACE_KEYS: set[str] = {
    "type",
    "mode",
    "x",
    "y",
    "z",
    "labels",
    "values",
    "name",
    "marker",
    "fill",
    "text",
}

# Allowed keys in the ``marker`` sub-dict.
_ALLOWED_MARKER_KEYS: set[str] = {"color", "size", "opacity", "symbol"}

# Allowed keys in the ``layout`` dict of a Plotly figure.
_ALLOWED_LAYOUT_KEYS: set[str] = {
    "title",
    "xaxis",
    "yaxis",
    "showlegend",
    "legend",
    "margin",
    "height",
    "width",
    "barmode",
    "template",
}

# Allowed dynamic config keys (for the ``renderDynamic`` frontend path).
_ALLOWED_DYNAMIC_KEYS: set[str] = {
    "chart_type",
    "x_data",
    "y_data",
    "z_data",
    "name",
    "color",
    "title",
    "x_label",
    "y_label",
}

# Allowed chart_type values for dynamic configs.
_ALLOWED_DYNAMIC_CHART_TYPES: set[str] = {
    "line",
    "bar",
    "scatter",
    "histogram",
    "box",
    "violin",
    "heatmap",
    "area",
}

# Max number of data points allowed in any single array to prevent
# memory exhaustion on the frontend.
_MAX_DATA_POINTS: int = 10000


def _is_safe_scalar(value: Any) -> bool:
    """Check if a scalar value is safe for frontend rendering.

    Rejects strings containing ``<script``, ``javascript:``, or ``data:`
    patterns that could be used for XSS.

    Args:
        value: The value to check.

    Returns:
        ``True`` if the value is safe.
    """
    if isinstance(value, str):
        lowered = value.lower()
        if "<script" in lowered or "javascript:" in lowered or "data:" in lowered:
            return False
    return True


def _sanitize_array(arr: Any) -> list[Any] | None:
    """Validate and return a data array, or ``None`` if unsafe/oversized.

    Args:
        arr: The value to validate as a data array.

    Returns:
        A list of safe scalar values, or ``None`` if the input is not a
        list, exceeds ``_MAX_DATA_POINTS``, or contains unsafe strings.
    """
    if not isinstance(arr, list):
        return None
    if len(arr) > _MAX_DATA_POINTS:
        logger.warning("Visualization array exceeds max data points (%d)", len(arr))
        return None
    for item in arr:
        if not _is_safe_scalar(item):
            return None
    return arr


def _sanitize_marker(marker: Any) -> dict[str, Any] | None:
    """Validate a Plotly ``marker`` sub-dict.

    Args:
        marker: The marker dict to validate.

    Returns:
        A sanitized marker dict, or ``None`` if invalid.
    """
    if not isinstance(marker, dict):
        return None
    safe: dict[str, Any] = {}
    for key in _ALLOWED_MARKER_KEYS:
        if key in marker and _is_safe_scalar(marker[key]):
            safe[key] = marker[key]
    return safe if safe else None


def _validate_plotly_figure(config: dict[str, Any]) -> dict[str, Any] | None:
    """Validate a Plotly figure config ``{data, layout}``.

    Whitelists trace types, strips unknown keys, and sanitises string
    values to prevent XSS.

    Args:
        config: The parsed JSON dict to validate.

    Returns:
        A sanitized figure dict, or ``None`` if validation fails.
    """
    safe_data: list[dict[str, Any]] = []
    raw_data = config.get("data", [])
    if not isinstance(raw_data, list):
        return None

    for trace in raw_data:
        if not isinstance(trace, dict):
            continue
        trace_type = trace.get("type", "scatter")
        if trace_type not in _ALLOWED_PLOTLY_TYPES:
            logger.warning("Rejected Plotly trace type: %s", trace_type)
            continue
        safe_trace: dict[str, Any] = {"type": trace_type}
        for key in _ALLOWED_TRACE_KEYS:
            if key not in trace:
                continue
            val = trace[key]
            if key in ("x", "y", "z", "labels", "values"):
                sanitized = _sanitize_array(val)
                if sanitized is not None:
                    safe_trace[key] = sanitized
            elif key == "marker":
                sanitized = _sanitize_marker(val)
                if sanitized is not None:
                    safe_trace[key] = sanitized
            elif _is_safe_scalar(val):
                safe_trace[key] = val
        safe_data.append(safe_trace)

    if not safe_data:
        return None

    safe_layout: dict[str, Any] = {}
    raw_layout = config.get("layout", {})
    if isinstance(raw_layout, dict):
        for key in _ALLOWED_LAYOUT_KEYS:
            if key in raw_layout and _is_safe_scalar(raw_layout[key]):
                safe_layout[key] = raw_layout[key]

    return {"data": safe_data, "layout": safe_layout}


def _validate_dynamic_config(config: dict[str, Any]) -> dict[str, Any] | None:
    """Validate a dynamic chart config (``renderDynamic`` frontend path).

    Whitelists chart types and config keys, sanitises data arrays.

    Args:
        config: The parsed JSON dict to validate.

    Returns:
        A sanitized config dict, or ``None`` if validation fails.
    """
    chart_type = str(config.get("chart_type", "bar")).lower()
    if chart_type not in _ALLOWED_DYNAMIC_CHART_TYPES:
        logger.warning("Rejected dynamic chart_type: %s", chart_type)
        return None

    safe: dict[str, Any] = {"chart_type": chart_type}
    for key in _ALLOWED_DYNAMIC_KEYS:
        if key not in config:
            continue
        val = config[key]
        if key in ("x_data", "y_data", "z_data"):
            sanitized = _sanitize_array(val)
            if sanitized is not None:
                safe[key] = sanitized
        elif _is_safe_scalar(val):
            safe[key] = val

    # Must have at least one data field to be useful
    if not any(k in safe for k in ("x_data", "y_data", "z_data")):
        return None

    return safe


def _validate_viz_config(config: Any) -> dict[str, Any] | None:
    """Validate an LLM-generated visualization config against a strict schema.

    Detects whether the config is a Plotly figure (has ``data`` or
    ``layout`` keys) or a dynamic config (has ``chart_type``) and routes
    to the appropriate validator.  Returns ``None`` if the config is
    invalid, not a dict, or fails validation — the caller should then
    return the raw text answer without visualization data.

    Args:
        config: The parsed JSON object from the LLM response.

    Returns:
        A sanitized config dict safe for frontend rendering, or ``None``.
    """
    if not isinstance(config, dict):
        return None

    # Plotly figure: has "data" array or "layout" dict
    if "data" in config or "layout" in config:
        return _validate_plotly_figure(config)

    # Dynamic config: has "chart_type"
    if "chart_type" in config:
        return _validate_dynamic_config(config)

    # Also handle configs with just "type" (Plotly shorthand)
    if "type" in config:
        return _validate_plotly_figure({"data": [config]})

    return None


def chat_with_data(
    query: str, df: pd.DataFrame, file_id: str, chroma_persist_dir: str
) -> dict[str, Any]:
    """Handle a data-explorer chat query about a specific uploaded dataset.

    Retrieves analysis memory from RAG, prepares a data summary, and
    invokes the LLM with the orchestrator chat prompt.

    Args:
        query:              The user's natural-language question.
        df:                 The uploaded DataFrame.
        file_id:            Identifier for the uploaded file.
        chroma_persist_dir: Path to the ChromaDB persistence directory.

    Returns:
        A dict with ``answer``, and optionally ``visualization_data``
        and ``visualization_type`` if the LLM produced a chart config.
    """
    logger.info("Data Explorer query for file_id=%s: %s", file_id, query)

    # 1. Retrieve Analysis Memory from RAG
    rag_kb = get_rag_kb(chroma_persist_dir)
    memory_context = ""
    try:
        chunks = rag_kb.retrieve(query, k=3)
        memory_context = "\n".join(chunks)
    except Exception as exc:
        logger.warning("RAG retrieval failed for chat: %s", exc)

    # 2. Prepare Data Summary for the LLM
    stats_dict = df.describe().iloc[:, :20].to_dict()

    # ── Enhanced Data Visibility ─────────────────────────────────────────────
    data_snapshots = ""
    try:
        date_cols = df.select_dtypes(
            include=["datetime", "datetimetz"]
        ).columns.tolist()
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if date_cols and num_cols:
            main_date = date_cols[0]
            val_col = num_cols[0]

            top_5 = df.nlargest(5, val_col)[[main_date, val_col]].to_string(index=False)
            bottom_5 = df.nsmallest(5, val_col)[[main_date, val_col]].to_string(
                index=False
            )

            data_snapshots = f"\nHIGHEST 5 RECORDS (Sorted by {val_col}):\n{top_5}\n"
            data_snapshots += f"\nLOWEST 5 RECORDS (Sorted by {val_col}):\n{bottom_5}\n"
    except Exception as exc:
        logger.warning("Failed to generate data highlights for chat: %s", exc)

    data_summary = f"""
    Dataset Stats:
    - Rows: {len(df)}
    - Statistical Summary: {stats_dict}
    {data_snapshots}
    """

    # 3. Setup LLM based on configuration
    llm = get_llm(temperature=0)

    prompt = ORCHESTRATOR_CHAT_PROMPT

    try:
        chain = prompt | llm
        response = chain.invoke(
            {
                "analysis_context": memory_context,
                "data_summary": data_summary,
                "query": query,
            }
        )

        content = response.content

        # Handle potential JSON visualization responses from LLM
        if "visualization_type" in content:
            clean_content = re.sub(r"```json\s*|\s*```", "", content).strip()
            try:
                parsed = json.loads(clean_content)
                validated = _validate_viz_config(parsed)
                if validated is not None:
                    return validated
            except json.JSONDecodeError:
                pass

        # Try to parse any JSON visualization configuration from the response
        json_pattern = r"\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}"
        matches = re.findall(json_pattern, content, re.DOTALL)

        for match in matches:
            try:
                potential_config = json.loads(match)
                if isinstance(potential_config, dict) and (
                    "data" in potential_config
                    or "type" in potential_config
                    or "layout" in potential_config
                    or "chart_type" in potential_config
                ):
                    validated = _validate_viz_config(potential_config)
                    if validated is None:
                        logger.warning(
                            "Rejected LLM visualization config: failed validation"
                        )
                        continue
                    clean_answer = content.replace(match, "").strip()
                    return {
                        "answer": clean_answer,
                        "visualization_data": validated,
                        "visualization_type": "dynamic",
                    }
            except json.JSONDecodeError:
                continue

        return {"answer": content}

    except Exception:
        logger.exception("LLM Chat failed")
        return {
            "answer": (
                "I encountered an error while processing your request. "
                "Please try again later."
            )
        }


def chat_general(query: str, chroma_persist_dir: str) -> dict[str, Any]:
    """Handle general questions using only the RAG knowledge base.

    No dataset is required — the LLM answers based on forecasting
    documentation stored in ChromaDB.

    Args:
        query:              The user's natural-language question.
        chroma_persist_dir: Path to the ChromaDB persistence directory.

    Returns:
        A dict with an ``answer`` key.
    """
    logger.info("General chat query: %s", query)

    # 1. Retrieve relevant information from RAG
    rag_kb = get_rag_kb(chroma_persist_dir)
    memory_context = ""
    try:
        chunks = rag_kb.retrieve(query, k=5)
        memory_context = "\n".join(chunks)
    except Exception as exc:
        logger.warning("RAG retrieval failed for general chat: %s", exc)

    # 2. Setup LLM based on configuration
    llm = get_llm(temperature=0)

    try:
        chain = GENERAL_CHAT_PROMPT | llm
        response = chain.invoke({"context": memory_context, "query": query})

        content = response.content
        return {"answer": content}

    except Exception:
        logger.exception("LLM General Chat failed")
        return {
            "answer": (
                "I encountered an error while processing your request. "
                "Please try again later."
            )
        }
