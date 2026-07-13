"""Stage 2 — LLM narrative generation for the executive report.

The :func:`generate_narratives` function receives a pre-populated
:class:`ExecutiveReport` (with all deterministic facts computed by the
builder) and fills the ``narrative`` text fields on each section using the
LLM.  The LLM receives the structured data for each section as JSON context
and is instructed to use ONLY the provided values.

On LLM failure, narrative fields fall back to deterministic templates that
use hedged language and no jargon — the report remains usable without the
LLM.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.config import GEMINI_TEMPERATURE
from core.llm_factory import get_llm
from core.logging_config import get_logger
from forecasting.selection_policy import validate_llm_output
from prompts.report_generation_prompt import (
    DATA_QUALITY_NARRATIVE_PROMPT,
    EXECUTIVE_SUMMARY_NARRATIVE_PROMPT,
    EXPLAINABILITY_NARRATIVE_PROMPT,
    FORECAST_OUTLOOK_NARRATIVE_PROMPT,
    HISTORICAL_ANALYSIS_NARRATIVE_PROMPT,
    MODEL_COMPARISON_NARRATIVE_PROMPT,
    RECOMMENDATION_NARRATIVE_PROMPT,
    STATISTICAL_AUDIT_NARRATIVE_PROMPT,
)
from report.models import ExecutiveReport
from utils.token_tracking import extract_token_usage, estimate_input_text

logger = get_logger(__name__)


def generate_narratives(
    report: ExecutiveReport,
    user_prompt: str | None = None,
) -> tuple[ExecutiveReport, dict[str, int]]:
    """Fill narrative fields on an :class:`ExecutiveReport` using the LLM.

    Args:
        report:      Pre-populated :class:`ExecutiveReport` from the builder.
        user_prompt: Optional extra instructions appended to each prompt.

    Returns:
        A tuple of (updated :class:`ExecutiveReport`, token_usage_dict).
    """
    llm = get_llm(temperature=GEMINI_TEMPERATURE)
    total_usage: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    extra = (
        f"\n\nADDITIONAL USER INSTRUCTIONS:\n{user_prompt.strip()}"
        if user_prompt and user_prompt.strip()
        else ""
    )

    # ── Executive Summary ─────────────────────────────────────────────────
    report.executive_summary.narrative = _generate_section(
        llm,
        EXECUTIVE_SUMMARY_NARRATIVE_PROMPT,
        report.executive_summary,
        "executive_summary",
        total_usage,
        extra,
    )

    # ── Data Quality ──────────────────────────────────────────────────────
    report.data_quality.narrative = _generate_section(
        llm,
        DATA_QUALITY_NARRATIVE_PROMPT,
        report.data_quality,
        "data_quality",
        total_usage,
        extra,
    )

    # ── Historical Analysis ───────────────────────────────────────────────
    report.historical_analysis.narrative = _generate_section(
        llm,
        HISTORICAL_ANALYSIS_NARRATIVE_PROMPT,
        report.historical_analysis,
        "historical_analysis",
        total_usage,
        extra,
    )

    # ── Forecast Outlook ──────────────────────────────────────────────────
    report.forecast_outlook.narrative = _generate_section(
        llm,
        FORECAST_OUTLOOK_NARRATIVE_PROMPT,
        report.forecast_outlook,
        "forecast_outlook",
        total_usage,
        extra,
    )

    # ── Model Comparison ──────────────────────────────────────────────────
    report.model_comparison.narrative = _generate_section(
        llm,
        MODEL_COMPARISON_NARRATIVE_PROMPT,
        report.model_comparison,
        "model_comparison",
        total_usage,
        extra,
    )

    # ── Statistical Audit ─────────────────────────────────────────────────
    report.statistical_audit.narrative = _generate_section(
        llm,
        STATISTICAL_AUDIT_NARRATIVE_PROMPT,
        report.statistical_audit,
        "statistical_audit",
        total_usage,
        extra,
    )

    # ── Explainability ────────────────────────────────────────────────────
    report.explainability.narrative = _generate_section(
        llm,
        EXPLAINABILITY_NARRATIVE_PROMPT,
        report.explainability,
        "explainability",
        total_usage,
        extra,
    )

    # ── Recommendations ───────────────────────────────────────────────────
    for rec in report.recommendations:
        rec.narrative = _generate_section(
            llm,
            RECOMMENDATION_NARRATIVE_PROMPT,
            rec,
            "recommendation",
            total_usage,
            extra,
        )

    logger.info("Narrative generation complete. Tokens: %s", total_usage)
    return report, total_usage


def _generate_section(
    llm: Any,
    prompt: Any,
    section: Any,
    section_name: str,
    total_usage: dict[str, int],
    extra_instructions: str = "",
) -> str:
    """Generate narrative for a single section via the LLM.

    On failure, falls back to a deterministic template.

    Args:
        llm:               LLM instance.
        prompt:            ChatPromptTemplate for this section.
        section:           Pydantic section model to serialise as context.
        section_name:      Name for logging.
        total_usage:       Mutable token usage dict to accumulate.
        extra_instructions: Optional extra user instructions.

    Returns:
        Narrative text string.
    """
    section_json = json.dumps(section.model_dump(), default=str, indent=2)
    if extra_instructions:
        section_json += extra_instructions

    try:
        chain = prompt | llm
        inputs = {"section_json": section_json}
        response = chain.invoke(inputs)
        usage = extract_token_usage(
            response,
            input_text=estimate_input_text(prompt, inputs),
        )
        for key in total_usage:
            total_usage[key] += usage.get(key, 0)
        narrative = str(response.content).strip()
        section_data = section.model_dump()
        valid_models = _models_in_evidence(section_data)
        validation_warnings = validate_llm_output(narrative, valid_models, section_data)
        if section_name == "forecast_outlook":
            expected_model = str(section_data.get("metrics", {}).get("model_used", ""))
            validation_warnings.extend(
                _unexpected_model_references(narrative, expected_model)
            )
            validation_warnings.extend(
                _contradictory_forecast_pattern(
                    narrative,
                    str(section_data.get("metrics", {}).get("forecast_pattern", "")),
                )
            )
        elif section_name == "executive_summary":
            outlook = str(section_data.get("strategic_outlook", ""))
            if "seasonal / variable" in outlook.lower():
                validation_warnings.extend(
                    _contradictory_forecast_pattern(narrative, "Seasonal / variable")
                )
        elif section_name == "model_comparison":
            validation_warnings.extend(
                _contradictory_model_selection(
                    narrative, str(section_data.get("selected_model", ""))
                )
            )
        if validation_warnings:
            logger.warning(
                "Unsupported narrative for %s: %s — using fallback.",
                section_name,
                "; ".join(validation_warnings),
            )
            return _fallback_narrative(section, section_name)
        logger.debug("Narrative generated for %s", section_name)
        return narrative
    except Exception as exc:
        logger.warning(
            "Narrative LLM call failed for %s: %s — using fallback.",
            section_name,
            exc,
        )
        return _fallback_narrative(section, section_name)


def _models_in_evidence(value: Any) -> list[str]:
    """Collect recognized forecast model names from structured evidence."""
    serialized = json.dumps(value, default=str).lower()
    known = (
        "ARIMA",
        "SARIMA",
        "Holt-Winters",
        "EWMA",
        "Naive",
        "Seasonal Naive",
        "Mean Forecast",
        "Drift",
    )
    return [name for name in known if name.lower() in serialized]


def _unexpected_model_references(text: str, expected_model: str) -> list[str]:
    """Reject forecast prose that names a model other than the fitted model."""
    normalized = re.sub(r"[‐‑‒–—−]", "-", text).lower()
    known = (
        "ARIMA",
        "SARIMA",
        "Holt-Winters",
        "EWMA",
        "Naive",
        "Seasonal Naive",
        "Mean Forecast",
        "Drift",
        "Constant",
    )
    return [
        f"Narrative referenced {name}; fitted model is {expected_model}."
        for name in known
        if re.search(rf"(?<![a-z]){re.escape(name.lower())}(?![a-z])", normalized)
        and name.lower() != expected_model.lower()
    ]


def _contradictory_model_selection(text: str, expected_model: str) -> list[str]:
    """Reject prose that attributes selection to a different named model."""
    normalized = re.sub(r"[‐‑‒–—−]", "-", text)
    patterns = (
        r"(?P<model>ARIMA|SARIMA|Holt-Winters|EWMA)\s+(?:was|is)\s+(?:selected|chosen)",
        r"selected\s+(?:model\s+)?(?:was|is|:)\s*(?P<model>ARIMA|SARIMA|Holt-Winters|EWMA)",
    )
    warnings: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            model = match.group("model")
            if model.lower() != expected_model.lower():
                warnings.append(
                    f"Narrative selected {model}; production model is {expected_model}."
                )
    return warnings


def _contradictory_forecast_pattern(text: str, pattern: str) -> list[str]:
    """Reject directional-trajectory claims for a variable forecast path."""
    if pattern.lower() != "seasonal / variable":
        return []
    normalized = re.sub(r"[‐‑‒–—−]", "-", text).lower()
    directional_claims = (
        "downward trajectory",
        "upward trajectory",
        "downward trend",
        "upward trend",
        "trends downward",
        "trends upward",
    )
    return [
        f"Narrative described a {claim}; forecast pattern is {pattern}."
        for claim in directional_claims
        if claim in normalized
    ]


def _fallback_forecast_outlook(data: dict[str, Any]) -> str:
    """Build a deterministic fallback narrative for the forecast outlook.

    Reads metrics defensively with ``.get()`` defaults so the last-resort
    path cannot raise ``KeyError`` or ``IndexError``.

    Args:
        data: The model-dumped dict for the forecast outlook section.

    Returns:
        Fallback narrative string.
    """
    m = data.get("metrics", {})
    first_value = m.get("first_value", "N/A")
    last_value = m.get("last_value", "N/A")
    pct_change = m.get("pct_change", 0.0)
    peak_value = m.get("peak_value")
    peak_date = m.get("peak_date")
    horizon = m.get("horizon", 0)
    intervals = m.get("prediction_intervals") or []
    if intervals and isinstance(intervals, list) and len(intervals) > 1:
        conf_level = intervals[0].get("confidence_level", "95%")
        peak_text = (
            f" A temporary seasonal peak of {peak_value} is projected"
            f" for {peak_date}."
            if peak_value is not None
            else ""
        )
        return (
            f"The forecast projects a change from {first_value} to "
            f"{last_value} (a first-to-last change of {pct_change:+.1f}%) over "
            f"{horizon} periods.{peak_text} Forecasts carry uncertainty — "
            f"the {conf_level} "
            f"prediction range should be used for planning."
        )
    return f"The forecast projects {pct_change:+.1f}% change over {horizon} periods."


def _fallback_narrative(section: Any, section_name: str) -> str:
    """Generate a deterministic fallback narrative for a section.

    Uses hedged language and no jargon — the report remains usable
    without the LLM.

    Args:
        section:      Pydantic section model.
        section_name: Name of the section.

    Returns:
        Fallback narrative string.
    """
    data = section.model_dump()
    if section_name == "executive_summary":
        return (
            f"{data['strategic_outlook']} "
            f"Expected growth is {data['expected_growth']}. "
            f"Confidence is {data['confidence_level']}. "
            f"The primary risk is that {data['primary_risk'].lower()}. "
            f"Recommended action: {data['recommended_action']}"
        )
    if section_name == "data_quality":
        return f"Data quality is rated {data['rating']}. {data['rating_explanation']}"
    if section_name == "historical_analysis":
        return (
            f"The data shows a {data['trend_direction'].lower()} trend "
            f"over the observed period. "
            + (
                "A recurring seasonal pattern was detected. "
                if data.get("seasonal_period")
                else "No strong seasonal pattern was detected. "
            )
            + "The historical pattern provides the basis for the forecast."
        )
    if section_name == "forecast_outlook":
        return _fallback_forecast_outlook(data)
    if section_name == "model_comparison":
        return (
            f"The forecasting model was selected based on validation "
            f"performance and data characteristics. "
            f"{data['selection_rationale'][:200]}"
        )
    if section_name == "statistical_audit":
        return (
            f"The independent statistical assessment verdict is "
            f"{data['verdict'].upper()}. "
            + (
                "Key concerns were identified — see the recommended follow-up actions."
                if data.get("key_concerns")
                else "The analysis is well-supported by the evidence."
            )
        )
    if section_name == "explainability":
        findings = "; ".join(item["finding"] for item in data.get("findings", []))
        return (
            f"The forecast is based on the following findings: {findings}. "
            "These patterns were detected in the historical data and "
            "projected forward."
        )
    if section_name == "recommendation":
        return data.get("recommendation", "")
    return ""
