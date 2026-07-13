"""Statistical analysis agent backed by one typed diagnostic pipeline."""

from __future__ import annotations

import re

import pandas as pd

from core.llm_factory import get_llm
from core.logging_config import get_logger
from forecasting.diagnostics import (
    assess_stationarity,
    assess_trend,
    detect_anomalies,
    detect_change_points_calibrated,
    detect_seasonality,
    test_white_noise,
)
from prompts.statistical_analysis_prompt import STATISTICAL_ANALYSIS_PROMPT
from schemas import StatisticalResult
from utils.token_tracking import estimate_input_text, extract_token_usage

logger = get_logger(__name__)


def _status_maps(*evidence: tuple[str, object]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return serializable diagnostic statuses and warnings."""
    statuses: dict[str, str] = {}
    warnings: dict[str, list[str]] = {}
    for name, item in evidence:
        statuses[name] = item.status.value
        warnings[name] = list(item.warnings)
    return statuses, warnings


def run_statistical_agent(
    series: pd.Series,
    seasonal_period: int = 12,
    user_domain: str = "General",
    disabled_tests: list[str] | None = None,
) -> StatisticalResult:
    """Compute typed evidence in Python and use the LLM only for explanation."""
    disabled = set(disabled_tests or [])
    values = series.dropna().astype(float)
    is_constant = values.nunique() <= 1

    seasonality = detect_seasonality(
        values,
        metadata_period=None if is_constant else seasonal_period,
        disabled="periodogram" in disabled or "stl" in disabled,
    )
    if is_constant:
        seasonality = seasonality.model_copy(
            update={
                "selected_period": 1,
                "selection_provenance": "constant_series",
                "seasonal_strength": 0.0,
                "candidate_periods": [],
                "dominant_period": None,
            }
        )
    stationarity = assess_stationarity(
        values, disabled="adf" in disabled and "kpss" in disabled
    )
    trend = assess_trend(values, disabled="trend" in disabled)
    anomalies = detect_anomalies(
        values,
        seasonal_period=seasonality.selected_period,
        disabled="outliers" in disabled,
    )
    change_points = detect_change_points_calibrated(
        values, disabled="change_points" in disabled
    )
    white_noise = (
        {"p_value": None, "is_white_noise": None, "interpretation": "disabled"}
        if "white_noise" in disabled
        else test_white_noise(values)
    )

    statuses, diagnostic_warnings = _status_maps(
        ("seasonality", seasonality),
        ("stationarity", stationarity),
        ("trend", trend),
        ("anomalies", anomalies),
        ("change_points", change_points),
    )
    profile = {
        "domain_context": user_domain,
        "seasonality": seasonality.model_dump(mode="json"),
        "stationarity": stationarity.model_dump(mode="json"),
        "trend": trend.model_dump(mode="json"),
        "anomalies": anomalies.model_dump(mode="json"),
        "change_points": change_points.model_dump(mode="json"),
        "white_noise": white_noise,
        "disabled_tests": sorted(disabled),
    }

    inferred_domain = user_domain
    summary = (
        f"Stationarity classification: {stationarity.classification}. "
        f"Selected seasonal period: {seasonality.selected_period}. "
        f"Trend detected: {trend.has_trend}. "
        f"Adjusted anomalies: {anomalies.anomaly_count}."
    )
    token_usage: dict[str, int] = {}
    reasoning_steps = [
        {
            "thought": "Computed typed statistical evidence in Python.",
            "observation": str(profile),
        }
    ]
    try:
        prompt = STATISTICAL_ANALYSIS_PROMPT
        inputs = {"profile": str(profile)}
        response = (prompt | get_llm(temperature=0)).invoke(inputs)
        narrative = str(response.content)
        if narrative.strip():
            summary = narrative
        token_usage = extract_token_usage(
            response, input_text=estimate_input_text(prompt, inputs)
        )
        if user_domain in {"Skip / Let AI Guess", "Other (Custom)"}:
            match = re.search(r"DOMAIN:\s*([^\n\.]+)", summary, re.IGNORECASE)
            inferred_domain = match.group(1).strip() if match else "General / Unknown"
        reasoning_steps.append(
            {
                "thought": "Generated a qualitative explanation of typed evidence.",
                "observation": "Complete",
            }
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Statistical narrative unavailable: %s", exc)

    # LLM prose cannot request transformations. Structured change-point
    # evidence may request a follow-up analysis but never mutates observations.
    remediation = (
        ["change_point_analysis"] if change_points.n_change_points > 0 else []
    )
    adf_p = stationarity.adf_p_value
    kpss_p = stationarity.kpss_p_value
    return StatisticalResult(
        is_stationary_adf=bool(adf_p is not None and adf_p < 0.05),
        adf_statistic=0.0,
        adf_p_value=adf_p if adf_p is not None else 1.0,
        is_stationary_kpss=bool(kpss_p is not None and kpss_p >= 0.05),
        kpss_statistic=0.0,
        kpss_p_value=kpss_p if kpss_p is not None else 1.0,
        has_trend=trend.has_trend,
        trend_slope=trend.slope,
        outlier_count=anomalies.anomaly_count,
        outlier_ratio=anomalies.anomaly_ratio,
        is_white_noise=bool(white_noise.get("is_white_noise")),
        white_noise_p_value=white_noise.get("p_value") or 1.0,
        recommended_remediation=remediation,
        domain=inferred_domain,
        seasonal_period=seasonality.selected_period,
        dominant_period=seasonality.dominant_period,
        disabled_tests=sorted(disabled),
        summary=summary,
        reasoning_steps=reasoning_steps,
        token_usage=token_usage,
        stationarity_classification=stationarity.classification,
        seasonal_strength=seasonality.seasonal_strength,
        seasonal_selection_provenance=seasonality.selection_provenance,
        anomaly_count_adjusted=anomalies.anomaly_count,
        anomaly_ratio_adjusted=anomalies.anomaly_ratio,
        change_point_count=change_points.n_change_points,
        variance_break_count=len(change_points.variance_breaks),
        trend_effect_size=trend.effect_size,
        trend_p_value_robust=trend.p_value,
        diagnostic_statuses=statuses,
        diagnostic_warnings=diagnostic_warnings,
        seasonality_candidates=seasonality.candidate_periods,
        observed_frequency=seasonality.observed_frequency,
        narrative_label="llm_interpretation_not_numerical_evidence",
        narrative_evidence=list(statuses),
    )
