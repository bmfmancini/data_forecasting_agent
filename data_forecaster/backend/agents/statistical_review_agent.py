"""Statistical review (QA) agent for the Data Forecaster backend.

This agent acts as a critic over the outputs of the statistical analysis,
model selection, and forecasting agents.  It performs deterministic
consistency pre-checks in Python and then invokes an LLM to produce a
structured review (verdict, flags, endorsements).

The review is injected into the report prompt and surfaced as a new field
on :class:`AnalysisResponse`.
"""

from __future__ import annotations

import re
import math
from typing import Any

from core.llm_factory import get_llm
from core.logging_config import get_logger
from forecasting.selection_policy import validate_llm_output
from prompts.statistical_review_prompt import STATISTICAL_REVIEW_PROMPT
from schemas import (
    ForecastResult,
    ModelSelectionResult,
    StatisticalResult,
    StatisticalReviewResult,
)
from utils.token_tracking import estimate_input_text, extract_token_usage

logger = get_logger(__name__)

_NOT_AVAILABLE = "not available"

_VERDICT_PATTERN = re.compile(r"Verdict:\s*(PASS|WARN|FAIL)", re.IGNORECASE)
_FLAG_PATTERN = re.compile(
    r"-\s*\[(CRITICAL|WARNING|INFO)\]\s*"
    r"\[agent:\s*(statistical|model_selection|forecasting)\]\s*"
    r"(.*?)\s*\|\s*Recommendation:\s*(.*)",
    re.IGNORECASE,
)


def _format_optional_metric(value: float | None, fmt: str) -> str:
    """Format a nullable metric for evidence passed to the reviewer."""
    return _NOT_AVAILABLE if value is None else format(value, fmt)


def _flag_key(flag: dict[str, Any]) -> str:
    """Return a semantic key so deterministic and LLM paraphrases do not repeat."""
    issue = str(flag.get("issue", "")).lower()
    if "residual" in issue and ("autocorrel" in issue or "ljung-box" in issue):
        return "residual_autocorrelation"
    if "change point" in issue or "structural break" in issue:
        return "change_points"
    if "outlier" in issue:
        return "outliers"
    return re.sub(r"[^a-z0-9]+", " ", issue).strip()


def _merge_review_flags(
    deterministic: list[dict[str, Any]], llm_flags: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge flags while preserving deterministic findings as canonical."""
    merged = list(deterministic)
    existing = {_flag_key(flag) for flag in merged}
    for flag in llm_flags:
        key = _flag_key(flag)
        if key not in existing:
            merged.append(flag)
            existing.add(key)
    return merged


def _check_seasonality_mismatch(
    stat_result: StatisticalResult,
    selected: str,
) -> dict[str, Any] | None:
    """Flag when ARIMA is selected despite a detected seasonal period.

    Args:
        stat_result: Output of the statistical analysis agent.
        selected:    The model selected by the model selection agent.

    Returns:
        A flag dict if the mismatch is detected, otherwise ``None``.
    """
    sp = stat_result.seasonal_period
    if sp and sp > 1 and selected == "ARIMA":
        return {
            "agent": "model_selection",
            "severity": "critical",
            "issue": (
                f"Seasonal period {sp} detected but ARIMA was selected. "
                "ARIMA does not model seasonality natively."
            ),
            "recommendation": (
                "Consider SARIMA or Holt-Winters which handle seasonality."
            ),
        }
    return None


def _check_nonstationary_ewma(
    stat_result: StatisticalResult,
    selected: str,
) -> dict[str, Any] | None:
    """Flag when EWMA is selected for a non-stationary series.

    Args:
        stat_result: Output of the statistical analysis agent.
        selected:    The model selected by the model selection agent.

    Returns:
        A flag dict if the mismatch is detected, otherwise ``None``.
    """
    if not stat_result.is_stationary_adf and selected == "EWMA":
        return {
            "agent": "model_selection",
            "severity": "critical",
            "issue": (
                "Series is non-stationary (ADF) but EWMA was selected. "
                "EWMA performs poorly on non-stationary data."
            ),
            "recommendation": (
                "Consider ARIMA/SARIMA with differencing for non-stationary series."
            ),
        }
    return None


def _check_high_mape(
    forecast_result: ForecastResult,
) -> dict[str, Any] | None:
    """Flag when the forecast MAPE exceeds 20%.

    Args:
        forecast_result: Output of the forecasting agent.

    Returns:
        A flag dict if MAPE is high, otherwise ``None``.
    """
    if forecast_result.mape is not None and forecast_result.mape > 20:
        return {
            "agent": "forecasting",
            "severity": "warning",
            "issue": (
                f"Forecast MAPE is {forecast_result.mape:.2f}%, indicating "
                "high prediction error."
            ),
            "recommendation": (
                "Review model adequacy, data transformations, or feature "
                "engineering to reduce error."
            ),
        }
    return None


def _check_outliers(
    stat_result: StatisticalResult,
) -> dict[str, Any] | None:
    """Flag when outliers are present but no remediation was recommended.

    Args:
        stat_result: Output of the statistical analysis agent.

    Returns:
        A flag dict if unaddressed outliers are detected, otherwise ``None``.
    """
    if stat_result.outlier_ratio > 0.05 and not stat_result.recommended_remediation:
        return {
            "agent": "statistical",
            "severity": "warning",
            "issue": (
                f"Outlier ratio is {stat_result.outlier_ratio:.1%} but no "
                "remediation was recommended."
            ),
            "recommendation": (
                "Consider IQR clipping, Z-score clipping, or robust "
                "transformations to mitigate outlier influence."
            ),
        }
    return None


def _check_trend_ewma_lag(
    stat_result: StatisticalResult,
    selected: str,
) -> dict[str, Any] | None:
    """Flag when EWMA is selected for a series with a detected trend.

    Args:
        stat_result: Output of the statistical analysis agent.
        selected:    The model selected by the model selection agent.

    Returns:
        A flag dict if EWMA is selected on a trending series, otherwise ``None``.
    """
    if stat_result.has_trend and selected == "EWMA":
        return {
            "agent": "model_selection",
            "severity": "warning",
            "issue": (
                f"Trend detected (slope={stat_result.trend_slope:.4f}) but "
                "EWMA was selected. EWMA lags behind trend changes."
            ),
            "recommendation": (
                "Consider Holt-Winters (trend component) or ARIMA with "
                "differencing for trending series."
            ),
        }
    return None


def _check_explanation_mismatch(
    model_selection: ModelSelectionResult,
    selected: str,
) -> dict[str, Any] | None:
    """Flag when the explanation mentions a different model than selected.

    The model selection agent's explanation text may mention a different
    model than the parsed selected_model field (e.g. due to markdown bold
    or unicode hyphens confusing the parser).  This creates narrative
    contradictions in the report.

    Args:
        model_selection: Output of the model selection agent.
        selected:         The model selected by the model selection agent.

    Returns:
        A flag dict if a mismatch is detected, otherwise ``None``.
    """
    explanation_lower = model_selection.explanation.lower()
    other_models = [
        m for m in ("ARIMA", "SARIMA", "Holt-Winters", "EWMA") if m != selected
    ]
    mentioned_models = [m for m in other_models if m.lower() in explanation_lower[:200]]
    # Only flag if another model is mentioned prominently in the first 200
    # chars of the explanation (where "Selected model:" text appears)
    # and the selected model itself is NOT mentioned there.
    if mentioned_models and selected.lower() not in explanation_lower[:200]:
        return {
            "agent": "model_selection",
            "severity": "critical",
            "issue": (
                f"Model selection explanation mentions '{mentioned_models[0]}' "
                f"but the parsed selected_model is '{selected}'. This "
                f"indicates a parsing mismatch that will cause narrative "
                f"contradictions in the report."
            ),
            "recommendation": (
                "Re-run model selection to resolve the explanation/"
                "selected_model mismatch."
            ),
        }
    return None


def _check_suboptimal_rmse(
    selected: str,
    all_metrics: dict[str, dict[str, float]],
) -> dict[str, Any] | None:
    """Flag when the selected model has significantly worse RMSE than the best.

    If the selected model has significantly worse RMSE than the best
    available model, flag it as a critical model selection issue.

    Args:
        selected:    The model selected by the model selection agent.
        all_metrics: Dict of all model metrics.

    Returns:
        A flag dict if the selected model is suboptimal, otherwise ``None``.
    """
    if not all_metrics or selected not in all_metrics:
        return None
    selected_rmse = all_metrics[selected].get("RMSE")
    if selected_rmse is None or not math.isfinite(selected_rmse):
        return None
    comparable = {
        name: metrics["RMSE"]
        for name, metrics in all_metrics.items()
        if metrics.get("RMSE") is not None and math.isfinite(metrics["RMSE"])
    }
    if not comparable:
        return None
    best_model = min(comparable, key=comparable.get)
    best_rmse = comparable[best_model]
    if best_model != selected and best_rmse > 0:
        ratio = selected_rmse / best_rmse
        if ratio > 1.5:
            return {
                "agent": "model_selection",
                "severity": "critical",
                "issue": (
                    f"Selected model '{selected}' has RMSE="
                    f"{selected_rmse:.4f} which is {ratio:.1f}x worse "
                    f"than the best model '{best_model}' (RMSE="
                    f"{best_rmse:.4f}). The selected model may be "
                    f"suboptimal."
                ),
                "recommendation": (
                    f"Re-evaluate model selection. '{best_model}' "
                    "achieved significantly better validation metrics."
                ),
            }
    return None


def _check_residual_autocorrelation(
    forecast_result: ForecastResult,
) -> dict[str, Any] | None:
    """Flag when model residuals are autocorrelated.

    Args:
        forecast_result: Output of the forecasting agent.

    Returns:
        A flag dict if autocorrelation is detected, otherwise ``None``.
    """
    diag = forecast_result.residual_diagnostics
    if diag and diag.is_uncorrelated is False:
        p_value = (
            f"{diag.ljung_box_p_value:.4f}"
            if diag.ljung_box_p_value is not None
            else _NOT_AVAILABLE
        )
        return {
            "agent": "forecasting",
            "severity": "warning",
            "issue": (
                f"Model residuals are autocorrelated (Ljung-Box p-value="
                f"{p_value}). The model may leave predictable "
                "patterns in the errors."
            ),
            "recommendation": (
                "Review model specification and compare residual diagnostics "
                "for viable alternatives before overriding the model selected "
                "by out-of-sample accuracy."
            ),
        }
    return None


def _check_residual_normality(
    forecast_result: ForecastResult,
) -> dict[str, Any] | None:
    """Flag when model residuals are not normally distributed.

    Args:
        forecast_result: Output of the forecasting agent.

    Returns:
        A flag dict if non-normality is detected, otherwise ``None``.
    """
    diag = forecast_result.residual_diagnostics
    if diag and diag.is_normal is False:
        p_value = (
            f"{diag.shapiro_wilk_p_value:.4f}"
            if diag.shapiro_wilk_p_value is not None
            else _NOT_AVAILABLE
        )
        return {
            "agent": "forecasting",
            "severity": "warning",
            "issue": (
                f"Model residuals are not normally distributed (Shapiro-Wilk "
                f"p-value={p_value})."
            ),
            "recommendation": (
                "The calculated prediction intervals may be unreliable. "
                "This can be caused by unhandled outliers or structural breaks."
            ),
        }
    return None


def _check_residual_mean(forecast_result: ForecastResult) -> dict[str, Any] | None:
    """Flag when model residuals have a non-zero mean."""
    diag = forecast_result.residual_diagnostics
    if diag and diag.is_zero_mean is False:
        return {
            "agent": "forecasting",
            "severity": "warning",
            "issue": f"Model residuals have a non-zero mean ({diag.mean:.4f}), indicating a systematic forecast bias.",
            "recommendation": "Review model specification; the forecast may be consistently too high or too low.",
        }
    return None


def _check_deterministic_policy_violation(
    model_selection: ModelSelectionResult,
    all_metrics: dict[str, dict[str, float]],
) -> dict[str, Any] | None:
    """Flag when a deterministic selection contradicts the metric evidence.

    The deterministic policy is the source of truth for model rankings. The
    review agent may only override it with a typed, code-recognized reason.
    This check detects when the selected model is objectively worse than
    another candidate by a large margin — a code-recognized reason to flag
    the selection.

    Args:
        model_selection: Output of the model selection agent.
        all_metrics:      Dict of all model metrics.

    Returns:
        A flag dict if a policy violation is detected, otherwise ``None``.
    """
    if model_selection.selection_method != "deterministic":
        return None
    selected = model_selection.selected_model
    if not all_metrics or selected not in all_metrics:
        return None
    selected_rmse = all_metrics[selected].get("RMSE")
    if selected_rmse is None or not math.isfinite(selected_rmse):
        return None
    comparable = {
        name: metrics["RMSE"]
        for name, metrics in all_metrics.items()
        if metrics.get("RMSE") is not None and math.isfinite(metrics["RMSE"])
    }
    if not comparable:
        return None
    best_model = min(comparable, key=comparable.get)
    best_rmse = comparable[best_model]
    if best_model != selected and best_rmse > 0:
        ratio = selected_rmse / best_rmse
        if ratio > 1.5:
            return {
                "agent": "model_selection",
                "severity": "critical",
                "issue": (
                    f"Deterministic policy selected '{selected}' (RMSE="
                    f"{selected_rmse:.4f}) which is {ratio:.1f}x worse than "
                    f"the best candidate '{best_model}' (RMSE="
                    f"{best_rmse:.4f})."
                ),
                "recommendation": (
                    "The selection policy may have excluded the best model "
                    "due to a status or assumption violation. Review the "
                    "selection_evidence for exclusion reasons."
                ),
            }
    return None


def _compute_override_eligibility(
    model_selection: ModelSelectionResult,
    pre_check_flags: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Determine whether the review can override the deterministic selection.

    The statistical review agent is a critic, but it cannot override the
    deterministic numerical policy without a typed, code-recognized reason.
    Only critical flags on the ``model_selection`` or ``forecasting``
    agents constitute valid override reasons.

    Args:
        model_selection: Output of the model selection agent.
        pre_check_flags:  Flags from the deterministic pre-check.

    Returns:
        A tuple of (can_override, override_reasons).
    """
    if model_selection.selection_method != "deterministic":
        # Non-deterministic selections (LLM, heuristic) can always be overridden.
        return True, []
    override_reasons = [
        flag["issue"]
        for flag in pre_check_flags
        if flag.get("severity") == "critical"
        and flag.get("agent") in ("model_selection", "forecasting")
    ]
    return bool(override_reasons), override_reasons


def _deterministic_pre_check(
    stat_result: StatisticalResult,
    model_selection: ModelSelectionResult,
    forecast_result: ForecastResult,
    all_metrics: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Run deterministic consistency checks on agent outputs.

    Args:
        stat_result:       Output of the statistical analysis agent.
        model_selection:   Output of the model selection agent.
        forecast_result:   Output of the forecasting agent.
        all_metrics:       Dict of all model metrics, e.g.
                           ``{"ARIMA": {"RMSE": x, "MAE": y, "MAPE": z}, ...}``.

    Returns:
        A list of flag dicts with keys ``agent``, ``severity``, ``issue``,
        and ``recommendation``.
    """
    selected = model_selection.selected_model

    checks: list[dict[str, Any] | None] = [
        _check_seasonality_mismatch(stat_result, selected),
        _check_nonstationary_ewma(stat_result, selected),
        _check_high_mape(forecast_result),
        _check_outliers(stat_result),
        _check_trend_ewma_lag(stat_result, selected),
        _check_explanation_mismatch(model_selection, selected),
        _check_suboptimal_rmse(selected, all_metrics),
        _check_deterministic_policy_violation(model_selection, all_metrics),
        _check_residual_autocorrelation(forecast_result),
        _check_residual_normality(forecast_result),
        _check_residual_mean(forecast_result),
    ]

    return [flag for flag in checks if flag is not None]


def _build_statistical_profile(stat_result: StatisticalResult) -> str:
    """Build a text summary of the statistical analysis for the LLM prompt."""
    return (
        f"- ADF stationary: {stat_result.is_stationary_adf} "
        f"(stat={stat_result.adf_statistic:.4f}, p={stat_result.adf_p_value:.4f})\n"
        f"- KPSS stationary: {stat_result.is_stationary_kpss} "
        f"(stat={stat_result.kpss_statistic:.4f}, p={stat_result.kpss_p_value:.4f})\n"
        f"- Trend: {stat_result.has_trend} (slope={stat_result.trend_slope:.4f})\n"
        f"- Seasonal period: {stat_result.seasonal_period}\n"
        f"- Dominant period: {stat_result.dominant_period}\n"
        f"- Outliers: {stat_result.outlier_count} "
        f"({stat_result.outlier_ratio:.1%})\n"
        f"- White noise: {stat_result.is_white_noise} "
        f"(p={stat_result.white_noise_p_value:.4f})\n"
        f"- Recommended remediation: {stat_result.recommended_remediation}\n"
        f"- Disabled statistical tests: {stat_result.disabled_tests}\n"
        f"- Summary: {stat_result.summary}"
    )


def _build_model_selection_text(
    model_selection: ModelSelectionResult,
) -> str:
    """Build a text summary of the model selection for the LLM prompt."""
    return (
        f"- Selected model: {model_selection.selected_model}\n"
        f"- Explanation: {model_selection.explanation}\n"
        f"- HW rejected: {model_selection.holt_winters_rejected_reason}\n"
        f"- ARIMA rejected: {model_selection.arima_rejected_reason}\n"
        f"- SARIMA rejected: {model_selection.sarima_rejected_reason}\n"
        f"- EWMA rejected: {model_selection.ewma_rejected_reason}"
    )


def _build_forecast_text(forecast_result: ForecastResult) -> str:
    """Build a text summary of the forecast for the LLM prompt."""
    forecast_sample = [round(v, 2) for v in forecast_result.forecast[:10]]
    residual_text = _NOT_AVAILABLE.capitalize() + "."
    if diag := forecast_result.residual_diagnostics:
        ljung_box = (
            f"{diag.ljung_box_p_value:.4f}"
            if diag.ljung_box_p_value is not None
            else "disabled"
        )
        shapiro = (
            f"{diag.shapiro_wilk_p_value:.4f}"
            if diag.shapiro_wilk_p_value is not None
            else "disabled"
        )
        residual_text = (
            f"Mean={diag.mean:.4f} (is_zero: {diag.is_zero_mean}), "
            f"Ljung-Box p={ljung_box} (is_uncorrelated: {diag.is_uncorrelated}), "
            f"Shapiro-Wilk p={shapiro} (is_normal: {diag.is_normal}), "
            f"Disabled: {diag.disabled_tests}"
        )

    candidate_text = (
        "; ".join(
            f"{candidate.model}={candidate.status.value}"
            + (f" ({candidate.failure_reason})" if candidate.failure_reason else "")
            for candidate in forecast_result.candidate_results
        )
        or _NOT_AVAILABLE
    )

    return (
        f"- Model used: {forecast_result.model_used}\n"
        f"- RMSE: {_format_optional_metric(forecast_result.rmse, '.4f')}\n"
        f"- MAE: {_format_optional_metric(forecast_result.mae, '.4f')}\n"
        f"- MAPE: {_format_optional_metric(forecast_result.mape, '.2f')}%\n"
        f"- Candidate fit statuses: {candidate_text}\n"
        f"- Forecast sample (first 10): {forecast_sample}\n"
        f"- Residual Diagnostics: {residual_text}\n"
        f"- Forecast dates: "
        f"{forecast_result.forecast_dates[:3]}...{forecast_result.forecast_dates[-3:]}"
    )


def _build_all_metrics_text(
    all_metrics: dict[str, dict[str, float | None]],
) -> str:
    """Build a text summary of all model metrics for the LLM prompt."""
    lines = []
    for name, metrics in all_metrics.items():
        lines.append(
            f"- {name}: RMSE={_format_optional_metric(metrics.get('RMSE'), '.4f')}, "
            f"MAE={_format_optional_metric(metrics.get('MAE'), '.4f')}, "
            f"MAPE={_format_optional_metric(metrics.get('MAPE'), '.2f')}%"
        )
    return "\n".join(lines) if lines else "No metrics available."


def _format_pre_check_flags(flags: list[dict[str, Any]]) -> str:
    """Format deterministic pre-check flags for the LLM prompt."""
    if not flags:
        return "None"
    lines = []
    for flag in flags:
        lines.append(
            f"- [{flag['severity'].upper()}] [agent: {flag['agent']}] "
            f"{flag['issue']} | Recommendation: {flag['recommendation']}"
        )
    return "\n".join(lines)


def _parse_verdict(text: str) -> str:
    """Extract the verdict from the LLM response text.

    Args:
        text: The raw LLM response content.

    Returns:
        One of ``"pass"``, ``"warn"``, ``"fail"``.  Defaults to ``"warn"``
        if no verdict is found.
    """
    match = _VERDICT_PATTERN.search(text)
    if match:
        return match.group(1).lower()
    return "warn"


def _parse_flags(text: str) -> list[dict[str, Any]]:
    """Extract structured flags from the LLM response text.

    Args:
        text: The raw LLM response content.

    Returns:
        A list of flag dicts with keys ``agent``, ``severity``, ``issue``,
        and ``recommendation``.
    """
    flags: list[dict[str, Any]] = []
    for match in _FLAG_PATTERN.finditer(text):
        severity = match.group(1).lower()
        agent = match.group(2).lower()
        issue = match.group(3).strip()
        recommendation = match.group(4).strip()
        if issue.lower() == "none":
            continue
        flags.append(
            {
                "agent": agent,
                "severity": severity,
                "issue": issue,
                "recommendation": recommendation,
            }
        )
    return flags


def _parse_endorsements(text: str) -> list[str]:
    """Extract endorsements from the LLM response text.

    Args:
        text: The raw LLM response content.

    Returns:
        A list of endorsement strings.
    """
    endorsements: list[str] = []
    # Find the endorsements section
    section_match = re.search(
        r"## Endorsements\s*\n(.*)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not section_match:
        return endorsements
    section = section_match.group(1)
    # Stop at the next section header if present
    next_section = re.search(r"##\s", section)
    if next_section:
        section = section[: next_section.start()]
    for line in section.strip().splitlines():
        line = line.strip()
        if line.startswith("- ") and line.lower() != "- none":
            endorsements.append(line[2:].strip())
    return endorsements


def _parse_summary(text: str) -> str:
    """Extract the summary section from the LLM response text.

    Args:
        text: The raw LLM response content.

    Returns:
        The summary text, or a default message if not found.
    """
    section_match = re.search(
        r"## Summary\s*\n(.*)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not section_match:
        return "Statistical review completed."
    section = section_match.group(1)
    # Stop at the next section header if present
    next_section = re.search(r"##\s", section)
    if next_section:
        section = section[: next_section.start()]
    return section.strip()


def _compute_verdict(
    base_verdict: str,
    deterministic_flags: list[dict[str, Any]],
) -> str:
    """Adjust the LLM verdict based on deterministic critical flags.

    Deterministic critical flags force the verdict to at least ``"warn"``.
    If the LLM also returned ``"fail"``, ``"fail"`` is preserved.

    Args:
        base_verdict:         The verdict parsed from the LLM response.
        deterministic_flags:  Flags from the deterministic pre-check.

    Returns:
        The final verdict string.
    """
    has_critical = any(f["severity"] == "critical" for f in deterministic_flags)
    if has_critical and base_verdict == "pass":
        return "warn"
    return base_verdict


def run_statistical_review_agent(
    stat_result: StatisticalResult,
    model_selection: ModelSelectionResult,
    forecast_result: ForecastResult,
    all_metrics: dict[str, dict[str, float]],
) -> StatisticalReviewResult:
    """Run the statistical review (QA) agent over pipeline outputs.

    Performs deterministic consistency pre-checks and then invokes an LLM
    to produce a structured review with verdict, flags, and endorsements.

    Args:
        stat_result:       Output of the statistical analysis agent.
        model_selection:   Output of the model selection agent.
        forecast_result:   Output of the forecasting agent.
        all_metrics:       Dict of all model metrics, e.g.
                           ``{"ARIMA": {"RMSE": x, "MAE": y, "MAPE": z}, ...}``.

    Returns:
        A :class:`StatisticalReviewResult` with verdict, flags, endorsements,
        summary, reasoning_steps, and token_usage.
    """
    logger.info("Statistical review agent starting.")

    # ── Deterministic pre-check ───────────────────────────────────────────────
    pre_check_flags = _deterministic_pre_check(
        stat_result, model_selection, forecast_result, all_metrics
    )
    if pre_check_flags:
        logger.info("Deterministic pre-check found %d flags.", len(pre_check_flags))

    # ── Build prompt inputs ───────────────────────────────────────────────────
    statistical_profile = _build_statistical_profile(stat_result)
    model_selection_text = _build_model_selection_text(model_selection)
    forecast_text = _build_forecast_text(forecast_result)
    all_metrics_text = _build_all_metrics_text(all_metrics)
    pre_check_text = _format_pre_check_flags(pre_check_flags)

    prompt = STATISTICAL_REVIEW_PROMPT
    token_usage: dict[str, int] = {}
    reasoning_steps: list[dict[str, Any]] = []
    narrative_uncertainty = "deterministic_precheck"

    try:
        llm = get_llm(temperature=0)
        chain = prompt | llm
        inputs = {
            "statistical_profile": statistical_profile,
            "model_selection": model_selection_text,
            "forecast_results": forecast_text,
            "all_metrics": all_metrics_text,
            "pre_check_flags": pre_check_text,
        }
        response = chain.invoke(inputs)
        output = response.content
        token_usage = extract_token_usage(
            response, input_text=estimate_input_text(prompt, inputs)
        )
        logger.info("Statistical review LLM output: %s", output[:200])

        verdict = _parse_verdict(output)
        narrative_uncertainty = "validated_llm_interpretation"
        llm_flags = _parse_flags(output)
        endorsements = _parse_endorsements(output)
        summary = _parse_summary(output)

        validation_warnings = validate_llm_output(
            output,
            list(all_metrics),
            {
                "all_metrics": all_metrics,
                "selected_model": model_selection.selected_model,
            },
        )
        for warning in validation_warnings:
            llm_flags.append(
                {
                    "agent": "statistical",
                    "severity": "warning",
                    "issue": f"Unsupported LLM review claim: {warning}",
                    "recommendation": "Use only the supplied deterministic evidence.",
                }
            )

        all_flags = _merge_review_flags(pre_check_flags, llm_flags)

        verdict = _compute_verdict(verdict, pre_check_flags)

        reasoning_steps = [
            {
                "thought": "Running deterministic consistency pre-checks...",
                "observation": pre_check_text,
            },
            {
                "thought": "LLM reviewing pipeline outputs for methodological soundness...",
                "observation": output,
            },
        ]

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Statistical review agent LLM call failed: %s — using pre-check only.",
            exc,
            exc_info=True,
        )
        verdict = "warn" if pre_check_flags else "pass"
        all_flags = list(pre_check_flags)
        endorsements = []
        summary = (
            "Statistical review completed via deterministic pre-check only "
            "(LLM unavailable)."
        )
        reasoning_steps = [
            {
                "thought": f"Statistical review agent LLM failed: {exc}",
                "observation": "Falling back to deterministic pre-check flags only.",
            }
        ]

    logger.info(
        "Statistical review complete: verdict=%s flags=%d", verdict, len(all_flags)
    )

    can_override, override_reasons = _compute_override_eligibility(
        model_selection, pre_check_flags
    )

    return StatisticalReviewResult(
        verdict=verdict,
        flags=all_flags,
        endorsements=endorsements,
        summary=summary,
        reasoning_steps=reasoning_steps,
        token_usage=token_usage,
        can_override_selection=can_override,
        override_reasons=override_reasons,
        narrative_claims=[
            {
                "claim": summary,
                "evidence_references": ["deterministic_pre_check", "all_metrics"],
                "uncertainty": narrative_uncertainty,
            }
        ],
    )
