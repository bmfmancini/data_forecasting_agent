"""Model selection agent for the Data Forecaster backend.

Python is the source of statistical decisions. When empirical metrics are
available, a deterministic selection policy selects the best model. The
LLM is used for context, critique, and explanation only — it never
decides model rankings. When no empirical metrics are available (first
run), the LLM provides a suitability-based recommendation, but the
deterministic heuristic fallback is used if the LLM is unavailable or its
output is invalid.

All suitability-assessment, heuristic-fallback, and LLM-output parsing
logic is implemented as small, focused module-level helpers so that the
public :func:`run_model_selection_agent` stays readable and well below
the SonarQube Cognitive Complexity threshold.
"""

from __future__ import annotations

import math

import numpy as np

from core.llm_factory import get_llm
from core.logging_config import get_logger
from forecasting.selection_policy import (
    CandidateEvidence,
    SelectionOutcome,
    select_model_deterministic,
    validate_llm_output,
)
from prompts.model_selection_prompt import MODEL_SELECTION_PROMPT
from schemas import ModelSelectionResult, StatisticalResult
from utils.token_tracking import estimate_input_text, extract_token_usage

logger = get_logger(__name__)

_MODELS = ("ARIMA", "SARIMA", "Holt-Winters", "EWMA")
_METRIC_PRIORITY = ("MASE", "WAPE", "RMSE", "MAE", "MAPE")

# Unicode hyphen characters that the LLM may emit instead of ASCII '-'.
_UNICODE_HYPHENS = (
    "\u2010",
    "\u2011",
    "\u2012",
    "\u2013",
    "\u2014",
    "\u2015",
)


# ── Suitability assessments ───────────────────────────────────────────────────


def _hw_suitability(stat_result: StatisticalResult) -> str:
    """Build the Holt-Winters suitability assessment string.

    Args:
        stat_result: Output of the statistical analysis agent.

    Returns:
        A multi-line bullet list describing Holt-Winters suitability.
    """
    points: list[str] = []
    sp = stat_result.seasonal_period
    if sp and sp > 1:
        points.append(
            f"Seasonal period {sp} detected — Holt-Winters models seasonality natively."
        )
    else:
        points.append(
            "No clear seasonal period — Holt-Winters seasonal component may not help."
        )
    if stat_result.has_trend:
        points.append(
            f"Trend detected (slope={stat_result.trend_slope:.4f}) — "
            "Holt-Winters handles trend via exponential smoothing."
        )
    else:
        points.append(
            "No significant trend — simple exponential smoothing may suffice."
        )
    if not stat_result.is_stationary_adf:
        points.append(
            "Non-stationary series — Holt-Winters does not require pre-differencing."
        )
    points.append(
        "Holt-Winters is fast, interpretable, and robust on short-to-medium series."
    )
    return "Holt-Winters Assessment:\n" + "\n".join(f"- {p}" for p in points)


def _arima_suitability(stat_result: StatisticalResult) -> str:
    """Build the ARIMA suitability assessment string.

    Args:
        stat_result: Output of the statistical analysis agent.

    Returns:
        A multi-line bullet list describing ARIMA suitability.
    """
    points: list[str] = []
    sp = stat_result.seasonal_period
    if sp and sp > 1:
        points.append(
            f"Seasonal period {sp} detected — plain ARIMA ignores seasonality; "
            "SARIMA may be better."
        )
    else:
        points.append("No strong seasonality — ARIMA is appropriate.")
    if not stat_result.is_stationary_adf:
        points.append(
            "Non-stationary series — ARIMA handles this via differencing (d parameter)."
        )
    else:
        points.append("Series is stationary — ARIMA(p,0,q) sufficient.")
    if stat_result.has_trend:
        points.append("Trend present — ARIMA differencing (d≥1) will remove it.")
    points.append(
        "ARIMA is well-suited for non-seasonal series with complex autocorrelation."
    )
    return "ARIMA Assessment:\n" + "\n".join(f"- {p}" for p in points)


def _sarima_suitability(stat_result: StatisticalResult) -> str:
    """Build the SARIMA suitability assessment string.

    Args:
        stat_result: Output of the statistical analysis agent.

    Returns:
        A multi-line bullet list describing SARIMA suitability.
    """
    points: list[str] = []
    sp = stat_result.seasonal_period
    if sp and sp > 1:
        points.append(
            f"Seasonal period {sp} confirmed — SARIMA explicitly models seasonal "
            "AR/MA/I components."
        )
        points.append(
            "SARIMA is the gold standard for stationary-transformable seasonal series."
        )
    else:
        points.append(
            "No seasonal period detected — SARIMA seasonal component would overfit."
        )
    if not stat_result.is_stationary_adf:
        points.append(
            "Non-stationary — SARIMA seasonal differencing (D≥1) will address this."
        )
    points.append(
        "SARIMA requires more data than ARIMA (at least 2 full seasonal cycles)."
    )
    return "SARIMA Assessment:\n" + "\n".join(f"- {p}" for p in points)


def _ewma_suitability(stat_result: StatisticalResult) -> str:
    """Build the EWMA suitability assessment string.

    Args:
        stat_result: Output of the statistical analysis agent.

    Returns:
        A multi-line bullet list describing EWMA suitability.
    """
    points: list[str] = []
    if stat_result.has_trend:
        points.append(
            f"Trend detected (slope={stat_result.trend_slope:.4f}) — "
            "EWMA will lag behind trend changes."
        )
    else:
        points.append("No significant trend — EWMA performs well on stable series.")
    if stat_result.outlier_ratio > 0.05:
        points.append(
            f"High outlier ratio ({stat_result.outlier_ratio:.1%}) — "
            "EWMA is sensitive to outliers."
        )
    else:
        points.append("Low outlier count — EWMA will be robust.")
    if stat_result.is_white_noise:
        points.append("Series appears random — EWMA may be as good as complex models.")
    points.append(
        "EWMA is simple, fast, and works well for short-term forecasts with stable "
        "patterns."
    )
    points.append(
        "Best for real-time applications where simplicity and speed are priorities."
    )
    return "EWMA Assessment:\n" + "\n".join(f"- {p}" for p in points)


def _build_suitability_summary(stat_result: StatisticalResult) -> str:
    """Combine all four model suitability assessments into one summary.

    Args:
        stat_result: Output of the statistical analysis agent.

    Returns:
        A single string containing all four assessments separated by blank lines.
    """
    sections = [
        _hw_suitability(stat_result),
        _arima_suitability(stat_result),
        _sarima_suitability(stat_result),
        _ewma_suitability(stat_result),
    ]
    if stat_result.disabled_tests:
        sections.append(
            "User-disabled statistical evidence for this forecast:\n"
            f"- {', '.join(stat_result.disabled_tests)}"
        )
    return "\n\n".join(sections)


# ── Heuristic fallback ────────────────────────────────────────────────────────


def _heuristic_fallback(
    stat_result: StatisticalResult,
) -> tuple[str, dict[str, str | None]]:
    """Determine the fallback model and reasoning based on statistical properties.

    Args:
        stat_result: Output of the statistical analysis agent.

    Returns:
        A tuple of (fallback_model, reasoning_dict) where reasoning_dict maps
        each model name to a rejection reason (or ``None`` for the selected model).
    """
    preference = _heuristic_preference(stat_result)
    fallback_model = preference[0]
    reasoning: dict[str, str | None] = {
        "Holt-Winters": None,
        "ARIMA": None,
        "SARIMA": None,
        "EWMA": None,
    }
    for m in preference[1:]:
        reasoning[m] = _heuristic_rejection_reason(stat_result, m)
    return fallback_model, reasoning


def _heuristic_preference(stat_result: StatisticalResult) -> list[str]:
    """Return models ordered by heuristic suitability for the statistics.

    The first element is the most suitable fallback; subsequent entries are
    next-best alternatives in descending preference order.  Used to select a
    statistically sound fallback when the primary choice is excluded.

    Args:
        stat_result: Output of the statistical analysis agent.

    Returns:
        A list of model names ordered by heuristic preference.
    """
    sp = stat_result.seasonal_period or 1
    if sp > 1:
        return ["SARIMA", "Holt-Winters", "ARIMA", "EWMA"]
    if stat_result.has_trend and abs(stat_result.trend_slope) > 0.1:
        return ["Holt-Winters", "ARIMA", "SARIMA", "EWMA"]
    if stat_result.is_white_noise:
        return ["EWMA", "ARIMA", "Holt-Winters", "SARIMA"]
    return ["ARIMA", "Holt-Winters", "SARIMA", "EWMA"]


def _heuristic_rejection_reason(
    stat_result: StatisticalResult,
    model: str,
) -> str:
    """Return the heuristic rejection reason for a non-preferred model.

    Args:
        stat_result: Output of the statistical analysis agent.
        model:       Model name to explain.

    Returns:
        A short rejection reason string.
    """
    sp = stat_result.seasonal_period or 1
    if sp > 1:
        reasons = {
            "Holt-Winters": (
                "Strong seasonality makes SARIMA/Holt-Winters preferable."
            ),
            "ARIMA": "Seasonal pattern detected; plain ARIMA ignores seasonality.",
            "EWMA": "Seasonal patterns present; EWMA does not capture seasonality.",
        }
    elif stat_result.has_trend and abs(stat_result.trend_slope) > 0.1:
        reasons = {
            "ARIMA": "Trend present but Holt-Winters handles it more naturally.",
            "SARIMA": "No strong seasonality confirmed; SARIMA may overfit.",
            "EWMA": "Strong trend present; EWMA will lag behind trend changes.",
        }
    elif stat_result.is_white_noise:
        reasons = {
            "Holt-Winters": "Series appears random; simple EWMA may suffice.",
            "ARIMA": "Series is random noise; complex models may overfit.",
            "SARIMA": "No patterns detected; SARIMA would overfit.",
        }
    else:
        reasons = {
            "Holt-Winters": "No clear seasonal pattern or strong trend detected.",
            "SARIMA": "No seasonal period confirmed; SARIMA would overfit.",
            "EWMA": "Series has patterns that ARIMA can better capture.",
        }
    return reasons.get(model, "Not selected based on heuristic reasoning.")


def _adjust_excluded_fallback(
    stat_result: StatisticalResult,
    fallback_model: str,
    exclude_model: str | None,
) -> str:
    """Adjust the fallback model if it matches the excluded model.

    When the heuristic fallback matches the excluded model, re-evaluates the
    candidate fallback using the same statistical suitability ordering used
    by :func:`_heuristic_fallback`, preserving the exclude filter and
    choosing the next-best model rather than relying on tuple order.

    Args:
        stat_result:     Output of the statistical analysis agent.
        fallback_model:   The heuristic fallback model.
        exclude_model:    Optional model name to exclude from consideration.

    Returns:
        The (possibly adjusted) fallback model name.
    """
    if not exclude_model or fallback_model != exclude_model:
        return fallback_model
    preference = [m for m in _heuristic_preference(stat_result) if m != exclude_model]
    if preference:
        adjusted = preference[0]
        logger.info(
            "Fallback adjusted to exclude rejected model: %s -> %s",
            exclude_model,
            adjusted,
        )
        return adjusted
    return fallback_model


# ── LLM output parsing ────────────────────────────────────────────────────────


def _normalize_output(output: str) -> str:
    """Normalize LLM output by stripping markdown bold/italic and unicode hyphens.

    Args:
        output: Raw LLM output string.

    Returns:
        Normalized string suitable for model-name matching.
    """
    normalized = output.replace("**", "").replace("__", "")
    for hyphen in _UNICODE_HYPHENS:
        normalized = normalized.replace(hyphen, "-")
    return normalized


def _match_exact(normalized_lower: str) -> str | None:
    """Try an exact case-insensitive 'selected model: X' match.

    Args:
        normalized_lower: Lower-cased, normalized LLM output.

    Returns:
        The matched model name, or ``None`` if no exact match is found.
    """
    for m in _MODELS:
        if f"selected model: {m.lower()}" in normalized_lower:
            return m
    return None


def _match_line_scan(normalized: str) -> str | None:
    """Scan 'Selected model' lines for a model name as a broader fallback.

    Checks longest model names first to avoid substring matches (e.g. "ARIMA"
    inside "SARIMA").

    Args:
        normalized: Normalized LLM output string.

    Returns:
        The matched model name, or ``None`` if no match is found.
    """
    for line in normalized.splitlines():
        if "selected model" not in line.lower():
            continue
        upper_line = line.upper()
        for m in sorted(_MODELS, key=len, reverse=True):
            if m.upper() in upper_line:
                return m
    return None


def _parse_selected_model(output: str, fallback_model: str) -> str:
    """Parse the selected model from LLM output, falling back to heuristic.

    Args:
        output: Raw LLM output string.
        fallback_model: Heuristic fallback model if parsing fails.

    Returns:
        The selected model name.
    """
    normalized = _normalize_output(output)
    selected = _match_exact(normalized.lower())
    if selected is None:
        selected = _match_line_scan(normalized)
    return selected if selected is not None else fallback_model


def _finite_metric(metrics: dict[str, float], metric: str) -> float | None:
    """Return a finite metric value or ``None`` when unavailable."""
    value = metrics.get(metric)
    if value is None or not math.isfinite(value):
        return None
    return value


def _format_metric(metric: str, value: float) -> str:
    """Format a validation metric for business-readable model explanations."""
    if metric == "WAPE":
        return f"WAPE {value * 100:.2f}%"
    if metric == "MAPE":
        return f"MAPE {value:.2f}%"
    if metric == "MASE":
        return f"MASE {value:.4f}"
    return f"{metric} {value:.4f}"


def _primary_metric(
    all_metrics: dict[str, dict[str, float]] | None,
    model: str,
) -> tuple[str, float] | None:
    """Find the highest-priority available metric for a model."""
    if not all_metrics or model not in all_metrics:
        return None
    metrics = all_metrics[model]
    for metric in _METRIC_PRIORITY:
        value = _finite_metric(metrics, metric)
        if value is not None:
            return metric, value
    return None


def _metric_rejection_reason(
    model: str,
    selected_model: str,
    all_metrics: dict[str, dict[str, float]] | None,
) -> str | None:
    """Explain a rejected model using comparable validation error metrics."""
    rejected_metric = _primary_metric(all_metrics, model)
    selected_metric = _primary_metric(all_metrics, selected_model)
    if not rejected_metric or not selected_metric:
        return None
    metric, rejected_value = rejected_metric
    selected_metric_name, selected_value = selected_metric
    if metric == selected_metric_name and rejected_value > selected_value:
        return (
            "Higher forecast error on validation data: "
            f"{_format_metric(metric, rejected_value)} versus "
            f"{_format_metric(metric, selected_value)} for {selected_model}."
        )
    if metric == selected_metric_name and rejected_value <= selected_value:
        return (
            "Validation error was competitive, but statistical review or "
            "model assumptions made it a weaker production choice."
        )
    return (
        "Less favorable validation evidence than the selected model: "
        f"{_format_metric(metric, rejected_value)} compared with "
        f"{_format_metric(selected_metric_name, selected_value)} for "
        f"{selected_model}."
    )


def _statistical_fit_reason(
    stat_result: StatisticalResult,
    model: str,
    selected: bool,
) -> str:
    """Explain model suitability using statistical properties of the series."""
    sp = stat_result.seasonal_period or 1
    has_seasonality = sp > 1
    has_trend = stat_result.has_trend and abs(stat_result.trend_slope) > 0.1

    if model == "ARIMA":
        if has_seasonality:
            if selected:
                return (
                    "ARIMA was selected despite detected seasonality; users "
                    "should monitor residuals for any remaining seasonal "
                    "pattern."
                )
            return (
                "Plain ARIMA does not model the recurring seasonal cycle, so "
                "predictable seasonal patterns could remain in the errors."
            )
        return (
            "ARIMA fits the non-seasonal structure well because no reliable "
            "seasonal cycle was detected."
        )
    if model == "SARIMA":
        if has_seasonality:
            return (
                f"SARIMA is statistically appropriate because it can model the "
                f"detected {sp}-period seasonal cycle."
            )
        if selected:
            return (
                "SARIMA was selected despite limited seasonality evidence; "
                "this should be monitored for avoidable model complexity."
            )
        return (
            "SARIMA was rejected because there was insufficient evidence of a "
            "stable seasonal cycle; seasonal terms would add complexity and "
            "overfitting risk."
        )
    if model == "Holt-Winters":
        if has_seasonality or has_trend:
            if selected:
                return (
                    "Holt-Winters is a reasonable choice because it can "
                    "represent trend and seasonality in an interpretable way."
                )
            return (
                "Holt-Winters can represent trend and seasonality, but it is "
                "less flexible than the selected model when validation error "
                "or residual behavior is weaker."
            )
        if selected:
            return (
                "Holt-Winters was selected for its simple smoothing behavior, "
                "although there is limited trend or seasonality to model."
            )
        return (
            "Holt-Winters was rejected because no strong trend or seasonal "
            "pattern was detected, so its smoothing components may add little "
            "business value."
        )
    if model == "EWMA":
        if has_seasonality:
            if selected:
                return (
                    "EWMA was selected as the simplest stable option, but it "
                    "does not explicitly model seasonality."
                )
            return (
                "EWMA was rejected because it smooths recent values but does "
                "not explicitly model seasonality."
            )
        if has_trend or not stat_result.is_stationary_adf:
            if selected:
                return (
                    "EWMA was selected as a simple benchmark-style forecast, "
                    "but the changing baseline means it may lag trend changes."
                )
            return (
                "EWMA was rejected because the series is changing over time; "
                "simple smoothing can lag behind the trend and become unstable."
            )
        return (
            "EWMA is simple and stable, but it can miss autocorrelation that a "
            "time-series model can use for more accurate forecasts."
        )
    return (
        "Selected for the best balance of validation accuracy, assumptions, "
        "and reliability."
        if selected
        else "Rejected because it offered a weaker balance of accuracy and assumptions."
    )


def _build_selection_explanation(
    selected_model: str,
    stat_result: StatisticalResult,
    all_metrics: dict[str, dict[str, float]] | None,
    review_feedback: str | None = None,
) -> str:
    """Build a concise business-readable explanation for the selected model."""
    metric = _primary_metric(all_metrics, selected_model)
    parts = [f"Selected model: {selected_model}."]
    if metric:
        metric_name, value = metric
        parts.append(
            f"It had the strongest available validation evidence "
            f"({_format_metric(metric_name, value)}, lower is better)."
        )
    parts.append(_statistical_fit_reason(stat_result, selected_model, selected=True))
    if review_feedback:
        parts.append(
            "The selection also accounts for statistical review feedback from "
            "the prior run."
        )
    return " ".join(parts)


def _business_selection_reasons(
    selected_model: str,
    stat_result: StatisticalResult,
    all_metrics: dict[str, dict[str, float]] | None = None,
) -> dict[str, str | None]:
    """Build per-model business explanations for selection and rejection."""
    reasons: dict[str, str | None] = {}
    for model in _MODELS:
        if model == selected_model:
            reasons[model] = None
            continue
        metric_reason = _metric_rejection_reason(model, selected_model, all_metrics)
        fit_reason = _statistical_fit_reason(stat_result, model, selected=False)
        reasons[model] = (
            f"{metric_reason} {fit_reason}" if metric_reason else fit_reason
        )
    return reasons


# ── LLM invocation ───────────────────────────────────────────────────────────


_NOT_AVAILABLE = "not available"


def _format_metric_value(
    value: float | None,
    fmt: str,
    percent: bool = False,
) -> str:
    """Format a nullable metric value, returning ``_NOT_AVAILABLE`` for None/NaN."""
    if value is None or not np.isfinite(value):
        return _NOT_AVAILABLE
    if percent:
        return format(value * 100, fmt) + "%"
    return format(value, fmt)


def _format_metrics_text(
    all_metrics: dict[str, dict[str, float]],
) -> str:
    """Format all model error metrics into a readable text block.

    Args:
        all_metrics: Dict of model metrics, e.g.
            ``{"ARIMA": {"RMSE": x, "MAE": y, "MAPE": z}, ...}``.

    Returns:
        A formatted string listing each model's metrics, or empty string.
    """
    if not all_metrics:
        return ""
    lines = []
    for name, metrics in all_metrics.items():
        rmse_s = _format_metric_value(metrics.get("RMSE"), ".4f")
        mae_s = _format_metric_value(metrics.get("MAE"), ".4f")
        mape_s = _format_metric_value(metrics.get("MAPE"), ".2f", percent=True)
        wape_s = _format_metric_value(metrics.get("WAPE"), ".2f", percent=True)
        mase_s = _format_metric_value(metrics.get("MASE"), ".4f")
        lines.append(
            f"- {name}: RMSE={rmse_s}, MAE={mae_s}, MAPE={mape_s}, "
            f"WAPE={wape_s}, MASE={mase_s}"
        )
    return (
        "\n".join(lines)
        + "\n(WAPE is a robust alternative to MAPE; MASE < 1 is better than a naive forecast)"
    )


def _select_best_metric_model(
    all_metrics: dict[str, dict[str, float]],
    exclude_model: str | None = None,
) -> str | None:
    """Deterministically select the model with the lowest RMSE.

    Used during review-triggered retries when actual error metrics are
    available. Prioritizes MASE, then falls back to WAPE, RMSE, MAE, then
    MAPE if others are unavailable.

    Args:
        all_metrics: Dict of model metrics.
        exclude_model: Optional model to exclude from consideration.

    Returns:
        The name of the best model, or ``None`` if no metrics are available.
    """
    candidates = {k: v for k, v in all_metrics.items() if k != exclude_model}
    if not candidates:
        return None

    # Select by MASE (primary), then WAPE, RMSE, MAE, then MAPE as tie-breakers
    def _metric_key(
        item: tuple[str, dict[str, float]],
    ) -> tuple[float, float, float, float, float]:
        m = item[1]
        return (
            m.get("MASE", float("inf")),
            m.get("WAPE", float("inf")),
            m.get("RMSE", float("inf")),
            m.get("MAE", float("inf")),
            m.get("MAPE", float("inf")),
        )

    return min(candidates.items(), key=_metric_key)[0]


def _build_suitability_input(
    suitability_summary: str,
    review_feedback: str | None,
    exclude_model: str | None,
    all_metrics: dict[str, dict[str, float]] | None = None,
) -> str:
    """Augment the suitability summary with review feedback and exclusion context.

    When ``all_metrics`` is provided (during a review-triggered retry), the
    actual error metrics from the forecasting run are included so the LLM
    can make an evidence-based selection rather than relying solely on
    statistical properties.

    Args:
        suitability_summary: Base suitability summary for all models.
        review_feedback: Optional feedback from a prior statistical review.
        exclude_model: Optional model name to exclude from consideration.
        all_metrics: Optional dict of actual model error metrics from the
            prior forecasting run.

    Returns:
        The augmented suitability input string for the LLM prompt.
    """
    suitability_input = suitability_summary
    if all_metrics:
        metrics_text = _format_metrics_text(all_metrics)
        suitability_input += (
            "\n\n## ACTUAL ERROR METRICS (from prior forecasting run)\n"
            f"{metrics_text}\n\n"
            "These are real validation metrics (lower is better) from fitting "
            "all candidate models on the same train-test split. You MUST give "
            "strong preference to the model with the lowest MASE (Mean Absolute "
            "Scaled Error), as it is the most reliable indicator of forecast "
            "accuracy. A model with a lower MASE is objectively better, "
            "more accurate and should be preferred unless there is a strong "
            "methodological reason not to."
        )
    if review_feedback:
        suitability_input += (
            "\n\n## Statistical Review Feedback (from prior run)\n"
            f"{review_feedback}\n\n"
            "The previous model selection was reviewed and found to have "
            "issues. Please select a DIFFERENT model that addresses the "
            "reviewer's concerns."
        )
    if exclude_model:
        suitability_input += (
            "\n\n## Model Exclusion\n"
            f"The model '{exclude_model}' was previously selected and "
            f"rejected by the statistical review. Do NOT select "
            f"'{exclude_model}' again."
        )
    return suitability_input


def _invoke_llm(
    suitability_input: str,
) -> tuple[str, dict[str, int]] | None:
    """Invoke the LLM chain and return (output, token_usage) or None on failure.

    Args:
        suitability_input: The suitability input string for the prompt.

    Returns:
        A tuple of (output, token_usage), or ``None`` if the LLM call failed.
    """
    llm = get_llm(temperature=0)
    prompt = MODEL_SELECTION_PROMPT
    try:
        chain = prompt | llm
        inputs = {"suitability": suitability_input}
        response = chain.invoke(inputs)
        token_usage = extract_token_usage(
            response, input_text=estimate_input_text(prompt, inputs)
        )
        return str(response.content), token_usage
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "Model selection agent LLM call failed: %s — using heuristic.", exc
        )
        return None


# ── Deterministic policy helpers ──────────────────────────────────────────────


def _finite_or_none(value: float | None) -> float | None:
    """Return the value if finite, otherwise ``None``."""
    if value is not None and math.isfinite(value):
        return value
    return None


def _build_adapter_result(
    name: str,
    metrics: dict[str, float],
) -> "ForecastAdapterResult":
    """Build a :class:`ForecastAdapterResult` from a metrics dict.

    Args:
        name:    Model name.
        metrics: Dict of metric values (uppercase keys).

    Returns:
        A :class:`ForecastAdapterResult` with typed metrics.
    """
    from forecasting.contracts import (
        ForecastAdapterResult,
        ForecastFitStatus,
        ForecastMetrics,
    )

    rmse = _finite_or_none(metrics.get("RMSE"))
    mae = _finite_or_none(metrics.get("MAE"))
    mape = _finite_or_none(metrics.get("MAPE"))
    wape = _finite_or_none(metrics.get("WAPE"))
    mase = _finite_or_none(metrics.get("MASE"))

    has_finite = any(v is not None for v in (rmse, mae, mape, wape, mase))
    status = ForecastFitStatus.OK if has_finite else ForecastFitStatus.FAILED

    return ForecastAdapterResult(
        status=status,
        forecast=[],
        lower_ci=[],
        upper_ci=[],
        metrics=ForecastMetrics(
            rmse=rmse, mae=mae, mape=mape, wape=wape, mase=mase,
        ),
        fitted_configuration={"model": name},
    )


def _build_candidate_evidence(
    all_metrics: dict[str, dict[str, float]],
) -> list[CandidateEvidence]:
    """Build :class:`CandidateEvidence` objects from the metrics dict.

    The metrics dict uses uppercase keys (``"RMSE"``, ``"MAE"``, etc.) from
    the forecasting agent. This helper converts them to typed
    :class:`ForecastAdapterResult`-backed evidence so the deterministic
    policy can rank them.

    Args:
        all_metrics: Dict mapping model names to metric dicts.

    Returns:
        A list of :class:`CandidateEvidence` objects.
    """
    candidates: list[CandidateEvidence] = []
    for name, metrics in all_metrics.items():
        is_baseline = name.lower().startswith(
            ("naive", "seasonal naive", "mean", "drift")
        )
        adapter_result = _build_adapter_result(name, metrics)
        candidates.append(
            CandidateEvidence(
                name=name,
                adapter_result=adapter_result,
                is_baseline=is_baseline,
            )
        )
    return candidates


def _build_deterministic_explanation(
    outcome: SelectionOutcome,
    stat_result: StatisticalResult,
    all_metrics: dict[str, dict[str, float]],
    review_feedback: str | None,
) -> str:
    """Build a business-readable explanation for the deterministic selection.

    Args:
        outcome:        The deterministic selection outcome.
        stat_result:    Output of the statistical analysis agent.
        all_metrics:    Dict of all model error metrics.
        review_feedback: Optional review feedback from a prior run.

    Returns:
        A concise explanation string.
    """
    parts = [f"Selected model: {outcome.selected_model}."]
    metric = _primary_metric(all_metrics, outcome.selected_model)
    if metric:
        metric_name, value = metric
        parts.append(
            f"It had the strongest available validation evidence "
            f"({_format_metric(metric_name, value)}, lower is better)."
        )
    parts.append(
        _statistical_fit_reason(stat_result, outcome.selected_model, selected=True)
    )
    if outcome.tie_break_note:
        parts.append(f"Tie-breaking: {outcome.tie_break_note}")
    if outcome.exclusion_reasons:
        excluded = ", ".join(outcome.exclusion_reasons.keys())
        parts.append(f"Excluded candidates: {excluded}.")
    if review_feedback:
        parts.append(
            "The selection also accounts for statistical review feedback from "
            "the prior run."
        )
    metrics_text = _format_metrics_text(all_metrics)
    parts.append(f"\n\nValidation metrics considered:\n{metrics_text}")
    parts.append(f"\n[Statistical Review Feedback]: {review_feedback or 'N/A'}")
    return " ".join(parts)


# ── Public entry point ───────────────────────────────────────────────────────


def run_model_selection_agent(
    stat_result: StatisticalResult,
    review_feedback: str | None = None,
    exclude_model: str | None = None,
    all_metrics: dict[str, dict[str, float]] | None = None,
) -> ModelSelectionResult:
    """Use the LLM to reason over statistical findings and select the best model.

    When ``all_metrics`` is provided (during a review-triggered retry), the
    actual error metrics from the prior forecasting run are included in the
    LLM prompt.  Additionally, if the metrics clearly indicate a superior
    model, a deterministic override selects the best-performing model
    directly — this prevents the LLM from ignoring empirical evidence.

    Args:
        stat_result:     Output of the statistical analysis agent.
        review_feedback: Optional feedback from a prior statistical review,
                         injected into the LLM prompt to influence reselection.
        exclude_model:   Optional model name to exclude from consideration
                         (e.g., the previously selected model that was rejected
                         by the statistical review).
        all_metrics:     Optional dict of actual model error metrics from the
                         prior forecasting run, used to make an evidence-based
                         reselection during retry.

    Returns:
        The :class:`ModelSelectionResult` with the selected model and reasoning.
    """
    suitability_summary = _build_suitability_summary(stat_result)
    fallback_model, fallback_reasoning = _heuristic_fallback(stat_result)
    fallback_model = _adjust_excluded_fallback(
        stat_result, fallback_model, exclude_model
    )

    # ── Deterministic policy when empirical metrics are available ────────
    # When actual error metrics are available, the deterministic selection
    # policy is the source of truth. The LLM never decides model rankings.
    # The policy excludes failed/degraded candidates, ranks by the
    # configured loss metric, applies tie-breaking (simpler model wins
    # negligible differences), and retains baselines when no complex model
    # adds demonstrated value.
    if all_metrics:
        candidates = _build_candidate_evidence(all_metrics)
        outcome = select_model_deterministic(
            candidates,
            exclude_models=[exclude_model] if exclude_model else None,
            user_loss_preference="mase",
        )
        if outcome.selected_model:
            logger.info(
                "Deterministic policy selected '%s' (method=%s, rankable=%d).",
                outcome.selected_model,
                outcome.method,
                len(outcome.ranking),
            )
            metrics_text = _format_metrics_text(all_metrics)
            explanation = _build_deterministic_explanation(
                outcome, stat_result, all_metrics, review_feedback
            )
            reasons = _business_selection_reasons(
                outcome.selected_model, stat_result, all_metrics
            )
            return ModelSelectionResult(
                selected_model=outcome.selected_model,
                explanation=explanation,
                holt_winters_rejected_reason=reasons["Holt-Winters"],
                arima_rejected_reason=reasons["ARIMA"],
                sarima_rejected_reason=reasons["SARIMA"],
                ewma_rejected_reason=reasons["EWMA"],
                reasoning_steps=[
                    {
                        "thought": (
                            "Deterministic selection policy applied with "
                            "empirical metrics."
                        ),
                        "observation": metrics_text,
                    },
                ],
                token_usage={},
                selection_method="deterministic",
                selection_evidence={
                    "ranking": outcome.ranking,
                    "exclusion_reasons": outcome.exclusion_reasons,
                    "tie_break_note": outcome.tie_break_note,
                    "evidence_summary": outcome.evidence_summary,
                },
            )

    suitability_input = _build_suitability_input(
        suitability_summary, review_feedback, exclude_model, all_metrics
    )
    llm_result = _invoke_llm(suitability_input)

    if llm_result is None:
        return _build_heuristic_result(fallback_model, fallback_reasoning, stat_result)

    output, token_usage = llm_result
    selected_model = _parse_selected_model(output, fallback_model)
    reasons = _business_selection_reasons(selected_model, stat_result)
    explanation = (
        f"{output}\n\n"
        "Business-readable selection summary: "
        f"{_build_selection_explanation(selected_model, stat_result, None)}"
    )
    logger.info("Model selection agent output: %s", output[:200])
    logger.info("Selected model: %s", selected_model)

    return ModelSelectionResult(
        selected_model=selected_model,
        explanation=explanation,
        holt_winters_rejected_reason=reasons["Holt-Winters"],
        arima_rejected_reason=reasons["ARIMA"],
        sarima_rejected_reason=reasons["SARIMA"],
        ewma_rejected_reason=reasons["EWMA"],
        reasoning_steps=[
            {
                "thought": "Assessing suitability metrics for all models...",
                "observation": suitability_summary,
            },
            {
                "thought": "Finalizing model selection decision...",
                "observation": "Complete",
            },
        ],
        token_usage=token_usage,
        selection_method="llm",
        selection_evidence={},
    )


def _build_heuristic_result(
    fallback_model: str,
    fallback_reasoning: dict[str, str | None],
    stat_result: StatisticalResult,
) -> ModelSelectionResult:
    """Build a :class:`ModelSelectionResult` from the heuristic fallback.

    Args:
        fallback_model: The heuristic fallback model name.
        fallback_reasoning: Rejection reasons for each model.
        stat_result: Output of the statistical analysis agent (for context).

    Returns:
        A :class:`ModelSelectionResult` reflecting the heuristic selection.
    """
    explanation = (
        "Heuristic fallback used because the model-selection LLM was "
        "unavailable. "
        + _build_selection_explanation(fallback_model, stat_result, None)
    )
    reasons = _business_selection_reasons(fallback_model, stat_result)
    for model, reason in fallback_reasoning.items():
        if model != fallback_model and reason:
            reasons[model] = f"{reason} {reasons[model]}"
    logger.info("Selected model: %s", fallback_model)
    return ModelSelectionResult(
        selected_model=fallback_model,
        explanation=explanation,
        holt_winters_rejected_reason=reasons["Holt-Winters"],
        arima_rejected_reason=reasons["ARIMA"],
        sarima_rejected_reason=reasons["SARIMA"],
        ewma_rejected_reason=reasons["EWMA"],
        reasoning_steps=[
            {
                "thought": "Model selection agent failed; using heuristic.",
                "observation": f"Falling back to heuristic selection: {fallback_model}",
            }
        ],
        token_usage={},
        selection_method="heuristic",
        selection_evidence={},
    )
