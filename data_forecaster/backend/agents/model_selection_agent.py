"""Model selection agent for the Data Forecaster backend.

This module uses an LLM to reason over statistical findings and select the
best forecasting model.  All suitability-assessment, heuristic-fallback, and
LLM-output parsing logic is implemented as small, focused module-level
helpers so that the public :func:`run_model_selection_agent` stays readable
and well below the SonarQube Cognitive Complexity threshold.
"""

from __future__ import annotations

from core.llm_factory import get_llm
from core.logging_config import get_logger
from prompts.model_selection_prompt import MODEL_SELECTION_PROMPT
from schemas import ModelSelectionResult, StatisticalResult
from utils.token_tracking import estimate_input_text, extract_token_usage

logger = get_logger(__name__)

_MODELS = ("ARIMA", "SARIMA", "Holt-Winters", "EWMA")

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
    return "\n\n".join([
        _hw_suitability(stat_result),
        _arima_suitability(stat_result),
        _sarima_suitability(stat_result),
        _ewma_suitability(stat_result),
    ])


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
    stat_result: StatisticalResult, model: str,
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
    preference = [
        m for m in _heuristic_preference(stat_result) if m != exclude_model
    ]
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


def _rejection_reasons(
    selected_model: str, reason: str = "Not selected based on LLM reasoning.",
) -> dict[str, str | None]:
    """Build rejection reasons for non-selected models.

    Args:
        selected_model: The model that was selected.
        reason:         Rejection reason text applied to all non-selected
            models.  Defaults to the LLM-selection wording; deterministic
            override paths should pass a metric/override-specific message.

    Returns:
        A dict mapping each model name to a rejection reason (or ``None`` for
        the selected model).
    """
    reasons: dict[str, str | None] = {}
    for m in _MODELS:
        if m == selected_model:
            reasons[m] = None
        else:
            reasons[m] = reason
    return reasons


# ── LLM invocation ───────────────────────────────────────────────────────────


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
        rmse = metrics.get("RMSE", float("nan"))
        mae = metrics.get("MAE", float("nan"))
        mape = metrics.get("MAPE", float("nan"))
        lines.append(
            f"- {name}: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%"
        )
    return "\n".join(lines)


def _select_best_metric_model(
    all_metrics: dict[str, dict[str, float]],
    exclude_model: str | None = None,
) -> str | None:
    """Deterministically select the model with the lowest RMSE.

    Used during review-triggered retries when actual error metrics are
    available.  Falls back to MAE then MAPE if RMSE is unavailable.

    Args:
        all_metrics: Dict of model metrics.
        exclude_model: Optional model to exclude from consideration.

    Returns:
        The name of the best model, or ``None`` if no metrics are available.
    """
    candidates = {
        k: v for k, v in all_metrics.items() if k != exclude_model
    }
    if not candidates:
        return None
    # Select by RMSE (primary), then MAE, then MAPE as tie-breakers
    def _metric_key(item: tuple[str, dict[str, float]]) -> tuple[float, float, float]:
        m = item[1]
        return (
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
            "all candidate models on the same train-test split. You MUST "
            "weight these empirical results heavily in your selection — "
            "a model with substantially lower RMSE/MAE/MAPE is objectively "
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
        logger.warning("Model selection agent LLM call failed: %s — using heuristic.", exc)
        return None


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

    # ── Deterministic override when empirical metrics are available ────────
    # During a review-triggered retry, if actual error metrics are available,
    # deterministically select the best-performing model rather than relying
    # on the LLM.  This prevents the LLM from re-selecting a suboptimal model
    # based on statistical properties alone.
    if all_metrics:
        best_model = _select_best_metric_model(all_metrics, exclude_model)
        if best_model:
            logger.info(
                "Deterministic override: selecting best-metric model '%s' "
                "based on empirical error metrics.",
                best_model,
            )
            metrics_text = _format_metrics_text(all_metrics)
            explanation = (
                f"Model re-selected based on empirical validation metrics "
                f"from the prior forecasting run. '{best_model}' achieved "
                f"the lowest error scores:\n{metrics_text}\n\n"
                f"[Statistical Review Feedback]: {review_feedback or 'N/A'}"
            )
            reasons = _rejection_reasons(
                best_model,
                reason=(
                    "Not selected — deterministic override chose the model "
                    "with the lowest empirical validation error."
                ),
            )
            return ModelSelectionResult(
                selected_model=best_model,
                explanation=explanation,
                holt_winters_rejected_reason=reasons["Holt-Winters"],
                arima_rejected_reason=reasons["ARIMA"],
                sarima_rejected_reason=reasons["SARIMA"],
                ewma_rejected_reason=reasons["EWMA"],
                reasoning_steps=[
                    {
                        "thought": (
                            "Review-triggered retry with empirical metrics "
                            "available — selecting best-performing model."
                        ),
                        "observation": metrics_text,
                    },
                ],
                token_usage={},
            )

    suitability_input = _build_suitability_input(
        suitability_summary, review_feedback, exclude_model, all_metrics
    )
    llm_result = _invoke_llm(suitability_input)

    if llm_result is None:
        return _build_heuristic_result(
            fallback_model, fallback_reasoning, stat_result
        )

    output, token_usage = llm_result
    selected_model = _parse_selected_model(output, fallback_model)
    reasons = _rejection_reasons(selected_model)
    logger.info("Model selection agent output: %s", output[:200])
    logger.info("Selected model: %s", selected_model)

    return ModelSelectionResult(
        selected_model=selected_model,
        explanation=output,
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
    sp = stat_result.seasonal_period
    explanation = (
        f"Heuristic selection: {fallback_model} chosen based on seasonal period={sp}."
    )
    logger.info("Selected model: %s", fallback_model)
    return ModelSelectionResult(
        selected_model=fallback_model,
        explanation=explanation,
        holt_winters_rejected_reason=fallback_reasoning["Holt-Winters"],
        arima_rejected_reason=fallback_reasoning["ARIMA"],
        sarima_rejected_reason=fallback_reasoning["SARIMA"],
        ewma_rejected_reason=fallback_reasoning["EWMA"],
        reasoning_steps=[
            {
                "thought": "Model selection agent failed; using heuristic.",
                "observation": f"Falling back to heuristic selection: {fallback_model}",
            }
        ],
        token_usage={},
    )
