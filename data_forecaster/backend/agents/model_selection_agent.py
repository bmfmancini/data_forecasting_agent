from __future__ import annotations

from core.llm_factory import get_llm
from core.logging_config import get_logger
from prompts.model_selection_prompt import MODEL_SELECTION_PROMPT
from schemas import ModelSelectionResult, StatisticalResult
from utils.token_tracking import estimate_input_text, extract_token_usage

logger = get_logger(__name__)

_MODELS = ("ARIMA", "SARIMA", "Holt-Winters", "EWMA")


def run_model_selection_agent(
    stat_result: StatisticalResult,
    review_feedback: str | None = None,
    exclude_model: str | None = None,
) -> ModelSelectionResult:
    """Use the LLM to reason over statistical findings and select the best model.

    Args:
        stat_result:      Output of the statistical analysis agent.
        review_feedback:  Optional feedback from a prior statistical review,
                          injected into the LLM prompt to influence reselection.
        exclude_model:    Optional model name to exclude from consideration
                          (e.g., the previously selected model that was rejected
                          by the statistical review).

    Returns:
        The :class:`ModelSelectionResult` with the selected model and reasoning.
    """

    # ── Logic to build suitability assessment in Python ──────────────────────
    def get_hw_suitability():
        points = []
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
                f"Trend detected (slope={stat_result.trend_slope:.4f}) — Holt-Winters handles trend via exponential smoothing."
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

    def get_arima_suitability():
        points = []
        sp = stat_result.seasonal_period
        if sp and sp > 1:
            points.append(
                f"Seasonal period {sp} detected — plain ARIMA ignores seasonality; SARIMA may be better."
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

    def get_sarima_suitability():
        points = []
        sp = stat_result.seasonal_period
        if sp and sp > 1:
            points.append(
                f"Seasonal period {sp} confirmed — SARIMA explicitly models seasonal AR/MA/I components."
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

    def get_ewma_suitability():
        points = []
        if stat_result.has_trend:
            points.append(
                f"Trend detected (slope={stat_result.trend_slope:.4f}) — EWMA will lag behind trend changes."
            )
        else:
            points.append("No significant trend — EWMA performs well on stable series.")
        if stat_result.outlier_ratio > 0.05:
            points.append(
                f"High outlier ratio ({stat_result.outlier_ratio:.1%}) — EWMA is sensitive to outliers."
            )
        else:
            points.append("Low outlier count — EWMA will be robust.")
        if stat_result.is_white_noise:
            points.append(
                "Series appears random — EWMA may be as good as complex models."
            )
        points.append(
            "EWMA is simple, fast, and works well for short-term forecasts with stable patterns."
        )
        points.append(
            "Best for real-time applications where simplicity and speed are priorities."
        )
        return "EWMA Assessment:\n" + "\n".join(f"- {p}" for p in points)

    suitability_summary = f"{get_hw_suitability()}\n\n{get_arima_suitability()}\n\n{get_sarima_suitability()}\n\n{get_ewma_suitability()}"

    def _get_heuristic_fallback() -> tuple[str, dict[str, str | None]]:
        """Determine the fallback model and reasoning based on statistical properties.

        Returns:
            tuple[str, dict[str, str | None]]: Fallback model and reasoning for each model.
        """
        sp = stat_result.seasonal_period or 1
        reasoning = {
            "Holt-Winters": None,
            "ARIMA": None,
            "SARIMA": None,
            "EWMA": None,
        }

        if sp > 1:
            # Strong seasonality detected
            fallback_model = "SARIMA"
            reasoning["Holt-Winters"] = (
                "Strong seasonality makes SARIMA/Holt-Winters preferable."
            )
            reasoning["ARIMA"] = (
                "Seasonal pattern detected; plain ARIMA ignores seasonality."
            )
            reasoning["EWMA"] = (
                "Seasonal patterns present; EWMA does not capture seasonality."
            )
        elif stat_result.has_trend and abs(stat_result.trend_slope) > 0.1:
            # Strong trend present
            fallback_model = "Holt-Winters"
            reasoning["ARIMA"] = (
                "Trend present but Holt-Winters handles it more naturally."
            )
            reasoning["SARIMA"] = "No strong seasonality confirmed; SARIMA may overfit."
            reasoning["EWMA"] = (
                "Strong trend present; EWMA will lag behind trend changes."
            )
        elif stat_result.is_white_noise:
            # Random series
            fallback_model = "EWMA"
            reasoning["Holt-Winters"] = (
                "Series appears random; simple EWMA may suffice."
            )
            reasoning["ARIMA"] = "Series is random noise; complex models may overfit."
            reasoning["SARIMA"] = "No patterns detected; SARIMA would overfit."
        else:
            # Default case
            fallback_model = "ARIMA"
            reasoning["Holt-Winters"] = (
                "No clear seasonal pattern or strong trend detected."
            )
            reasoning["SARIMA"] = "No seasonal period confirmed; SARIMA would overfit."
            reasoning["EWMA"] = "Series has patterns that ARIMA can better capture."

        return fallback_model, reasoning

    fallback_model, fallback_reasoning = _get_heuristic_fallback()

    # If a model is excluded (e.g., rejected by statistical review), adjust
    # the fallback so the heuristic does not pick it again.
    if exclude_model and fallback_model == exclude_model:
        # Pick the next best fallback from the remaining models.
        remaining = [m for m in _MODELS if m != exclude_model]
        if remaining:
            fallback_model = remaining[0]
            logger.info(
                "Fallback adjusted to exclude rejected model: %s",
                exclude_model,
            )

    # ── LLM Setup ────────────────────────────────────────────────────────────
    llm = get_llm(temperature=0)

    prompt = MODEL_SELECTION_PROMPT
    token_usage: dict[str, int] = {}

    try:
        chain = prompt | llm
        # Build the suitability summary with optional review feedback and
        # model exclusion context so the LLM can make a different choice.
        suitability_input = suitability_summary
        if review_feedback:
            suitability_input += (
                f"\n\n## Statistical Review Feedback (from prior run)\n"
                f"{review_feedback}\n\n"
                f"The previous model selection was reviewed and found to have "
                f"issues. Please select a DIFFERENT model that addresses the "
                f"reviewer's concerns."
            )
        if exclude_model:
            suitability_input += (
                f"\n\n## Model Exclusion\n"
                f"The model '{exclude_model}' was previously selected and "
                f"rejected by the statistical review. Do NOT select "
                f"'{exclude_model}' again."
            )
        inputs = {"suitability": suitability_input}
        response = chain.invoke(inputs)
        output = response.content
        token_usage = extract_token_usage(
            response, input_text=estimate_input_text(prompt, inputs)
        )
        logger.info("Model selection agent output: %s", output[:200])
        reasoning_steps = [
            {
                "thought": "Assessing suitability metrics for all models...",
                "observation": suitability_summary,
            },
            {
                "thought": "Finalizing model selection decision...",
                "observation": "Complete",
            },
        ]

        # Parse selected model from output.
        # Normalize: strip markdown bold/italic markers and unicode hyphens
        # so that "**Selected model:** Holt-Winters" matches correctly.
        normalized = output.replace("**", "").replace("__", "")
        normalized = normalized.replace("\u2010", "-").replace("\u2011", "-")
        normalized = normalized.replace("\u2012", "-").replace("\u2013", "-")
        normalized = normalized.replace("\u2014", "-").replace("\u2015", "-")

        selected_model = fallback_model
        exact_match_found = False
        normalized_lower = normalized.lower()
        for m in _MODELS:
            # Case-insensitive exact match: lower-case both sides so that
            # "selected model: arima" is reachable for all model names.
            if f"selected model: {m.lower()}" in normalized_lower:
                selected_model = m
                exact_match_found = True
                break
        # Broader fallback: look for "Selected model" line specifically
        # rather than scanning the first 100 chars blindly (which may contain
        # suitability text mentioning other model names).
        # Only run if the exact match did NOT find a model.
        if not exact_match_found:
            model_found = False
            for line in normalized.splitlines():
                if "selected model" in line.lower():
                    upper_line = line.upper()
                    # Check longest model names first to avoid substring
                    # matches (e.g. "ARIMA" inside "SARIMA").
                    for m in sorted(_MODELS, key=len, reverse=True):
                        if m.upper() in upper_line:
                            selected_model = m
                            model_found = True
                            break
                if model_found:
                    break

        explanation = output
        hw_rej = (
            None
            if selected_model == "Holt-Winters"
            else "Not selected based on LLM reasoning."
        )
        arima_rej = (
            None
            if selected_model == "ARIMA"
            else "Not selected based on LLM reasoning."
        )
        sarima_rej = (
            None
            if selected_model == "SARIMA"
            else "Not selected based on LLM reasoning."
        )
        ewma_rej = (
            None if selected_model == "EWMA" else "Not selected based on LLM reasoning."
        )

    except Exception as exc:
        logger.warning(
            "Model selection agent LLM call failed: %s — using heuristic.", exc
        )
        selected_model = fallback_model
        sp = stat_result.seasonal_period
        explanation = f"Heuristic selection: {fallback_model} chosen based on seasonal period={sp}."
        hw_rej = fallback_reasoning["Holt-Winters"]
        arima_rej = fallback_reasoning["ARIMA"]
        sarima_rej = fallback_reasoning["SARIMA"]
        ewma_rej = fallback_reasoning["EWMA"]
        reasoning_steps = [
            {
                "thought": f"Model selection agent failed: {str(exc)}",
                "observation": f"Falling back to heuristic selection: {selected_model}",
            }
        ]

    logger.info("Selected model: %s", selected_model)

    return ModelSelectionResult(
        selected_model=selected_model,
        explanation=explanation,
        holt_winters_rejected_reason=hw_rej,
        arima_rejected_reason=arima_rej,
        sarima_rejected_reason=sarima_rej,
        ewma_rejected_reason=ewma_rej,
        reasoning_steps=reasoning_steps,
        token_usage=token_usage,
    )
