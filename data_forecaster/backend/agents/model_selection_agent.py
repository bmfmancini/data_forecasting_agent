from __future__ import annotations

from core.llm_factory import get_llm
from core.logging_config import get_logger
from prompts.model_selection_prompt import MODEL_SELECTION_PROMPT
from schemas import ModelSelectionResult, StatisticalResult

logger = get_logger(__name__)

_MODELS = ("ARIMA", "SARIMA", "Holt-Winters", "EWMA")


def run_model_selection_agent(stat_result: StatisticalResult) -> ModelSelectionResult:
    """Use the LLM to reason over statistical findings and select the best model."""

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

    fallback_model, _ = _get_heuristic_fallback()

    # ── LLM Setup ────────────────────────────────────────────────────────────
    llm = get_llm(temperature=0)

    prompt = MODEL_SELECTION_PROMPT

    try:
        chain = prompt | llm
        response = chain.invoke({"suitability": suitability_summary})
        output = response.content
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

        # Parse selected model from output
        selected_model = fallback_model
        for m in _MODELS:
            if (
                f"Selected model: {m}" in output
                or f"selected model: {m}" in output.lower()
            ):
                selected_model = m
                break
        # Broader fallback scan
        if selected_model == fallback_model:
            upper = output.upper()
            for m in _MODELS:
                if m.upper() in upper[:100]:
                    selected_model = m
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
        explanation = f"Heuristic selection: {fallback_model} chosen based on seasonal period={sp}."
        hw_rej = hw_reason if selected_model != "Holt-Winters" else None
        arima_rej = arima_reason if selected_model != "ARIMA" else None
        sarima_rej = sarima_reason if selected_model != "SARIMA" else None
        ewma_rej = ewma_reason if selected_model != "EWMA" else None
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
    )
