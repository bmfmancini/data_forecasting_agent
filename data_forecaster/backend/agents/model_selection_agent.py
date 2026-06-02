from __future__ import annotations

from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from core.config import GEMINI_MODEL, USE_OLLAMA, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_API_KEY
from core.logging_config import get_logger
from schemas import ModelSelectionResult, StatisticalResult

logger = get_logger(__name__)

_MODELS = ("ARIMA", "SARIMA", "Holt-Winters")


def run_model_selection_agent(stat_result: StatisticalResult) -> ModelSelectionResult:
    """Use the LLM to reason over statistical findings and select the best model."""
    
    # ── Logic to build suitability assessment in Python ──────────────────────
    def get_hw_suitability():
        points = []
        sp = stat_result.seasonal_period
        if sp and sp > 1:
            points.append(f"Seasonal period {sp} detected — Holt-Winters models seasonality natively.")
        else:
            points.append("No clear seasonal period — Holt-Winters seasonal component may not help.")
        if stat_result.has_trend:
            points.append(f"Trend detected (slope={stat_result.trend_slope:.4f}) — Holt-Winters handles trend via exponential smoothing.")
        else:
            points.append("No significant trend — simple exponential smoothing may suffice.")
        if not stat_result.is_stationary_adf:
            points.append("Non-stationary series — Holt-Winters does not require pre-differencing.")
        points.append("Holt-Winters is fast, interpretable, and robust on short-to-medium series.")
        return "Holt-Winters Assessment:\n" + "\n".join(f"- {p}" for p in points)

    def get_arima_suitability():
        points = []
        sp = stat_result.seasonal_period
        if sp and sp > 1:
            points.append(f"Seasonal period {sp} detected — plain ARIMA ignores seasonality; SARIMA may be better.")
        else:
            points.append("No strong seasonality — ARIMA is appropriate.")
        if not stat_result.is_stationary_adf:
            points.append("Non-stationary series — ARIMA handles this via differencing (d parameter).")
        else:
            points.append("Series is stationary — ARIMA(p,0,q) sufficient.")
        if stat_result.has_trend:
            points.append("Trend present — ARIMA differencing (d≥1) will remove it.")
        points.append("ARIMA is well-suited for non-seasonal series with complex autocorrelation.")
        return "ARIMA Assessment:\n" + "\n".join(f"- {p}" for p in points)

    def get_sarima_suitability():
        points = []
        sp = stat_result.seasonal_period
        if sp and sp > 1:
            points.append(f"Seasonal period {sp} confirmed — SARIMA explicitly models seasonal AR/MA/I components.")
            points.append("SARIMA is the gold standard for stationary-transformable seasonal series.")
        else:
            points.append("No seasonal period detected — SARIMA seasonal component would overfit.")
        if not stat_result.is_stationary_adf:
            points.append("Non-stationary — SARIMA seasonal differencing (D≥1) will address this.")
        points.append("SARIMA requires more data than ARIMA (at least 2 full seasonal cycles).")
        return "SARIMA Assessment:\n" + "\n".join(f"- {p}" for p in points)

    suitability_summary = f"{get_hw_suitability()}\n\n{get_arima_suitability()}\n\n{get_sarima_suitability()}"

    # ── Heuristic fallback selection (used if LLM fails) ─────────────────────
    sp = stat_result.seasonal_period or 1
    if sp > 1:
        fallback_model = "SARIMA"
        hw_reason = "Strong seasonality makes SARIMA/Holt-Winters preferable."
        arima_reason = "Seasonal pattern detected; plain ARIMA ignores seasonality."
        sarima_reason = None
    else:
        fallback_model = "ARIMA"
        hw_reason = "No clear seasonal pattern detected."
        arima_reason = None
        sarima_reason = "No seasonal period confirmed; SARIMA would overfit."

    # ── LLM Setup ────────────────────────────────────────────────────────────
    if USE_OLLAMA:
        llm = ChatOllama(
            model=OLLAMA_MODEL, 
            base_url=OLLAMA_BASE_URL, 
            temperature=0,
            headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else None,
        )
    else:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in time series model selection. Choose the best model based on statistical assessments."),
        ("human", (
            "Evaluate the suitability of ARIMA, SARIMA, and Holt-Winters based on these assessments:\n\n"
            "{suitability}\n\n"
            "Select the SINGLE best model and provide a detailed rationale.\n"
            "Your output MUST follow this exact structure:\n"
            "Selected model: <MODEL_NAME>\n\n"
            "## Why <MODEL_NAME> was chosen\n"
            "<Detailed explanation referencing metrics>\n\n"
            "## Model Assessment Summary\n"
            "- **ARIMA**: <suitability detail> — Suitability: High/Medium/Low\n"
            "- **SARIMA**: <suitability detail> — Suitability: High/Medium/Low\n"
            "- **Holt-Winters**: <suitability detail> — Suitability: High/Medium/Low\n\n"
            "## Why other models were not chosen\n"
            "- **<Rejected Model 1>**: <reason>\n"
            "- **<Rejected Model 2>**: <reason>"
        ))
    ])

    try:
        chain = prompt | llm
        response = chain.invoke({"suitability": suitability_summary})
        output = response.content
        logger.info("Model selection agent output: %s", output[:200])
        reasoning_steps = [
            {"thought": "Assessing suitability metrics for all models...", "observation": suitability_summary},
            {"thought": "Finalizing model selection decision...", "observation": "Complete"}
        ]

        # Parse selected model from output
        selected_model = fallback_model
        for m in _MODELS:
            if f"Selected model: {m}" in output or f"selected model: {m}" in output.lower():
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
        hw_rej = None if selected_model == "Holt-Winters" else "Not selected based on LLM reasoning."
        arima_rej = None if selected_model == "ARIMA" else "Not selected based on LLM reasoning."
        sarima_rej = None if selected_model == "SARIMA" else "Not selected based on LLM reasoning."

    except Exception as exc:
        logger.warning("Model selection agent LLM call failed: %s — using heuristic.", exc)
        selected_model = fallback_model
        explanation = f"Heuristic selection: {fallback_model} chosen based on seasonal period={sp}."
        hw_rej = hw_reason if selected_model != "Holt-Winters" else None
        arima_rej = arima_reason if selected_model != "ARIMA" else None
        sarima_rej = sarima_reason if selected_model != "SARIMA" else None
        reasoning_steps = [{
            "thought": f"Model selection agent failed: {str(exc)}",
            "observation": f"Falling back to heuristic selection: {selected_model}"
        }]

    logger.info("Selected model: %s", selected_model)

    return ModelSelectionResult(
        selected_model=selected_model,
        explanation=explanation,
        holt_winters_rejected_reason=hw_rej,
        arima_rejected_reason=arima_rej,
        sarima_rejected_reason=sarima_rej,
        reasoning_steps=reasoning_steps,
    )
