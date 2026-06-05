from __future__ import annotations

from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from core.config import GEMINI_MODEL, USE_OLLAMA, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_API_KEY
from core.logging_config import get_logger
from forecasting.arima_model import fit_arima
from forecasting.holt_winters import fit_holt_winters
from forecasting.sarima_model import fit_sarima
from forecasting.ewma_model import fit_ewma
from schemas import ForecastResult, ModelSelectionResult, StatisticalResult

logger = get_logger(__name__)


def run_forecasting_agent(
    series: pd.Series,
    model_selection: ModelSelectionResult,
    stat_result: StatisticalResult,
    forecast_horizon: int,
    freq: str,
) -> tuple[ForecastResult, dict[str, dict[str, float]]]:
    """Run all three forecasting models, return ForecastResult for the selected model
    and an all-metrics dict for the comparison chart.

    Returns:
        (ForecastResult, all_metrics_dict)
        all_metrics_dict: {"ARIMA": {"RMSE": x, "MAE": y, "MAPE": z}, ...}
    """
    seasonal_period = stat_result.seasonal_period or 12
    results_store: dict[str, dict[str, Any]] = {}
    
    # ── Fit all models directly in Python ─────────────────────────────────────
    for name, fn, kwargs in [
        ("Holt-Winters", fit_holt_winters, {}),
        ("ARIMA", fit_arima, {}),
        ("SARIMA", fit_sarima, {"seasonal_period": seasonal_period}),
        ("EWMA", fit_ewma, {}),
    ]:
        try:
            results_store[name] = fn(series, forecast_horizon, **kwargs)
        except Exception as exc:
            logger.warning("%s fitting failed: %s", name, exc)

    comparison_summary = "Model comparison metrics (lower is better):\n"
    for name, res in results_store.items():
        comparison_summary += (
            f"- {name}: RMSE={res['rmse']:.4f}, MAE={res['mae']:.4f}, MAPE={res['mape']:.2f}%\n"
        )

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
        ("system", "You are a forecasting expert. Review the performance of multiple models."),
        ("human", (
            "The pre-selected model is: {selected}.\n\n"
            "Review these fitting results:\n"
            "{summary}\n\n"
            "Explain why the selected model is optimal or note if another model achieved better MAPE."
        ))
    ])

    try:
        chain = prompt | llm
        response = chain.invoke({
            "selected": model_selection.selected_model,
            "summary": comparison_summary
        })
        reasoning_steps = [
            {"thought": "Fitting Holt-Winters, ARIMA, and SARIMA in Python...", "observation": comparison_summary},
            {"thought": "Analyzing metrics for performance comparison...", "observation": response.content}
        ]
    except Exception as exc:
        logger.warning("Forecasting agent LLM call failed: %s", exc)
        reasoning_steps = [{
            "thought": "LLM analysis failed, relying on direct Python metrics.",
            "observation": comparison_summary
        }]

    # ── Select result for the chosen model ───────────────────────────────────
    selected = model_selection.selected_model
    if selected not in results_store:
        # Try to fit the selected model directly
        try:
            if selected == "Holt-Winters":
                results_store[selected] = fit_holt_winters(series, forecast_horizon)
            elif selected == "ARIMA":
                results_store[selected] = fit_arima(series, forecast_horizon)
            else:
                results_store[selected] = fit_sarima(series, forecast_horizon, seasonal_period)
        except Exception as exc:
            logger.error("Could not fit selected model %s: %s", selected, exc)
            # Fall back to any available result
            if results_store:
                selected = next(iter(results_store))
                logger.warning("Falling back to %s", selected)
            else:
                raise RuntimeError("All forecasting models failed.") from exc

    res = results_store[selected]

    # ── Generate forecast dates ───────────────────────────────────────────────
    last_date = series.index[-1] if hasattr(series.index, "max") else None
    forecast_dates: list[str] = []
    if last_date is not None:
        try:
            date_range = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq=freq)[1:]
            forecast_dates = date_range.strftime("%Y-%m-%d").tolist()
        except Exception:
            forecast_dates = [str(i + 1) for i in range(forecast_horizon)]
    else:
        forecast_dates = [str(i + 1) for i in range(forecast_horizon)]

    # ── Build all_metrics dict for comparison chart ───────────────────────────
    all_metrics = {
        name: {"RMSE": r["rmse"], "MAE": r["mae"], "MAPE": r["mape"]}
        for name, r in results_store.items()
    }

    logger.info("Forecasting complete. Selected: %s", selected)

    forecast_result = ForecastResult(
        model_used=selected,
        forecast=res["forecast"],
        lower_ci=res["lower_ci"],
        upper_ci=res["upper_ci"],
        forecast_dates=forecast_dates,
        rmse=res["rmse"],
        mae=res["mae"],
        mape=res["mape"],
        reasoning_steps=reasoning_steps,
    )
    return forecast_result, all_metrics
