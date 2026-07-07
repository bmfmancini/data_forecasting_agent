from __future__ import annotations

from typing import Any

import pandas as pd

from core.llm_factory import get_llm
from core.logging_config import get_logger
from forecasting.arima_model import fit_arima
from forecasting.ewma_model import fit_ewma
from forecasting.holt_winters import fit_holt_winters
from forecasting.sarima_model import fit_sarima
from prompts.forecasting_prompt import FORECASTING_PROMPT
from schemas import ForecastResult, ModelSelectionResult, StatisticalResult
from utils.token_tracking import estimate_input_text, extract_token_usage

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
        comparison_summary += f"- {name}: RMSE={res['rmse']:.4f}, MAE={res['mae']:.4f}, MAPE={res['mape']:.2f}%\n"

    # ── LLM Setup ────────────────────────────────────────────────────────────
    llm = get_llm(temperature=0)

    prompt = FORECASTING_PROMPT
    token_usage: dict[str, int] = {}

    try:
        chain = prompt | llm
        inputs = {
            "selected": model_selection.selected_model,
            "summary": comparison_summary,
        }
        response = chain.invoke(inputs)
        token_usage = extract_token_usage(
            response, input_text=estimate_input_text(prompt, inputs)
        )
        reasoning_steps = [
            {
                "thought": "Fitting Holt-Winters, ARIMA, and SARIMA in Python...",
                "observation": comparison_summary,
            },
            {
                "thought": "Analyzing metrics for performance comparison...",
                "observation": response.content,
            },
        ]
    except Exception as exc:
        logger.warning("Forecasting agent LLM call failed: %s", exc)
        reasoning_steps = [
            {
                "thought": "LLM analysis failed, relying on direct Python metrics.",
                "observation": comparison_summary,
            }
        ]

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
                results_store[selected] = fit_sarima(
                    series, forecast_horizon, seasonal_period
                )
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
            date_range = pd.date_range(
                start=last_date, periods=forecast_horizon + 1, freq=freq
            )[1:]
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
        token_usage=token_usage,
    )
    return forecast_result, all_metrics
