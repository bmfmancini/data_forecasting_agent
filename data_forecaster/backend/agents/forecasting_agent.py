"""Forecasting agent that selects and runs statistical model implementations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.llm_factory import get_llm
from core.logging_config import get_logger
from forecasting.arima_model import fit_arima
from forecasting.ewma_model import fit_ewma
from forecasting.holt_winters import fit_holt_winters
from forecasting.sarima_model import fit_sarima
from prompts.forecasting_prompt import FORECASTING_PROMPT
from schemas import ForecastResult, ModelSelectionResult, StatisticalResult
from utils.statistical_analysis import analyze_residuals
from utils.token_tracking import estimate_input_text, extract_token_usage

logger = get_logger(__name__)


def _has_required_metrics(result: dict[str, Any]) -> bool:
    """Return whether required comparison metrics are present and finite."""
    for metric in ("rmse", "mae", "mape"):
        value = result.get(metric)
        if value is None or not np.isfinite(value):
            return False
    return True


def _calculate_additional_metrics(
    y_true: pd.Series, y_pred: pd.Series, y_train: pd.Series, seasonal_period: int
) -> dict[str, float]:
    """Calculate WAPE and MASE."""
    metrics = {}
    # WAPE: Sum of absolute errors / sum of absolute actuals
    # Stable alternative to MAPE, especially with zeros in y_true.
    absolute_errors = np.abs(y_true - y_pred)
    sum_of_actuals = np.sum(np.abs(y_true))
    if sum_of_actuals != 0:
        metrics["wape"] = np.sum(absolute_errors) / sum_of_actuals
    else:
        metrics["wape"] = np.nan # Avoid division by zero

    # MASE: Mean Absolute Error / MAE of a naive seasonal forecast on training data
    # The gold standard for comparing forecast accuracy across different series.
    mae = np.mean(absolute_errors)
    if y_train.shape[0] > seasonal_period:
        y_train_naive = y_train.shift(seasonal_period).dropna()
        train_residuals_naive = y_train[y_train_naive.index] - y_train_naive
        mae_naive = np.mean(np.abs(train_residuals_naive))
        if mae_naive != 0:
            metrics["mase"] = mae / mae_naive
        else:
            metrics["mase"] = np.inf # Should be rare
    else:
        # Fallback for very short series where seasonal naive is not possible
        mae_naive = np.mean(np.abs(np.diff(y_train)))
        if mae_naive != 0:
            metrics["mase"] = mae / mae_naive
        else:
            metrics["mase"] = np.inf
    return metrics

def run_forecasting_agent(
    series: pd.Series,
    model_selection: ModelSelectionResult,
    stat_result: StatisticalResult,
    forecast_horizon: int,
    freq: str,
    existing_metrics: dict[str, dict[str, float]] | None = None,
    disabled_tests: list[str] | None = None,
) -> tuple[ForecastResult, dict[str, dict[str, float]]]:
    """Run all three forecasting models, return ForecastResult for the selected model
    and an all-metrics dict for the comparison chart.

    Args:
        existing_metrics: Optional pre-existing metrics dict (e.g. from a prior
            run or baseline models) to merge into the returned dict so that
            re-runs preserve previously computed metrics.

    Returns:
        (ForecastResult, all_metrics_dict)
        all_metrics_dict: {"ARIMA": {"RMSE": x, "MAE": y, "MAPE": z, "WAPE": w, "MASE": m}, ...}
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
            # Post-hoc calculation of WAPE and MASE
            # Assumes fit functions return test set actuals and training data
            if "y_test" in results_store[name] and "forecast" in results_store[name]:
                y_test = results_store[name]["y_test"]
                forecast = results_store[name]["forecast"]
                y_train = results_store[name].get(
                    "y_train",
                    series[: len(series) - len(y_test)],
                )
                
                additional_metrics = _calculate_additional_metrics(
                    y_test, forecast, y_train, seasonal_period
                )
                results_store[name].update(additional_metrics)

        except Exception as exc:
            logger.warning("%s fitting failed: %s", name, exc)

    comparison_summary = "Model comparison metrics (lower is better):\n"
    for name, res in results_store.items():
        if not _has_required_metrics(res):
            comparison_summary += f"- {name}: required metrics unavailable\n"
            continue
        wape_text = (
            f", WAPE={res.get('wape', np.nan) * 100:.2f}%"
            if "wape" in res
            else ""
        )
        mase_text = f", MASE={res.get('mase', np.nan):.4f}" if 'mase' in res else ""
        comparison_summary += (
            f"- {name}: RMSE={res['rmse']:.4f}, MAE={res['mae']:.4f}, "
            f"MAPE={res['mape']:.2f}%{wape_text}{mase_text}\n"
        )

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
        name: {
            "RMSE": r["rmse"],
            "MAE": r["mae"],
            "MAPE": r["mape"],
            "WAPE": r.get("wape", np.nan),
            "MASE": r.get("mase", np.nan),
        }
        for name, r in results_store.items() if _has_required_metrics(r)
    }
    # Merge any pre-existing metrics (e.g. baselines) passed in by the caller
    # so re-runs preserve previously computed results.
    if existing_metrics is not None:
        for name, metrics in existing_metrics.items():
            all_metrics.setdefault(name, metrics)

    # ── Residual Analysis ─────────────────────────────────────────────────────
    residual_diagnostics = None
    if "residuals" in res and isinstance(res["residuals"], pd.Series):
        try:
            residual_diagnostics = analyze_residuals(
                res["residuals"], disabled_tests=disabled_tests
            )
        except Exception as exc:
            logger.warning("Residual analysis failed: %s", exc)

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
        wape=res.get("wape"),
        mase=res.get("mase"),
        residual_diagnostics=residual_diagnostics,
        reasoning_steps=reasoning_steps,
        token_usage=token_usage,
    )
    return forecast_result, all_metrics
