"""Forecasting agent that selects and runs statistical model implementations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.llm_factory import get_llm
from core.logging_config import get_logger
from forecasting.arima_model import fit_arima
from forecasting.contracts import ForecastAdapterResult, ForecastFitStatus
from forecasting.ewma_model import fit_ewma
from forecasting.holt_winters import fit_holt_winters
from forecasting.sarima_model import fit_sarima
from prompts.forecasting_prompt import FORECASTING_PROMPT
from schemas import ForecastResult, ModelSelectionResult, StatisticalResult
from utils.statistical_analysis import analyze_residuals
from utils.token_tracking import estimate_input_text, extract_token_usage

logger = get_logger(__name__)


def _has_required_metrics(result: ForecastAdapterResult) -> bool:
    """Return whether required comparison metrics are present and finite.

    A model is rankable only when ``status == ok`` and the core point-error
    metrics (RMSE, MAE, MAPE) are all present and finite. Finiteness alone
    is insufficient — a degraded or failed model is never rankable.
    """
    if result.status != ForecastFitStatus.OK:
        return False
    for metric in (result.metrics.rmse, result.metrics.mae, result.metrics.mape):
        if metric is None or not np.isfinite(metric):
            return False
    return True


def _format_metric(value: float | None, fmt: str) -> str:
    """Format a nullable metric, returning 'not available' when ``None``."""
    if value is None or not np.isfinite(value):
        return "not available"
    return format(value, fmt)


def run_forecasting_agent(
    series: pd.Series,
    model_selection: ModelSelectionResult,
    stat_result: StatisticalResult,
    forecast_horizon: int,
    freq: str,
    existing_metrics: dict[str, dict[str, float]] | None = None,
    disabled_tests: list[str] | None = None,
) -> tuple[ForecastResult, dict[str, dict[str, float]]]:
    """Run all forecasting models, return ForecastResult for the selected model
    and an all-metrics dict for the comparison chart.

    Args:
        series: Historical time series data.
        model_selection: Output of the model selection agent.
        stat_result: Output of the statistical analysis agent.
        forecast_horizon: Number of periods to forecast.
        freq: Frequency string for generating forecast dates.
        existing_metrics: Optional pre-existing metrics dict (e.g. from a prior
            run or baseline models) to merge into the returned dict so that
            re-runs preserve previously computed metrics.
        disabled_tests: Optional list of residual diagnostic tests to skip.

    Returns:
        (ForecastResult, all_metrics_dict) where all_metrics_dict maps model
        names to ``{"RMSE": x, "MAE": y, "MAPE": z, "WAPE": w, "MASE": m}``.

    Raises:
        RuntimeError: If no forecasting model produces valid evaluation metrics.
    """
    seasonal_period = stat_result.seasonal_period or 12
    results_store: dict[str, ForecastAdapterResult] = {}

    # ── Fit all models directly in Python ─────────────────────────────────────
    for name, fn, kwargs in [
        ("Holt-Winters", fit_holt_winters, {}),
        ("ARIMA", fit_arima, {}),
        ("SARIMA", fit_sarima, {"seasonal_period": seasonal_period}),
        ("EWMA", fit_ewma, {}),
    ]:
        try:
            results_store[name] = fn(series, forecast_horizon, **kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("%s fitting failed: %s", name, exc)

    comparison_summary = "Model comparison metrics (lower is better):\n"
    for name, res in results_store.items():
        if not _has_required_metrics(res):
            comparison_summary += f"- {name}: required metrics unavailable\n"
            continue
        wape_text = (
            f", WAPE={_format_metric(res.metrics.wape, '.2%')}"
            if res.metrics.wape is not None
            else ""
        )
        mase_text = (
            f", MASE={_format_metric(res.metrics.mase, '.4f')}"
            if res.metrics.mase is not None
            else ""
        )
        comparison_summary += (
            f"- {name}: RMSE={res.metrics.rmse:.4f}, MAE={res.metrics.mae:.4f}, "
            f"MAPE={res.metrics.mape:.2f}%{wape_text}{mase_text}\n"
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
    except Exception as exc:  # pylint: disable=broad-except
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
            elif selected == "EWMA":
                results_store[selected] = fit_ewma(series, forecast_horizon)
            else:
                results_store[selected] = fit_sarima(
                    series, forecast_horizon, seasonal_period
                )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Could not fit selected model %s: %s", selected, exc)
            # Fall back to any available result
            if results_store:
                selected = next(iter(results_store))
                logger.warning("Falling back to %s", selected)
            else:
                raise RuntimeError("All forecasting models failed.") from exc

    res = results_store[selected]
    if not _has_required_metrics(res):
        rankable = {
            name: candidate
            for name, candidate in results_store.items()
            if _has_required_metrics(candidate)
        }
        if not rankable:
            raise RuntimeError(
                "No forecasting model produced valid evaluation metrics."
            )
        # Deterministic policy: lowest RMSE wins. The LLM never decides
        # model rankings.
        selected = min(
            rankable, key=lambda name: rankable[name].metrics.rmse or float("inf")
        )
        res = rankable[selected]
        res = res.model_copy(update={"is_fallback": True})
        logger.warning(
            "Selected model lacked valid evaluation evidence; falling back to %s",
            selected,
        )

    # ── Generate forecast dates ───────────────────────────────────────────────
    last_date = series.index[-1] if hasattr(series.index, "max") else None
    forecast_dates: list[str] = []
    if last_date is not None:
        try:
            date_range = pd.date_range(
                start=last_date, periods=forecast_horizon + 1, freq=freq
            )[1:]
            forecast_dates = date_range.strftime("%Y-%m-%d").tolist()
        except Exception:  # pylint: disable=broad-except
            forecast_dates = [str(i + 1) for i in range(forecast_horizon)]
    else:
        forecast_dates = [str(i + 1) for i in range(forecast_horizon)]

    # ── Build all_metrics dict for comparison chart ───────────────────────────
    all_metrics: dict[str, dict[str, float]] = {}
    for name, r in results_store.items():
        if not _has_required_metrics(r):
            continue
        all_metrics[name] = {
            "RMSE": r.metrics.rmse or float("nan"),
            "MAE": r.metrics.mae or float("nan"),
            "MAPE": r.metrics.mape or float("nan"),
            "WAPE": r.metrics.wape if r.metrics.wape is not None else float("nan"),
            "MASE": r.metrics.mase if r.metrics.mase is not None else float("nan"),
        }
    # Merge any pre-existing metrics (e.g. baselines) passed in by the caller
    # so re-runs preserve previously computed results.
    if existing_metrics is not None:
        for name, metrics in existing_metrics.items():
            all_metrics.setdefault(name, metrics)

    # ── Residual Analysis ─────────────────────────────────────────────────────
    residual_diagnostics = None
    # Residuals are not currently returned by the typed adapters; this
    # branch will be activated when adapters expose innovations.
    del disabled_tests  # Unused until adapters return residuals.

    logger.info("Forecasting complete. Selected: %s", selected)

    forecast_result = ForecastResult(
        model_used=selected,
        status=res.status,
        failure_reason=res.failure_reason,
        is_fallback=res.is_fallback,
        forecast=res.forecast,
        lower_ci=res.lower_ci,
        upper_ci=res.upper_ci,
        forecast_dates=forecast_dates,
        rmse=res.metrics.rmse,
        mae=res.metrics.mae,
        mape=res.metrics.mape,
        wape=res.metrics.wape,
        mase=res.metrics.mase,
        residual_diagnostics=residual_diagnostics,
        reasoning_steps=reasoning_steps,
        token_usage=token_usage,
    )
    return forecast_result, all_metrics
