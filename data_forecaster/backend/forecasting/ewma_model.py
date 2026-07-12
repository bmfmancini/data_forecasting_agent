"""Exponentially weighted moving average forecasting implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.logging_config import get_logger
from utils.validation import perform_rolling_origin_validation

logger = get_logger(__name__)


def fit_ewma(series: pd.Series, forecast_horizon: int, alpha: float = 0.3) -> dict:
    """Fit Exponential Weighted Moving Average model and return forecast + metrics.

    Args:
        series: Time series data
        forecast_horizon: Number of periods to forecast
        alpha: Smoothing parameter (0 < alpha < 1)

    Returns:
        dict with keys: forecast, lower_ci, upper_ci, rmse, mae, mape
    """
    series = series.dropna().astype(float)

    # ── Metrics via rolling-origin validation ────────────────────────────────
    def _ewma_fit_forecast(train_series: pd.Series, horizon: int) -> pd.Series:
        """Fit EWMA and produce a forecast for one validation split."""
        train_ewma = train_series.ewm(alpha=alpha).mean()
        last_train_value = train_ewma.iloc[-1]
        return pd.Series([last_train_value] * horizon)

    metrics = perform_rolling_origin_validation(
        series, forecast_horizon, _ewma_fit_forecast
    )
    rmse = metrics.get("rmse")
    mae = metrics.get("mae")
    mape = metrics.get("mape")
    if not metrics:
        logger.warning("EWMA rolling validation failed; metrics unavailable.")

    # ── Full-series fit for forecast ─────────────────────────────────────────
    # Calculate EWMA for entire series
    full_ewma = series.ewm(alpha=alpha).mean()
    last_full_value = full_ewma.iloc[-1]

    # Forecast: use the last EWMA value for all future periods
    forecast_values = [last_full_value] * forecast_horizon

    # Calculate confidence intervals using rolling standard deviation
    residuals = series - full_ewma
    std_residuals = np.std(residuals.dropna())

    # 95% confidence intervals (approximate)
    lower_ci = [f - 1.96 * std_residuals for f in forecast_values]
    upper_ci = [f + 1.96 * std_residuals for f in forecast_values]

    logger.info("EWMA model fitted with alpha=%.2f", alpha)

    return {
        "forecast": forecast_values,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }
