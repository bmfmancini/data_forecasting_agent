from __future__ import annotations

import numpy as np
import pandas as pd

from core.logging_config import get_logger

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

    # ── Metrics via train/test split ─────────────────────────────────────────
    split = max(int(len(series) * 0.8), len(series) - forecast_horizon)
    train, test = series.iloc[:split], series.iloc[split:]

    # Calculate EWMA for training set
    train_ewma = train.ewm(alpha=alpha).mean()

    # Forecast for test set (use last training value for all forecasts)
    last_train_value = train_ewma.iloc[-1]
    test_forecast = pd.Series([last_train_value] * len(test), index=test.index)

    try:
        rmse = float(np.sqrt(np.mean((test.values - test_forecast.values) ** 2)))
        mae = float(np.mean(np.abs(test.values - test_forecast.values)))
        mape = float(
            np.mean(np.abs((test.values - test_forecast.values) / (test.values + 1e-8)))
            * 100
        )
    except Exception as exc:
        logger.warning("EWMA metrics failed: %s", exc)
        rmse = mae = mape = 0.0

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
        "y_train": train,
        "y_test": test,
        "forecast_test": test_fc,
    }
