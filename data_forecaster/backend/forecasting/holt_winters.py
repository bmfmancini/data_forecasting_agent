from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from core.logging_config import get_logger

logger = get_logger(__name__)


def fit_holt_winters(series: pd.Series, forecast_horizon: int) -> dict:
    """Fit Holt-Winters Triple Exponential Smoothing and return forecast + metrics.

    Returns:
        dict with keys: forecast, lower_ci, upper_ci, rmse, mae, mape
    """
    series = series.dropna().astype(float)
    seasonal_period = _infer_seasonal_period(series)
    use_seasonal = len(series) >= 2 * seasonal_period

    seasonal = None
    trend = "add"

    if use_seasonal:
        if (series > 0).all():
            try:
                m_fit = ExponentialSmoothing(
                    series, trend="add", seasonal="mul", seasonal_periods=seasonal_period
                ).fit(optimized=True)
                a_fit = ExponentialSmoothing(
                    series, trend="add", seasonal="add", seasonal_periods=seasonal_period
                ).fit(optimized=True)
                seasonal = "mul" if m_fit.aic < a_fit.aic else "add"
            except Exception:
                seasonal = "add"
        else:
            seasonal = "add"

    logger.info(
        "Holt-Winters config: seasonal=%s seasonal_period=%d series_len=%d",
        seasonal, seasonal_period, len(series),
    )

    # ── Metrics via train/test split ─────────────────────────────────────────
    split = max(int(len(series) * 0.8), len(series) - forecast_horizon)
    train, test = series.iloc[:split], series.iloc[split:]

    try:
        train_fit = ExponentialSmoothing(
            train, trend=trend, seasonal=seasonal,
            seasonal_periods=seasonal_period if use_seasonal else None,
        ).fit(optimized=True)
        test_fc = train_fit.forecast(len(test))
        rmse = float(np.sqrt(np.mean((test.values - test_fc.values) ** 2)))
        mae = float(np.mean(np.abs(test.values - test_fc.values)))
        mape = float(np.mean(np.abs((test.values - test_fc.values) / (test.values + 1e-8))) * 100)
    except Exception as exc:
        logger.warning("Holt-Winters metrics failed: %s", exc)
        rmse = mae = mape = 0.0

    # ── Full-series fit for forecast ─────────────────────────────────────────
    full_fit = ExponentialSmoothing(
        series, trend=trend, seasonal=seasonal,
        seasonal_periods=seasonal_period if use_seasonal else None,
    ).fit(optimized=True)

    forecast_values = full_fit.forecast(forecast_horizon)
    resid_std = float(np.std(full_fit.resid))
    h = np.arange(1, forecast_horizon + 1)
    lower_ci = (forecast_values.values - 1.96 * resid_std * np.sqrt(h)).tolist()
    upper_ci = (forecast_values.values + 1.96 * resid_std * np.sqrt(h)).tolist()

    return {
        "forecast": forecast_values.tolist(),
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }


def _infer_seasonal_period(series: pd.Series) -> int:
    if hasattr(series.index, "freq") and series.index.freq is not None:
        freq_str = str(series.index.freq).upper()
        if "MS" in freq_str or freq_str.startswith("M"):
            return 12
        if "QS" in freq_str or freq_str.startswith("Q"):
            return 4
        if "W" in freq_str:
            return 52
        if freq_str.startswith("D"):
            return 7
    return 12
