from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import periodogram as scipy_periodogram
from scipy.stats import linregress
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from core.logging_config import get_logger

logger = get_logger(__name__)


def run_adf_test(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller test for stationarity.

    Returns:
        statistic, p_value, is_stationary, interpretation
    """
    values = series.dropna().astype(float).values
    result = adfuller(values, autolag="AIC")
    stat, p_value = float(result[0]), float(result[1])
    is_stationary = p_value < 0.05

    if is_stationary:
        interpretation = (
            f"ADF statistic={stat:.4f}, p={p_value:.4f}: Reject unit-root null. "
            "Series is stationary."
        )
    else:
        interpretation = (
            f"ADF statistic={stat:.4f}, p={p_value:.4f}: Cannot reject unit-root null. "
            "Series is non-stationary; differencing likely required."
        )

    logger.debug("ADF: stat=%.4f p=%.4f stationary=%s", stat, p_value, is_stationary)
    return {
        "statistic": stat,
        "p_value": p_value,
        "is_stationary": is_stationary,
        "interpretation": interpretation,
    }


def run_kpss_test(series: pd.Series) -> dict:
    """KPSS test for stationarity (null = stationary).

    Returns:
        statistic, p_value, is_stationary, interpretation
    """
    values = series.dropna().astype(float).values
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = kpss(values, regression="c", nlags="auto")
    stat, p_value = float(result[0]), float(result[1])
    is_stationary = p_value > 0.05

    if is_stationary:
        interpretation = (
            f"KPSS statistic={stat:.4f}, p={p_value:.4f}: Cannot reject stationarity null. "
            "Series appears stationary."
        )
    else:
        interpretation = (
            f"KPSS statistic={stat:.4f}, p={p_value:.4f}: Reject stationarity null. "
            "Series is non-stationary."
        )

    logger.debug("KPSS: stat=%.4f p=%.4f stationary=%s", stat, p_value, is_stationary)
    return {
        "statistic": stat,
        "p_value": p_value,
        "is_stationary": is_stationary,
        "interpretation": interpretation,
    }


def run_stl_decomposition(series: pd.Series, period: int = 12) -> dict:
    """STL decomposition into trend, seasonal, and residual components.

    Returns:
        trend, seasonal, residual as float lists
    """
    values = series.dropna().astype(float)
    period = max(period, 2)
    # STL needs at least 2 full cycles
    if len(values) < 2 * period:
        logger.warning("Series too short for STL with period=%d; using period=2", period)
        period = 2

    stl = STL(values, period=period, robust=True)
    res = stl.fit()
    logger.debug("STL decomposition complete. period=%d", period)
    return {
        "trend": res.trend.tolist(),
        "seasonal": res.seasonal.tolist(),
        "residual": res.resid.tolist(),
    }


def compute_acf_pacf(series: pd.Series, lags: int = 40) -> dict:
    """Compute ACF and PACF values.

    Returns:
        acf_values, pacf_values, lags as lists
    """
    values = series.dropna().astype(float).values
    max_lags = min(lags, len(values) // 2 - 1)

    acf_vals = acf(values, nlags=max_lags, fft=True)
    pacf_vals = pacf(values, nlags=max_lags, method="ywm")
    lag_list = list(range(len(acf_vals)))

    return {
        "acf_values": acf_vals.tolist(),
        "pacf_values": pacf_vals.tolist(),
        "lags": lag_list,
    }


def run_periodogram(series: pd.Series) -> dict:
    """Compute periodogram and identify dominant period.

    Returns:
        dominant_period, frequencies, power
    """
    values = series.dropna().astype(float).values
    freqs, power = scipy_periodogram(values)

    # Skip DC component (index 0, freq=0)
    if len(freqs) > 1:
        dominant_idx = int(np.argmax(power[1:])) + 1
        dominant_freq = float(freqs[dominant_idx])
        dominant_period = float(1.0 / dominant_freq) if dominant_freq > 0 else float("inf")
    else:
        dominant_period = float("inf")

    logger.debug("Periodogram dominant period=%.2f", dominant_period)
    return {
        "dominant_period": dominant_period,
        "frequencies": freqs.tolist(),
        "power": power.tolist(),
    }


def detect_trend(series: pd.Series) -> dict:
    """Detect linear trend using OLS regression on time index.

    Returns:
        has_trend, slope, interpretation
    """
    values = series.dropna().astype(float).values
    x = np.arange(len(values), dtype=float)
    slope, intercept, r_value, p_value, std_err = linregress(x, values)

    has_trend = p_value < 0.05
    direction = "upward" if slope > 0 else "downward"

    if has_trend:
        interpretation = (
            f"Statistically significant {direction} trend detected. "
            f"slope={slope:.6f}, p={p_value:.4f}, R²={r_value**2:.4f}."
        )
    else:
        interpretation = (
            f"No significant trend detected. slope={slope:.6f}, p={p_value:.4f}."
        )

    logger.debug("Trend: has_trend=%s slope=%.6f", has_trend, slope)
    return {
        "has_trend": has_trend,
        "slope": float(slope),
        "interpretation": interpretation,
    }
