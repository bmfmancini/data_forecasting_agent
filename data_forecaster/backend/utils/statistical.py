from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.signal import periodogram as scipy_periodogram
from scipy.stats import linregress, boxcox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

from core.logging_config import get_logger

logger = get_logger(__name__)


def run_adf_test(series: pd.Series) -> dict[str, Any]:
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


def run_kpss_test(series: pd.Series) -> dict[str, Any]:
    """KPSS test for stationarity (null = stationary).

    Returns:
        statistic, p_value, is_stationary, interpretation
    """
    values = series.dropna().astype(float).values
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


def run_stl_decomposition(series: pd.Series, period: int = 12) -> dict[str, list[float]]:
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


def compute_acf_pacf(series: pd.Series, lags: int = 40) -> dict[str, list[float] | list[int]]:
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


def run_periodogram(series: pd.Series) -> dict[str, Any]:
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


def detect_trend(series: pd.Series) -> dict[str, Any]:
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

def detect_outliers_iqr(series: pd.Series) -> dict[str, Any]:
    """
    Detects outliers using the Interquartile Range (IQR) method.
    Returns the count, ratio, and recommended bounds.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    count = len(outliers)
    ratio = count / len(series) if len(series) > 0 else 0
    
    interpretation = (
        f"Found {count} outliers ({ratio:.1%}). "
        f"Recommended clipping bounds: [{lower_bound:.2f}, {upper_bound:.2f}]."
    )
    
    return {
        "count": count,
        "ratio": ratio,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "interpretation": interpretation
    }

def run_white_noise_test(series: pd.Series) -> dict[str, Any]:
    """
    Ljung-Box test for white noise. If p-value > 0.05, the series is likely 
    random noise and forecasting will be ineffective.
    """
    # Test up to 10 lags or 1/5th of series length
    lags = min(10, len(series) // 5)
    if lags < 1: lags = 1
    
    res = acorr_ljungbox(series.dropna(), lags=[lags], return_df=True)
    p_value = float(res.lb_pvalue.iloc[0])
    is_white_noise = p_value > 0.05
    
    interpretation = (
        f"Ljung-Box p-value: {p_value:.4f}. "
        f"{'Series is white noise (random).' if is_white_noise else 'Series contains significant signal.'}"
    )
    
    return {
        "p_value": p_value,
        "is_white_noise": is_white_noise,
        "interpretation": interpretation
    }

def check_variance_stability(series: pd.Series) -> dict[str, Any]:
    """
    Checks if the variance changes with the level (heteroskedasticity).
    If the correlation between mean and std of rolling windows is high, 
    a Box-Cox transform is recommended.
    """
    if len(series) < 20:
        return {"is_unstable": False, "correlation": 0.0, "interpretation": "Series too short for variance check."}
    
    window = max(5, len(series) // 10)
    rolling_mean = series.rolling(window=window).mean().dropna()
    rolling_std = series.rolling(window=window).std().dropna()
    
    corr = float(rolling_mean.corr(rolling_std))
    is_unstable = abs(corr) > 0.6 and series.min() > 0
    
    interpretation = (
        f"Variance-to-mean correlation: {corr:.2f}. "
        f"{'Variance is unstable; transformation recommended.' if is_unstable else 'Variance appears stable.'}"
    )
    
    return {
        "is_unstable": is_unstable,
        "correlation": corr,
        "interpretation": interpretation
    }

def apply_boxcox(series: pd.Series) -> tuple[pd.Series, float]:
    """Applies Box-Cox transformation. Returns (transformed_series, lambda_val)."""
    # Ensure strictly positive
    vals = series.values + (abs(series.min()) + 1 if series.min() <= 0 else 0)
    transformed, lam = boxcox(vals)
    return pd.Series(transformed, index=series.index), float(lam)


def apply_iqr_clipping(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Applies IQR clipping (Winsorization) to remove outliers from a time series.
    
    Args:
        series: The time series to clip
        multiplier: IQR multiplier (default 1.5, can be adjusted for more/less aggressive clipping)
        
    Returns:
        pd.Series: The clipped time series
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    # Clip values to within the bounds
    clipped_series = series.clip(lower=lower_bound, upper=upper_bound)
    
    logger.debug(
        "IQR clipping applied: %.2f%% of values clipped",
        (len(series) - len(clipped_series[clipped_series == series])) / len(series) * 100
    )
    
    return clipped_series


def detect_change_points(series: pd.Series, method: str = "cusum", threshold: float = None) -> dict[str, Any]:
    """
    Detects change points in a time series using various methods.
    
    Args:
        series: The time series to analyze
        method: Change point detection method ('cusum', 'rolling_mean', or 'fft')
        threshold: Threshold for change detection (auto-calculated if None)
        
    Returns:
        dict with keys: change_points, method_used, threshold, interpretation
    """
    series = series.dropna()
    change_points = []
    
    if len(series) < 10:
        return {
            "change_points": [],
            "method_used": method,
            "threshold": None,
            "interpretation": "Series too short for change point detection."
        }
    
    if method == "cusum":
        # Cumulative Sum method
        if threshold is None:
            threshold = 2 * series.std()
        
        # Calculate cumulative sum of deviations from mean
        mean_val = series.mean()
        cusum = np.cumsum(series - mean_val)
        
        # Find points where CUSUM exceeds threshold
        for i in range(len(cusum)):
            if abs(cusum[i]) > threshold:
                change_points.append(series.index[i])
                
    elif method == "rolling_mean":
        # Rolling mean comparison method
        if threshold is None:
            threshold = series.std()
            
        # Calculate rolling means with different window sizes
        window1 = max(3, len(series) // 10)
        window2 = max(5, len(series) // 5)
        
        rolling1 = series.rolling(window=window1, center=True).mean()
        rolling2 = series.rolling(window=window2, center=True).mean()
        
        # Find points where the difference exceeds threshold
        diff = abs(rolling1 - rolling2)
        for i in range(len(diff)):
            if diff.iloc[i] > threshold:
                change_points.append(series.index[i])
                
    elif method == "fft":
        # Frequency domain method
        if threshold is None:
            threshold = 0.1 * series.std()
            
        # Apply FFT and look for frequency changes
        # This is a simplified approach - real implementation would be more complex
        # For now, we'll use a variance-based approach on rolling windows
        window = max(10, len(series) // 4)
        rolling_var = series.rolling(window=window, center=True).var()
        
        # Find points where variance changes significantly
        mean_var = rolling_var.mean()
        for i in range(len(rolling_var)):
            if rolling_var.iloc[i] > (mean_var + threshold):
                change_points.append(series.index[i])
    
    # Remove duplicates and sort
    change_points = sorted(list(set(change_points)))
    
    interpretation = (
        f"Detected {len(change_points)} change points using {method} method. "
        f"Change points: {change_points[:5]}{'...' if len(change_points) > 5 else ''}. "
        f"{'Consider structural break analysis if change points are significant.' if len(change_points) > 0 else 'No significant structural breaks detected.'}"
    )
    
    return {
        "change_points": change_points,
        "method_used": method,
        "threshold": threshold,
        "interpretation": interpretation
    }
