"""Comprehensive time‑series data cleaning utilities.

This module centralises all data‑cleaning primitives used across the
forecasting pipeline. It implements the industry‑standard preprocessing
steps described in the freeCodeCamp guide "How to Clean Time Series Data in
Python" — auditing, reindexing, missing‑value imputation, outlier detection
& treatment, duplicate resolution, frequency alignment, smoothing, and
schema validation.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from statsmodels.tsa.seasonal import seasonal_decompose

from core.logging_config import get_logger

logger: logging.Logger = get_logger(__name__)

__all__ = [
    "audit_series",
    "time_index_quality",
    "reindex_series",
    "impute_missing",
    "detect_outliers_iqr",
    "detect_outliers_zscore",
    "apply_iqr_clipping",
    "apply_zscore_clipping",
    "treat_outliers",
    "resolve_duplicates",
    "smooth_series",
    "validate_schema",
]


def time_index_quality(
    index: pd.DatetimeIndex,
    freq: str | None = None,
) -> tuple[int, bool, str | None]:
    """Return missing-period count and regularity for a calendar time index."""
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Index must be a pandas DatetimeIndex.")
    unique = pd.DatetimeIndex(index.dropna().unique()).sort_values()
    if len(unique) < 2:
        return 0, True, freq
    inferred = pd.infer_freq(unique) if len(unique) >= 3 else None
    effective_freq = inferred or freq
    if effective_freq:
        try:
            expected = pd.date_range(unique[0], unique[-1], freq=effective_freq)
            missing = len(expected.difference(unique))
            unexpected = len(unique.difference(expected))
            return missing, missing == 0 and unexpected == 0, effective_freq
        except (TypeError, ValueError):
            pass
    diffs = unique.to_series().diff().dropna()
    return 0, bool(diffs.nunique() <= 1), None


def audit_series(series: pd.Series) -> dict[str, Any]:
    """Return a quick audit of a time‑series.

    Mirrors the checklist from the freeCodeCamp article and is consumed by
    the pre‑flight and validation agents.

    Args:
        series: A ``pd.Series`` with a ``DatetimeIndex``.

    Returns:
        A dictionary containing:
            - ``length`` (int): number of observations.
            - ``missing`` (int): count of ``NaN`` values.
            - ``duplicate_timestamps`` (int): number of duplicate index entries.
            - ``irregular`` (bool): ``True`` if the index is not regular.
            - ``freq`` (str | None): inferred pandas frequency alias.
            - ``outlier_counts`` (dict): counts from IQR and Z‑score methods.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a pandas DatetimeIndex.")

    length = len(series)
    missing = int(series.isna().sum())
    duplicate_timestamps = int(series.index.duplicated().sum())

    _, is_regular, freq = time_index_quality(series.index)
    irregular = not is_regular

    outlier_counts = {
        "iqr": int(detect_outliers_iqr(series.dropna())["count"]),
        "zscore": int(detect_outliers_zscore(series.dropna())["count"]),
    }

    return {
        "length": length,
        "missing": missing,
        "duplicate_timestamps": duplicate_timestamps,
        "irregular": irregular,
        "freq": freq,
        "outlier_counts": outlier_counts,
    }


def reindex_series(series: pd.Series, freq: str) -> pd.Series:
    """Reindex a series to a canonical frequency.

    Args:
        series: Input series with a ``DatetimeIndex``.
        freq: Pandas frequency alias (e.g. ``"H"``, ``"D"``).

    Returns:
        Series reindexed to ``freq`` with ``NaN`` for missing timestamps.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a pandas DatetimeIndex.")
    series = series.sort_index()
    full_idx = pd.date_range(
        start=series.index.min(), end=series.index.max(), freq=freq
    )
    reindexed = series.reindex(full_idx)
    logger.info(
        "Reindexed series from %d to %d points using freq='%s'",
        len(series),
        len(reindexed),
        freq,
    )
    return reindexed


def impute_missing(
    series: pd.Series,
    method: Literal["forward-fill", "interpolate", "seasonal-decompose"],
    limit: int | None = None,
) -> pd.Series:
    """Impute missing values using the specified strategy.

    Args:
        series: Input series (may contain ``NaN``).
        method: Imputation method.
        limit: Maximum number of consecutive NaNs to fill (for forward‑fill).

    Returns:
        Series with missing values filled.
    """
    if method == "forward-fill":
        filled = series.ffill(limit=limit)
    elif method == "interpolate":
        filled = series.interpolate(method="time", limit=limit)
    elif method == "seasonal-decompose":
        period = _infer_seasonal_period(series)
        if len(series) < 2 * period:
            logger.warning(
                "Series too short for seasonal decomposition imputation; falling back to interpolation."
            )
            filled = series.interpolate(method="time", limit=limit)
        else:
            temp = _fill_for_decomposition(series, limit)
            try:
                decomp = seasonal_decompose(
                    temp, model="additive", period=period, extrapolate_trend="freq"
                )
            except ValueError as exc:
                logger.warning(
                    "Seasonal decomposition imputation failed; falling back to interpolation: %s",
                    exc,
                )
                filled = temp
            else:
                reconstructed = decomp.trend + decomp.seasonal
                filled = series.copy()
                filled[filled.isna()] = reconstructed[filled.isna()]
                filled = _fill_for_decomposition(filled, limit)
    else:
        raise ValueError(f"Unsupported imputation method: {method}")
    logger.info("Imputed missing values using method='%s'", method)
    return filled


def _fill_for_decomposition(series: pd.Series, limit: int | None = None) -> pd.Series:
    """Return a copy with interior and edge gaps filled for decomposition."""
    filled = series.interpolate(method="time", limit=limit)
    return filled.ffill().bfill()


def detect_outliers_iqr(series: pd.Series) -> dict[str, Any]:
    """Detect outliers using the Interquartile Range (IQR) method.

    Args:
        series: The time series to analyse.

    Returns:
        Dictionary with keys: ``count``, ``ratio``, ``lower_bound``,
        ``upper_bound``, ``interpretation``.
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
        "interpretation": interpretation,
    }


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> dict[str, Any]:
    """Detect outliers using the Z‑score method (moved from ``utils.statistical``).

    Z‑score is more appropriate for normally distributed data, while IQR is
    better for skewed distributions.

    Args:
        series: The time series to analyse.
        threshold: Z‑score threshold for outlier detection (default 3.0).

    Returns:
        Dictionary with keys: ``count``, ``ratio``, ``mean``, ``std``,
        ``lower_bound``, ``upper_bound``, ``interpretation``.
    """
    clean_series = series.dropna()
    mean = clean_series.mean()
    std = clean_series.std()

    if std == 0:
        return {
            "count": 0,
            "ratio": 0.0,
            "mean": mean,
            "std": std,
            "lower_bound": mean,
            "upper_bound": mean,
            "interpretation": "No outliers detected (constant series).",
        }

    z_scores = np.abs((clean_series - mean) / std)
    outliers = clean_series[z_scores > threshold]
    count = len(outliers)
    ratio = count / len(clean_series) if len(clean_series) > 0 else 0

    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std

    interpretation = (
        f"Found {count} outliers ({ratio:.1%}) using Z-score (threshold={threshold}). "
        f"Recommended clipping bounds: [{lower_bound:.2f}, {upper_bound:.2f}]. "
        f"Series mean: {mean:.2f}, std: {std:.2f}."
    )

    return {
        "count": count,
        "ratio": ratio,
        "mean": mean,
        "std": std,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "interpretation": interpretation,
    }


def apply_iqr_clipping(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Apply IQR clipping (Winsorization) to a time series.

    Args:
        series: The time series to clip.
        multiplier: IQR multiplier (default 1.5).

    Returns:
        Clipped series.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    clipped_series = series.clip(lower=lower_bound, upper=upper_bound)
    logger.debug(
        "IQR clipping applied: lower=%.4f upper=%.4f", lower_bound, upper_bound
    )
    return clipped_series


def apply_zscore_clipping(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Apply Z‑score clipping (Winsorization) to a time series.

    Args:
        series: The time series to clip.
        threshold: Z‑score threshold (default 3.0).

    Returns:
        Clipped series.
    """
    mean = series.mean()
    std = series.std()

    if std == 0:
        logger.debug("Z-score clipping: Series is constant, no clipping applied")
        return series

    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std
    clipped_series = series.clip(lower=lower_bound, upper=upper_bound)

    z_scores = np.abs((series - mean) / std)
    clipped_count = int(np.sum(z_scores > threshold))
    clipped_percentage = clipped_count / len(series) * 100 if len(series) > 0 else 0

    logger.debug(
        "Z-score clipping applied: %.2f%% of values clipped", clipped_percentage
    )
    return clipped_series


def treat_outliers(
    series: pd.Series,
    strategy: Literal["clip", "winsorize", "remove", "zscore_clip", "none"],
) -> pd.Series:
    """Apply an outlier handling strategy.

    Args:
        series: Input series.
        strategy: One of ``"clip"``, ``"winsorize"``, ``"remove"``,
            ``"zscore_clip"`` or ``"none"``.

    Returns:
        Series with outliers treated.
    """
    if strategy == "none":
        return series
    if strategy in {"clip", "winsorize"}:
        return apply_iqr_clipping(series)
    if strategy == "zscore_clip":
        return apply_zscore_clipping(series)
    if strategy == "remove":
        info = detect_outliers_iqr(series)
        series = series.copy()
        mask = (series < info["lower_bound"]) | (series > info["upper_bound"])
        series[mask] = np.nan
        return series
    raise ValueError(f"Unsupported outlier strategy: {strategy}")


def resolve_duplicates(
    series: pd.Series,
    strategy: Literal["keep-first", "mean", "sum", "latest"],
) -> pd.Series:
    """Resolve duplicate timestamps.

    Args:
        series: Series with possibly duplicated index.
        strategy: Resolution method.

    Returns:
        Series with unique index.
    """
    if not series.index.duplicated().any():
        return series
    if strategy == "keep-first":
        return series[~series.index.duplicated(keep="first")]
    if strategy == "latest":
        return series[~series.index.duplicated(keep="last")]
    if strategy == "mean":
        return series.groupby(level=0).mean()
    if strategy == "sum":
        return series.groupby(level=0).sum()
    raise ValueError(f"Unsupported duplicate strategy: {strategy}")


def smooth_series(
    series: pd.Series,
    method: Literal["ewma", "savgol", "none"],
    **params: Any,
) -> pd.Series:
    """Smooth a series using the selected method.

    Args:
        series: Input series.
        method: ``"ewma"``, ``"savgol"`` or ``"none"``.
        **params: Additional parameters for the smoothing algorithm.

    Returns:
        Smoothed series.
    """
    if method == "none":
        return series
    if method == "ewma":
        span = params.get("span", 6)
        adjust = params.get("adjust", False)
        return series.ewm(span=span, adjust=adjust).mean()
    if method == "savgol":
        if len(series) < 3:
            logger.warning("Series too short for Savitzky-Golay smoothing; skipping.")
            return series
        window = min(params.get("window", 11), len(series))
        polyorder = params.get("polyorder", 2)
        if window % 2 == 0:
            window += 1
        if window > len(series):
            window -= 2
        if window <= polyorder:
            logger.warning("Series too short for Savitzky-Golay smoothing; skipping.")
            return series
        filtered = savgol_filter(
            series.values, window_length=window, polyorder=polyorder
        )
        return pd.Series(filtered, index=series.index)
    raise ValueError(f"Unsupported smoothing method: {method}")


def validate_schema(
    series: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate a series against a simple schema.

    The ``config`` dictionary may contain:
        - ``expected_freq`` (str): pandas frequency alias.
        - ``max_missing_rate`` (float): proportion of allowed missing values.
        - ``min_value`` / ``max_value`` (float): allowed range.

    Args:
        series: Series to validate.
        config: Validation configuration.

    Returns:
        Dictionary with boolean flags and diagnostic messages.
    """
    report: dict[str, Any] = {}
    inferred = pd.infer_freq(series.index)
    report["freq_regular"] = bool(inferred == config.get("expected_freq"))
    missing_rate = series.isna().mean()
    report["missing_below_threshold"] = bool(
        missing_rate <= config.get("max_missing_rate", 0.05)
    )
    report["missing_rate"] = round(missing_rate, 4)
    if series.dropna().empty:
        report["values_in_range"] = True
        report["out_of_range_count"] = 0
    else:
        in_range = series.dropna().between(
            config.get("min_value", -np.inf),
            config.get("max_value", np.inf),
        )
        report["values_in_range"] = bool(in_range.all())
        report["out_of_range_count"] = int((~in_range).sum())
    report["no_duplicates"] = bool(not series.index.duplicated().any())
    report["index_monotonic"] = bool(series.index.is_monotonic_increasing)
    return report


# ---------------------------------------------------------------------------


def _infer_seasonal_period(series: pd.Series) -> int:
    """Infer seasonal period from the series frequency.

    Returns an integer period; defaults to 12 (monthly).
    """
    freq = series.index.freq or pd.infer_freq(series.index)
    if not freq:
        return 12
    mapping = {
        "M": 12,
        "ME": 12,
        "MS": 12,
        "Q": 4,
        "QE": 4,
        "QS": 4,
        "W": 52,
        "D": 7,
        "B": 5,
        "H": 24,
        "h": 24,
    }
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
        base = offset.name
    except (ValueError, TypeError):
        base = str(freq)
    # Strip anchored-suffix (e.g. "W-SUN" -> "W", "QS-JAN" -> "QS").
    base = base.split("-")[0]
    return mapping.get(base, 12)
