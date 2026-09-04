"""Deterministic synthetic regression fixtures for forecast model testing.

Every fixture uses a fixed random seed so results are reproducible across
runs. Fixtures return :class:`pd.Series` with a regular ``DatetimeIndex``
unless otherwise noted, so that adapters and metrics can exercise the same
code paths as production data.

Fixture categories (per R1 requirements):
- constant and near-constant series
- stationary AR series
- random walk
- additive and multiplicative seasonality
- trend without seasonality
- zeros and negative values
- missing and duplicate timestamps
- short seasonal series (fewer than two cycles)
- isolated anomalies
- structural breaks
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed seed for all stochastic fixtures.
_FIXTURE_SEED = 42


def _index(n: int, freq: str = "MS") -> pd.DatetimeIndex:
    """Return a regular monthly DatetimeIndex of length ``n``."""
    return pd.date_range(start="2020-01-01", periods=n, freq=freq)


def constant_series(n: int = 24) -> pd.Series:
    """Return a constant series (all values identical).

    Args:
        n: Number of observations.

    Returns:
        A constant series with value 100.0 at every point.
    """
    return pd.Series([100.0] * n, index=_index(n), name="constant")


def near_constant_series(n: int = 24) -> pd.Series:
    """Return a near-constant series with tiny additive noise.

    Args:
        n: Number of observations.

    Returns:
        A series centered at 50.0 with noise of amplitude 0.01.
    """
    rng = np.random.default_rng(_FIXTURE_SEED)
    return pd.Series(
        50.0 + rng.normal(0, 0.01, n), index=_index(n), name="near_constant"
    )


def stationary_ar_series(
    n: int = 60, phi: float = 0.7, noise_std: float = 1.0
) -> pd.Series:
    """Return a stationary AR(1) series: ``y_t = phi * y_{t-1} + e_t``.

    Args:
        n: Number of observations.
        phi: Autoregressive coefficient (must be < 1 for stationarity).
        noise_std: Standard deviation of the innovation noise.

    Returns:
        A stationary AR(1) series.
    """
    rng = np.random.default_rng(_FIXTURE_SEED)
    values = np.zeros(n)
    for t in range(1, n):
        values[t] = phi * values[t - 1] + rng.normal(0, noise_std)
    return pd.Series(values, index=_index(n), name="stationary_ar")


def random_walk_series(n: int = 60, noise_std: float = 1.0) -> pd.Series:
    """Return a random walk: ``y_t = y_{t-1} + e_t``.

    Args:
        n: Number of observations.
        noise_std: Standard deviation of the innovation noise.

    Returns:
        A random walk series starting at 0.
    """
    rng = np.random.default_rng(_FIXTURE_SEED)
    return pd.Series(
        np.cumsum(rng.normal(0, noise_std, n)), index=_index(n), name="random_walk"
    )


def additive_seasonal_series(
    n: int = 48, period: int = 12, amplitude: float = 10.0, noise_std: float = 1.0
) -> pd.Series:
    """Return a series with additive seasonality and no trend.

    Args:
        n: Number of observations.
        period: Seasonal period.
        amplitude: Peak seasonal amplitude.
        noise_std: Standard deviation of additive noise.

    Returns:
        An additive seasonal series centered at 50.0.
    """
    rng = np.random.default_rng(_FIXTURE_SEED)
    t = np.arange(n)
    seasonal = amplitude * np.sin(2 * np.pi * t / period)
    noise = rng.normal(0, noise_std, n)
    return pd.Series(50.0 + seasonal + noise, index=_index(n), name="additive_seasonal")


def multiplicative_seasonal_series(
    n: int = 48, period: int = 12, base: float = 50.0, factor: float = 0.3
) -> pd.Series:
    """Return a series with multiplicative seasonality (all positive).

    Args:
        n: Number of observations.
        period: Seasonal period.
        base: Base level of the series.
        factor: Multiplicative seasonal factor (0.3 means ±30% of base).

    Returns:
        A strictly positive multiplicative seasonal series.
    """
    rng = np.random.default_rng(_FIXTURE_SEED)
    t = np.arange(n)
    seasonal = 1.0 + factor * np.sin(2 * np.pi * t / period)
    noise = 1.0 + rng.normal(0, 0.01, n)
    return pd.Series(
        base * seasonal * noise, index=_index(n), name="multiplicative_seasonal"
    )


def trend_series(
    n: int = 48, slope: float = 2.0, intercept: float = 10.0, noise_std: float = 1.0
) -> pd.Series:
    """Return a linear trend series without seasonality.

    Args:
        n: Number of observations.
        slope: Slope of the linear trend.
        intercept: Starting value of the trend.
        noise_std: Standard deviation of additive noise.

    Returns:
        A trending series with no seasonal component.
    """
    rng = np.random.default_rng(_FIXTURE_SEED)
    t = np.arange(n)
    noise = rng.normal(0, noise_std, n)
    return pd.Series(intercept + slope * t + noise, index=_index(n), name="trend")


def zeros_series(n: int = 24) -> pd.Series:
    """Return a series containing all zeros.

    Args:
        n: Number of observations.

    Returns:
        A series of zeros (tests MAPE unavailability).
    """
    return pd.Series([0.0] * n, index=_index(n), name="zeros")


def negative_values_series(n: int = 36) -> pd.Series:
    """Return a series with negative values.

    Args:
        n: Number of observations.

    Returns:
        A series with both positive and negative values.
    """
    rng = np.random.default_rng(_FIXTURE_SEED)
    return pd.Series(rng.normal(0, 5, n), index=_index(n), name="negative_values")


def missing_timestamps_series(n: int = 36, missing_count: int = 5) -> pd.Series:
    """Return a series with missing timestamps (gaps in the index).

    Args:
        n: Number of observations before removing gaps.
        missing_count: Number of timestamps to remove.

    Returns:
        A series with an irregular index (some periods missing).
    """
    full = pd.Series(np.arange(n, dtype=float), index=_index(n), name="missing_timestamps")
    rng = np.random.default_rng(_FIXTURE_SEED)
    drop_idx = rng.choice(n, size=missing_count, replace=False)
    return full.drop(full.index[drop_idx])


def duplicate_timestamps_series(n: int = 30) -> pd.Series:
    """Return a series with duplicate timestamps.

    Args:
        n: Number of observations (some will share timestamps).

    Returns:
        A series with a non-unique index.
    """
    idx = _index(n)
    # Duplicate the last 3 timestamps
    dup_idx = idx.append(idx[-3:])
    values = np.arange(len(dup_idx), dtype=float)
    return pd.Series(values, index=dup_idx, name="duplicate_timestamps")


def short_seasonal_series(period: int = 12) -> pd.Series:
    """Return a series shorter than two full seasonal cycles.

    Args:
        period: The seasonal period that would be requested.

    Returns:
        A series with ``period + 1`` observations (less than 2 * period).
    """
    n = period + 1
    rng = np.random.default_rng(_FIXTURE_SEED)
    return pd.Series(50.0 + rng.normal(0, 2, n), index=_index(n), name="short_seasonal")


def isolated_anomalies_series(n: int = 48) -> pd.Series:
    """Return a series with isolated spike anomalies.

    Args:
        n: Number of observations.

    Returns:
        A series with two large spikes at known positions.
    """
    rng = np.random.default_rng(_FIXTURE_SEED)
    values = 50.0 + rng.normal(0, 1, n)
    # Inject two large positive anomalies
    values[10] += 30.0
    values[30] -= 25.0
    return pd.Series(values, index=_index(n), name="isolated_anomalies")


def structural_break_series(n: int = 48, break_point: int = 24) -> pd.Series:
    """Return a series with a level shift (structural break).

    Args:
        n: Number of observations.
        break_point: Index at which the level shifts.

    Returns:
        A series with a constant level before ``break_point`` and a
        different constant level after.
    """
    rng = np.random.default_rng(_FIXTURE_SEED)
    values = np.zeros(n)
    values[:break_point] = 20.0 + rng.normal(0, 1, break_point)
    values[break_point:] = 60.0 + rng.normal(0, 1, n - break_point)
    return pd.Series(values, index=_index(n), name="structural_break")


ALL_FIXTURES: dict[str, callable] = {
    "constant": constant_series,
    "near_constant": near_constant_series,
    "stationary_ar": stationary_ar_series,
    "random_walk": random_walk_series,
    "additive_seasonal": additive_seasonal_series,
    "multiplicative_seasonal": multiplicative_seasonal_series,
    "trend": trend_series,
    "zeros": zeros_series,
    "negative_values": negative_values_series,
    "missing_timestamps": missing_timestamps_series,
    "duplicate_timestamps": duplicate_timestamps_series,
    "short_seasonal": short_seasonal_series,
    "isolated_anomalies": isolated_anomalies_series,
    "structural_break": structural_break_series,
}
