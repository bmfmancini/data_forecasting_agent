"""Statistically principled simulation helpers for ETS prediction intervals."""

from __future__ import annotations

import numpy as np


def simulate_ets_paths(
    fitted: object,
    forecast_horizon: int,
    *,
    repetitions: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Simulate future paths through fitted exponential-smoothing states.

    Residuals are centred before resampling so the bootstrap innovations have
    mean zero.  ``statsmodels`` then propagates those innovations through the
    fitted level, trend, damping, and seasonal state equations.

    Args:
        fitted: A fitted statsmodels Holt-Winters/SES results object exposing
            ``resid`` and ``simulate``.
        forecast_horizon: Number of future periods to simulate.
        repetitions: Number of bootstrap paths.
        seed: Seed for deterministic tests and reproducible reports.

    Returns:
        Array with shape ``(forecast_horizon, repetitions)``. An empty second
        dimension indicates that innovations were unavailable.

    Raises:
        ValueError: If the horizon or repetitions are not positive.
    """
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be positive.")
    if repetitions < 1:
        raise ValueError("repetitions must be positive.")

    residuals = np.asarray(fitted.resid, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        return np.empty((forecast_horizon, 0), dtype=float)

    innovations = residuals - float(np.mean(residuals))
    rng = np.random.default_rng(seed)
    random_errors = rng.choice(
        innovations,
        size=(forecast_horizon, repetitions),
        replace=True,
    )
    simulations = fitted.simulate(
        forecast_horizon,
        anchor="end",
        repetitions=repetitions,
        error="add",
        random_errors=random_errors,
    )
    paths = np.asarray(simulations, dtype=float)
    if paths.shape != (forecast_horizon, repetitions):
        paths = paths.reshape(forecast_horizon, repetitions)
    return paths


def ets_prediction_interval(
    fitted: object,
    forecast_horizon: int,
    *,
    repetitions: int = 1000,
    seed: int = 42,
    coverage: float = 0.95,
) -> tuple[list[float], list[float]]:
    """Return a residual-bootstrap ETS prediction interval.

    Args:
        fitted: Fitted statsmodels exponential-smoothing results object.
        forecast_horizon: Number of future periods.
        repetitions: Number of simulated paths.
        seed: Random seed.
        coverage: Central interval coverage in ``(0, 1)``.

    Returns:
        Lower and upper interval bounds.

    Raises:
        ValueError: If coverage is outside ``(0, 1)``.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between zero and one.")
    paths = simulate_ets_paths(
        fitted,
        forecast_horizon,
        repetitions=repetitions,
        seed=seed,
    )
    if paths.shape[1] == 0:
        return [], []
    tail = (1.0 - coverage) / 2.0
    return (
        np.quantile(paths, tail, axis=1).astype(float).tolist(),
        np.quantile(paths, 1.0 - tail, axis=1).astype(float).tolist(),
    )
