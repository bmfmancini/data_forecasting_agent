"""Prediction intervals generated from the fitted forecasting process."""

from __future__ import annotations

import numpy as np


def smoothing_paths(
    fitted, horizon: int, *, seed: int = 42, repetitions: int = 2000
) -> np.ndarray:
    """Simulate the fitted level/trend/seasonal recursion with centred errors.

    Conditional on fitted parameters; parameter uncertainty is not included.
    Returned layout is (simulation, horizon).
    """
    residuals = np.asarray(fitted.resid, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size < 2:
        raise ValueError("At least two fitted innovations are needed for intervals.")
    residuals = residuals - residuals.mean()
    errors = np.random.default_rng(seed).choice(residuals, size=(horizon, repetitions))
    paths = (
        np.asarray(
            fitted.simulate(
                horizon,
                anchor="end",
                repetitions=repetitions,
                error="add",
                random_errors=errors,
            ),
            dtype=float,
        )
        .reshape(horizon, repetitions)
        .T
    )
    if not np.isfinite(paths).all():
        raise ValueError("Simulation produced non-finite future paths.")
    return paths


def path_intervals(
    paths: np.ndarray, coverage: float = 0.95
) -> tuple[list[float], list[float]]:
    alpha = (1 - coverage) / 2
    return (
        np.quantile(paths, alpha, axis=0).tolist(),
        np.quantile(paths, 1 - alpha, axis=0).tolist(),
    )
