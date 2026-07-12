"""Shared metric calculations for forecast model adapters."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class PredictsIntervals(Protocol):
    """Protocol for fitted models that can forecast with confidence intervals."""

    def predict(
        self,
        n_periods: int,
        return_conf_int: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return forecast values and confidence intervals."""


def calculate_holdout_metrics(
    test: pd.Series,
    model: PredictsIntervals | None,
) -> tuple[float, float, float]:
    """Calculate RMSE, MAE, and MAPE for a fitted forecast model.

    Args:
        test: Holdout observations to compare against model predictions.
        model: Fitted model exposing a pmdarima-style ``predict`` method.

    Returns:
        Tuple of ``(rmse, mae, mape)``. Empty holdout data or a missing model
        returns zeroed metrics so callers can preserve existing fallback behavior.
    """
    if len(test) == 0 or model is None:
        return 0.0, 0.0, 0.0

    test_fc, _ = model.predict(n_periods=len(test), return_conf_int=True)
    residuals = test.values - test_fc
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    mape = float(np.mean(np.abs(residuals / (test.values + 1e-8))) * 100)
    return rmse, mae, mape
