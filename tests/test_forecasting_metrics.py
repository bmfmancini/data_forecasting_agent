"""Tests for shared forecast metric calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting.metrics import calculate_holdout_metrics


class _Model:
    """Minimal pmdarima-style model used by metric tests."""

    def __init__(self, forecast: np.ndarray) -> None:
        self._forecast = forecast

    def predict(
        self,
        n_periods: int,
        return_conf_int: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the configured forecast and placeholder intervals."""
        return self._forecast[:n_periods], np.zeros((n_periods, 2))


def test_calculate_holdout_metrics_matches_expected_values() -> None:
    """RMSE, MAE, and MAPE are calculated from holdout residuals."""
    test = pd.Series([10.0, 20.0, 40.0])
    model = _Model(np.array([8.0, 22.0, 44.0]))

    rmse, mae, mape = calculate_holdout_metrics(test, model)

    assert rmse == pytest.approx(np.sqrt(8.0))
    assert mae == pytest.approx(8.0 / 3.0)
    assert mape == pytest.approx(np.mean([0.2, 0.1, 0.1]) * 100)


def test_calculate_holdout_metrics_zeroes_empty_or_missing_model() -> None:
    """Fallback behavior remains stable when metrics cannot be calculated."""
    assert calculate_holdout_metrics(pd.Series(dtype=float), None) == (0.0, 0.0, 0.0)
    assert calculate_holdout_metrics(pd.Series([1.0]), None) == (0.0, 0.0, 0.0)
