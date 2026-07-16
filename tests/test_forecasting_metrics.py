"""Tests for shared forecast metric calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting.contracts import ForecastAdapterResult, ForecastFitStatus
from forecasting.metrics import calculate_forecast_metrics, calculate_holdout_metrics
from forecasting import ewma_model
from agents.forecasting_agent import _has_required_metrics
from services.baseline_service import run_baseline_models
from utils.statistical_analysis import analyze_residuals


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

    metrics = calculate_holdout_metrics(test, model)

    assert metrics.rmse == pytest.approx(np.sqrt(8.0))
    assert metrics.mae == pytest.approx(8.0 / 3.0)
    assert metrics.mape == pytest.approx(np.mean([0.2, 0.1, 0.1]) * 100)


def test_calculate_holdout_metrics_marks_missing_evidence_unavailable() -> None:
    """Missing evaluation evidence is not encoded as perfect performance."""
    empty = calculate_holdout_metrics(pd.Series(dtype=float), None)
    missing_model = calculate_holdout_metrics(pd.Series([1.0]), None)
    assert empty.rmse is None
    assert missing_model.rmse is None
    assert empty.unavailable_reasons


def test_mape_is_unavailable_when_actual_contains_zero() -> None:
    """MAPE does not use an arbitrary epsilon for zero actual values."""
    metrics = calculate_forecast_metrics(np.array([0.0, 10.0]), np.array([1.0, 9.0]))
    assert metrics.mae == pytest.approx(1.0)
    assert metrics.mape is None
    assert "mape" in metrics.unavailable_reasons


def test_wape_uses_absolute_actual_denominator() -> None:
    """WAPE should remain positive when actual values include negatives."""
    metrics = calculate_forecast_metrics(
        np.array([-10.0, 10.0]),
        np.array([-8.0, 8.0]),
    )

    assert metrics.wape == pytest.approx(0.2)


def test_seasonal_naive_cycles_final_season_for_long_horizon() -> None:
    """Seasonal naive forecasts should wrap through the last seasonal window."""
    series = pd.Series(
        [
            10.0,
            20.0,
            10.0,
            20.0,
            10.0,
            20.0,
            10.0,
            20.0,
            10.0,
            20.0,
            10.0,
            20.0,
            10.0,
            20.0,
            10.0,
        ]
    )

    metrics = run_baseline_models(series, forecast_horizon=3, seasonal_period=2)

    assert metrics["Seasonal Naive"].metrics.mae == pytest.approx(0.0)


def test_analyze_residuals_bounds_ljung_box_lag_for_short_series() -> None:
    """Short residual series should not request an impossible Ljung-Box lag."""
    diagnostics = analyze_residuals(pd.Series([1.0, -1.0, 0.5]))

    assert diagnostics.ljung_box_p_value is not None


def test_ewma_returns_typed_result_with_central_metrics() -> None:
    """EWMA returns a ForecastAdapterResult with centrally computed metrics."""
    result = ewma_model.fit_ewma(pd.Series(np.arange(1.0, 21.0)), forecast_horizon=3)

    assert isinstance(result, ForecastAdapterResult)
    assert result.status == ForecastFitStatus.OK
    assert result.metrics.rmse is not None
    assert result.metrics.mae is not None
    assert len(result.forecast) == 3
    assert result.fitted_configuration["model"] == "EWMA"
    assert result.fitted_configuration["alpha"] is not None


def test_ewma_short_series_returns_not_estimable() -> None:
    """Short series returns not_estimable status with persistence fallback."""
    result = ewma_model.fit_ewma(pd.Series([1.0, 2.0]), forecast_horizon=2)

    assert result.status == ForecastFitStatus.NOT_ESTIMABLE
    assert result.is_fallback is True
    assert result.metrics.rmse is None
    assert result.metrics.mae is None
    assert result.metrics.mape is None
    assert not _has_required_metrics(result)
