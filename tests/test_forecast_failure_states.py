"""Failure-state tests for forecast model adapters and ranking.

These tests verify the R1 honesty guarantees:
- Failed/degraded models cannot win ranking.
- Missing holdout metrics remain ``None``.
- Short-series persistence output is explicitly ``not_estimable``.
- No successful evaluation is fabricated after a fitting exception.
- All model result objects serialize correctly.
- Fitted configuration survives refitting.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from forecasting.contracts import ForecastAdapterResult, ForecastFitStatus
from forecasting.arima_model import fit_arima
from forecasting.ewma_model import fit_ewma
from forecasting.holt_winters import fit_holt_winters
from forecasting.sarima_model import fit_sarima
from agents.forecasting_agent import _has_required_metrics


class TestFailedModelCannotWinRanking:
    """A failed or degraded model must never be rankable."""

    def test_not_estimable_is_not_rankable(self) -> None:
        """A not_estimable result is excluded from ranking."""
        result = ForecastAdapterResult(
            status=ForecastFitStatus.NOT_ESTIMABLE,
            failure_reason="Too short",
            is_fallback=True,
            forecast=[1.0],
        )
        assert not _has_required_metrics(result)

    def test_degraded_is_not_rankable(self) -> None:
        """A degraded result (metrics unavailable) is excluded from ranking."""
        result = ForecastAdapterResult(
            status=ForecastFitStatus.DEGRADED,
            failure_reason="Metrics unavailable",
            forecast=[1.0],
        )
        assert not _has_required_metrics(result)

    def test_failed_is_not_rankable(self) -> None:
        """A failed result is excluded from ranking."""
        result = ForecastAdapterResult(
            status=ForecastFitStatus.FAILED,
            failure_reason="Exception during fit",
            forecast=[1.0],
        )
        assert not _has_required_metrics(result)

    def test_ok_with_none_metrics_is_not_rankable(self) -> None:
        """Even an ok status with None metrics is not rankable."""
        result = ForecastAdapterResult(
            status=ForecastFitStatus.OK,
            forecast=[1.0],
        )
        assert not _has_required_metrics(result)

    def test_ok_with_all_metrics_is_rankable(self) -> None:
        """An ok status with all required metrics is rankable."""
        from forecasting.contracts import ForecastMetrics

        result = ForecastAdapterResult(
            status=ForecastFitStatus.OK,
            forecast=[1.0],
            metrics=ForecastMetrics(rmse=1.0, mae=0.5, mape=10.0, wape=0.1, mase=0.8),
        )
        assert _has_required_metrics(result)

    def test_ok_with_nan_rmse_is_not_rankable(self) -> None:
        """An ok status with NaN RMSE is not rankable."""
        from forecasting.contracts import ForecastMetrics

        result = ForecastAdapterResult(
            status=ForecastFitStatus.OK,
            forecast=[1.0],
            metrics=ForecastMetrics(rmse=float("nan"), mae=0.5, mape=10.0),
        )
        assert not _has_required_metrics(result)


class TestMissingMetricsRemainNone:
    """Missing holdout metrics must remain None, never zero."""

    def test_arima_short_series_metrics_are_none(self) -> None:
        """ARIMA on a 2-point series returns not_estimable with None metrics."""
        series = pd.Series([1.0, 2.0])
        result = fit_arima(series, forecast_horizon=2)
        assert result.status == ForecastFitStatus.NOT_ESTIMABLE
        assert result.metrics.rmse is None
        assert result.metrics.mae is None
        assert result.metrics.mape is None

    def test_ewma_short_series_metrics_are_none(self) -> None:
        """EWMA on a 2-point series returns not_estimable with None metrics."""
        series = pd.Series([1.0, 2.0])
        result = fit_ewma(series, forecast_horizon=2)
        assert result.status == ForecastFitStatus.NOT_ESTIMABLE
        assert result.metrics.rmse is None
        assert result.metrics.mae is None
        assert result.metrics.mape is None

    def test_zeros_series_mape_is_none(self) -> None:
        """MAPE is None when actuals contain zeros."""
        from forecasting.metrics import calculate_forecast_metrics

        metrics = calculate_forecast_metrics(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0, 3.0])
        )
        assert metrics.mape is None
        assert "mape" in metrics.unavailable_reasons


class TestShortSeriesPersistence:
    """Short-series persistence output must be explicitly not_estimable."""

    def test_arima_short_series_is_not_estimable(self) -> None:
        series = pd.Series([5.0])
        result = fit_arima(series, forecast_horizon=3)
        assert result.status == ForecastFitStatus.NOT_ESTIMABLE
        assert result.is_fallback is True
        assert result.failure_reason is not None

    def test_ewma_short_series_is_not_estimable(self) -> None:
        series = pd.Series([5.0])
        result = fit_ewma(series, forecast_horizon=3)
        assert result.status == ForecastFitStatus.NOT_ESTIMABLE
        assert result.is_fallback is True
        assert result.failure_reason is not None

    def test_short_series_forecast_is_persistence(self) -> None:
        """Short-series forecast repeats the last observed value."""
        series = pd.Series([42.0])
        result = fit_arima(series, forecast_horizon=3)
        assert result.forecast == [42.0, 42.0, 42.0]


class TestNoFabricatedEvaluation:
    """No successful evaluation is fabricated after a fitting exception."""

    def test_arima_empty_series_does_not_produce_metrics(self) -> None:
        """An empty series must not produce fabricated metrics."""
        series = pd.Series([], dtype=float)
        result = fit_arima(series, forecast_horizon=2)
        assert result.metrics.rmse is None
        assert result.metrics.mae is None

    def test_ewma_empty_series_does_not_produce_metrics(self) -> None:
        """An empty series must not produce fabricated metrics."""
        series = pd.Series([], dtype=float)
        result = fit_ewma(series, forecast_horizon=2)
        assert result.metrics.rmse is None
        assert result.metrics.mae is None


class TestResultSerialization:
    """All model result objects must serialize correctly."""

    @pytest.mark.parametrize(
        "adapter,kwargs",
        [
            (fit_arima, {}),
            (fit_ewma, {}),
            (fit_holt_winters, {}),
            (fit_sarima, {"seasonal_period": 12}),
        ],
    )
    def test_result_serializes_to_json(self, adapter, kwargs) -> None:
        """ForecastAdapterResult must be JSON-serializable."""
        series = pd.Series(
            np.arange(1.0, 25.0),
            index=pd.date_range("2020-01-01", periods=24, freq="MS"),
        )
        result = adapter(series, forecast_horizon=3, **kwargs)
        data = result.model_dump()
        # Pydantic model_dump produces JSON-compatible types
        json_str = json.dumps(data, default=str)
        assert json.loads(json_str) is not None

    def test_contract_result_serializes(self) -> None:
        """A bare ForecastAdapterResult serializes correctly."""
        result = ForecastAdapterResult(
            status=ForecastFitStatus.OK,
            forecast=[1.0, 2.0],
            lower_ci=[0.5, 1.5],
            upper_ci=[1.5, 2.5],
        )
        data = result.model_dump()
        json_str = json.dumps(data, default=str)
        parsed = json.loads(json_str)
        assert parsed["status"] == "ok"
        assert parsed["forecast"] == [1.0, 2.0]


class TestFittedConfigurationSurvivesRefit:
    """Fitted configuration must survive the train-to-full refit."""

    def test_arima_fitted_configuration_has_order(self) -> None:
        """ARIMA fitted_configuration contains the selected order."""
        series = pd.Series(
            np.arange(1.0, 49.0) + np.random.default_rng(42).normal(0, 1, 48),
            index=pd.date_range("2020-01-01", periods=48, freq="MS"),
        )
        result = fit_arima(series, forecast_horizon=6)
        config = result.fitted_configuration
        assert config["model"] == "ARIMA"
        assert config["order"] is not None
        assert isinstance(config["order"], list)
        assert len(config["order"]) == 3

    def test_sarima_fitted_configuration_has_seasonal_order(self) -> None:
        """SARIMA fitted_configuration contains seasonal order."""
        rng = np.random.default_rng(42)
        t = np.arange(60)
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        series = pd.Series(
            50 + seasonal + rng.normal(0, 1, 60),
            index=pd.date_range("2020-01-01", periods=60, freq="MS"),
        )
        result = fit_sarima(series, forecast_horizon=6, seasonal_period=12)
        config = result.fitted_configuration
        assert config["model"] == "SARIMA"
        assert config["seasonal_order"] is not None
        assert isinstance(config["seasonal_order"], list)
        assert len(config["seasonal_order"]) == 4

    def test_holt_winters_fitted_configuration_has_trend_and_seasonal(self) -> None:
        """Holt-Winters fitted_configuration contains trend and seasonal type."""
        rng = np.random.default_rng(42)
        t = np.arange(48)
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        series = pd.Series(
            50 + seasonal + rng.normal(0, 1, 48),
            index=pd.date_range("2020-01-01", periods=48, freq="MS"),
        )
        result = fit_holt_winters(series, forecast_horizon=6)
        config = result.fitted_configuration
        assert config["model"] == "Holt-Winters"
        assert config["trend"] == "add"
        assert config["seasonal"] in ("add", "mul", None)
        assert "seasonal_period" in config

    def test_ewma_fitted_configuration_has_alpha(self) -> None:
        """EWMA fitted_configuration contains the estimated alpha."""
        series = pd.Series(
            np.arange(1.0, 25.0),
            index=pd.date_range("2020-01-01", periods=24, freq="MS"),
        )
        result = fit_ewma(series, forecast_horizon=6)
        config = result.fitted_configuration
        assert config["model"] == "EWMA"
        assert config["alpha"] is not None
        assert 0.0 < config["alpha"] < 1.0
        assert config["estimated"] is True

    def test_arima_refit_preserves_intercept_config(self) -> None:
        """ARIMA refit preserves the with_intercept configuration."""
        rng = np.random.default_rng(42)
        series = pd.Series(
            100 + rng.normal(0, 2, 48),
            index=pd.date_range("2020-01-01", periods=48, freq="MS"),
        )
        result = fit_arima(series, forecast_horizon=6)
        config = result.fitted_configuration
        # with_intercept should be present (either True, False, or None)
        assert "with_intercept" in config
        assert "refit_order" in config
        assert config["refit_order"] == config["order"]
