"""Regression tests for deterministic synthetic forecast fixtures.

These tests verify that every fixture is deterministic (same seed), has the
expected length, and exercises the intended edge case. They also confirm
that the four model adapters handle each fixture without crashing and
return a :class:`ForecastAdapterResult`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting.contracts import ForecastAdapterResult, ForecastFitStatus
from forecasting.fixtures import ALL_FIXTURES
from forecasting.arima_model import fit_arima
from forecasting.ewma_model import fit_ewma
from forecasting.holt_winters import fit_holt_winters
from forecasting.sarima_model import fit_sarima

FORECAST_HORIZON = 6


class TestFixtureDeterminism:
    """Verify fixtures are reproducible across calls."""

    @pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
    def test_fixture_is_deterministic(self, name: str) -> None:
        """Calling a fixture twice produces identical values."""
        fn = ALL_FIXTURES[name]
        first = fn()
        second = fn()
        np.testing.assert_array_equal(first.values, second.values)

    @pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
    def test_fixture_returns_named_series(self, name: str) -> None:
        """Every fixture returns a named pd.Series."""
        fn = ALL_FIXTURES[name]
        result = fn()
        assert isinstance(result, pd.Series)
        assert result.name == name


class TestFixtureProperties:
    """Verify each fixture has the expected statistical properties."""

    def test_constant_series_has_zero_variance(self) -> None:
        from forecasting.fixtures import constant_series

        s = constant_series()
        assert s.std() == 0.0
        assert (s == 100.0).all()

    def test_near_constant_series_has_low_variance(self) -> None:
        from forecasting.fixtures import near_constant_series

        s = near_constant_series()
        assert s.std() < 0.1

    def test_stationary_ar_is_not_trending(self) -> None:
        from forecasting.fixtures import stationary_ar_series

        s = stationary_ar_series(n=200)
        # Mean should be near zero for a zero-mean AR(1)
        assert abs(s.mean()) < 2.0

    def test_random_walk_is_non_stationary(self) -> None:
        from forecasting.fixtures import random_walk_series

        s = random_walk_series(n=100)
        # A random walk's variance grows; the range should be wide
        assert s.max() - s.min() > 5.0

    def test_additive_seasonal_has_periodic_autocorrelation(self) -> None:
        from forecasting.fixtures import additive_seasonal_series

        s = additive_seasonal_series(n=48, period=12)
        # Autocorrelation at lag 12 should be high
        acf_12 = s.autocorr(lag=12)
        assert acf_12 > 0.5

    def test_multiplicative_seasonal_is_positive(self) -> None:
        from forecasting.fixtures import multiplicative_seasonal_series

        s = multiplicative_seasonal_series()
        assert (s > 0).all()

    def test_trend_series_has_significant_slope(self) -> None:
        from forecasting.fixtures import trend_series

        s = trend_series(n=48, slope=2.0)
        # Linear regression slope should be close to 2.0
        t = np.arange(len(s))
        slope = np.polyfit(t, s.values, 1)[0]
        assert slope == pytest.approx(2.0, abs=0.5)

    def test_zeros_series_is_all_zero(self) -> None:
        from forecasting.fixtures import zeros_series

        s = zeros_series()
        assert (s == 0.0).all()

    def test_negative_values_has_negatives(self) -> None:
        from forecasting.fixtures import negative_values_series

        s = negative_values_series()
        assert (s < 0).any()

    def test_missing_timestamps_has_gaps(self) -> None:
        from forecasting.fixtures import missing_timestamps_series

        s = missing_timestamps_series(n=36, missing_count=5)
        assert len(s) == 31  # 36 - 5

    def test_duplicate_timestamps_has_dupes(self) -> None:
        from forecasting.fixtures import duplicate_timestamps_series

        s = duplicate_timestamps_series(n=30)
        assert s.index.duplicated().any()

    def test_short_seasonal_is_below_two_cycles(self) -> None:
        from forecasting.fixtures import short_seasonal_series

        s = short_seasonal_series(period=12)
        assert len(s) < 2 * 12

    def test_isolated_anomalies_has_spikes(self) -> None:
        from forecasting.fixtures import isolated_anomalies_series

        s = isolated_anomalies_series()
        # The anomalies should be detectable as outliers
        median = s.median()
        mad = np.median(np.abs(s - median))
        # At least 2 points should be > 5 MADs from the median
        outliers = (np.abs(s - median) > 5 * mad).sum()
        assert outliers >= 2

    def test_structural_break_has_level_shift(self) -> None:
        from forecasting.fixtures import structural_break_series

        s = structural_break_series(n=48, break_point=24)
        before = s.iloc[:24].mean()
        after = s.iloc[24:].mean()
        assert abs(after - before) > 30.0


class TestAdapterFixtureSurvival:
    """Every adapter must return a ForecastAdapterResult for every fixture.

    This is a survival test — it verifies no fixture causes an unhandled
    exception. Metric correctness is tested separately.
    """

    @pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
    def test_arima_survives_fixture(self, name: str) -> None:
        fn = ALL_FIXTURES[name]
        series = fn()
        result = fit_arima(series, FORECAST_HORIZON)
        assert isinstance(result, ForecastAdapterResult)
        assert len(result.forecast) == FORECAST_HORIZON

    @pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
    def test_ewma_survives_fixture(self, name: str) -> None:
        fn = ALL_FIXTURES[name]
        series = fn()
        result = fit_ewma(series, FORECAST_HORIZON)
        assert isinstance(result, ForecastAdapterResult)
        assert len(result.forecast) == FORECAST_HORIZON

    @pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
    def test_holt_winters_survives_fixture(self, name: str) -> None:
        fn = ALL_FIXTURES[name]
        series = fn()
        result = fit_holt_winters(series, FORECAST_HORIZON)
        assert isinstance(result, ForecastAdapterResult)
        assert len(result.forecast) == FORECAST_HORIZON

    @pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
    def test_sarima_survives_fixture(self, name: str) -> None:
        fn = ALL_FIXTURES[name]
        series = fn()
        result = fit_sarima(series, FORECAST_HORIZON, seasonal_period=12)
        assert isinstance(result, ForecastAdapterResult)
        assert len(result.forecast) == FORECAST_HORIZON
