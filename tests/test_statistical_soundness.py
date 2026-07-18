"""Reference calculations and invariants for statistical forecasting code.

These tests are intentionally independent of report formatting and LLM code.
They form the blocking statistical-soundness gate in CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from forecasting import sarima_model
from forecasting.contracts import (
    ForecastAdapterResult,
    ForecastFitStatus,
)
from forecasting.ewma_model import fit_ewma
from forecasting.intervals import simulate_ets_paths
from forecasting.metrics import calculate_forecast_metrics
from forecasting.residual_diagnostics import (
    calibrate_interval_width,
    interval_nonconformity_scores,
)


def test_all_point_metrics_match_hand_calculation() -> None:
    """Every point metric agrees with a small independently derived example."""
    actual = np.array([10.0, 20.0])
    predicted = np.array([8.0, 22.0])
    training = np.array([2.0, 4.0, 8.0, 14.0])

    metrics = calculate_forecast_metrics(
        actual,
        predicted,
        training=training,
        mase_period=1,
    )

    assert metrics.rmse == pytest.approx(2.0)
    assert metrics.mae == pytest.approx(2.0)
    assert metrics.mape == pytest.approx(15.0)
    assert metrics.wape == pytest.approx(4.0 / 30.0)
    assert metrics.mase == pytest.approx(0.5)
    assert metrics.rmsse == pytest.approx(np.sqrt(4.0 / (56.0 / 3.0)))
    assert metrics.smape == pytest.approx(200.0 * ((2.0 / 18.0 + 2.0 / 42.0) / 2.0))


def test_mase_uses_configured_seasonal_naive_lag() -> None:
    """Seasonal MASE scales errors against lag-m rather than lag-one changes."""
    training = np.array([10.0, 20.0, 11.0, 21.0, 12.0, 22.0])
    metrics = calculate_forecast_metrics(
        np.array([13.0, 23.0]),
        np.array([11.0, 21.0]),
        training=training,
        mase_period=2,
    )

    assert metrics.mase == pytest.approx(2.0)
    assert metrics.rmsse == pytest.approx(2.0)


def test_smape_counts_joint_zero_as_zero_error() -> None:
    """A zero/zero pair contributes zero without changing the sample size."""
    metrics = calculate_forecast_metrics(
        np.array([0.0, 10.0]),
        np.array([0.0, 0.0]),
    )

    assert metrics.smape == pytest.approx(100.0)
    assert metrics.n_evaluated == 2


def test_partial_horizon_cannot_be_ranked_as_complete_evidence() -> None:
    """A model is not rankable when any requested prediction is non-finite."""
    metrics = calculate_forecast_metrics(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, np.nan, 3.0]),
    )
    result = ForecastAdapterResult(status=ForecastFitStatus.OK, metrics=metrics)

    assert metrics.n_evaluated == 2
    assert metrics.n_missing == 1
    assert result.is_rankable is False


def test_ses_simulation_matches_analytical_horizon_variance() -> None:
    """SES simulation propagates level uncertainty across future horizons."""
    rng = np.random.default_rng(31415)
    series = pd.Series(50.0 + rng.normal(0.0, 2.0, 500))
    alpha = 0.4
    fitted = SimpleExpSmoothing(
        series,
        initialization_method="estimated",
    ).fit(smoothing_level=alpha, optimized=False)

    paths = simulate_ets_paths(fitted, 5, repetitions=20_000, seed=2718)
    variances = np.var(paths, axis=1, ddof=1)
    observed_ratio = variances / variances[0]
    expected_ratio = 1.0 + alpha**2 * np.arange(5, dtype=float)

    assert observed_ratio == pytest.approx(expected_ratio, rel=0.05, abs=0.05)


def test_ets_simulation_is_seed_reproducible() -> None:
    """Bootstrap paths are deterministic for audit and regression testing."""
    series = pd.Series([9.0, 11.0, 10.0, 12.0, 9.5, 10.5, 11.5, 10.0])
    fitted = SimpleExpSmoothing(series, initialization_method="estimated").fit(
        smoothing_level=0.3,
        optimized=False,
    )

    first = simulate_ets_paths(fitted, 3, repetitions=50, seed=7)
    second = simulate_ets_paths(fitted, 3, repetitions=50, seed=7)

    np.testing.assert_array_equal(first, second)


def test_conformal_scores_measure_only_interval_misses() -> None:
    """Conformal scores equal distance beyond the violated interval bound."""
    scores = interval_nonconformity_scores(
        actual=[0.0, 4.0, 12.0],
        lower=[-1.0, 5.0, 8.0],
        upper=[1.0, 7.0, 10.0],
    )

    assert scores == pytest.approx([0.0, 1.0, 2.0])


def test_conformal_calibration_uses_finite_sample_higher_quantile() -> None:
    """A tail miss expands both future bounds by its held-out miss distance."""
    scores = [0.0] * 19 + [3.0]
    lower, upper = calibrate_interval_width(
        [8.0, 9.0],
        [12.0, 13.0],
        calibration_scores=scores,
        nominal_coverage=0.95,
    )

    assert lower == pytest.approx([5.0, 6.0])
    assert upper == pytest.approx([15.0, 16.0])


def test_conformal_quantile_uses_exact_one_based_finite_sample_rank() -> None:
    """The conformal rank is ceil((n + 1) coverage), without an off-by-one."""
    lower, upper = calibrate_interval_width(
        [0.0],
        [1.0],
        calibration_scores=np.arange(1.0, 101.0),
        nominal_coverage=0.9,
    )

    assert lower == pytest.approx([-91.0])
    assert upper == pytest.approx([92.0])


def test_conformal_calibration_refuses_tiny_samples() -> None:
    """Sparse coverage evidence must not be presented as calibration."""
    lower, upper = calibrate_interval_width(
        [8.0],
        [12.0],
        calibration_scores=[4.0] * 9,
    )

    assert lower == [8.0]
    assert upper == [12.0]


@pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.01])
def test_ewma_rejects_invalid_smoothing_parameters(alpha: float) -> None:
    """Invalid alpha values return typed non-estimable evidence."""
    result = fit_ewma(pd.Series(np.arange(1.0, 21.0)), 3, alpha=alpha)

    assert result.status == ForecastFitStatus.NOT_ESTIMABLE
    assert result.forecast == []
    assert result.failure_reason is not None


@pytest.mark.parametrize("period", [0, -1])
def test_sarima_rejects_invalid_seasonal_period(period: int) -> None:
    """Invalid seasonal periods never propagate library exceptions."""
    result = sarima_model.fit_sarima(
        pd.Series(np.arange(30.0)),
        forecast_horizon=3,
        seasonal_period=period,
    )

    assert result.status == ForecastFitStatus.NOT_ESTIMABLE
    assert result.failure_reason is not None


def test_sarima_seasonal_sufficiency_uses_training_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Held-out observations cannot be counted as seasonal training cycles."""
    observed: dict[str, object] = {}

    def capture_auto_arima(_train: pd.Series, **kwargs: object) -> None:
        observed.update(kwargs)
        raise ValueError("stop after configuration capture")

    monkeypatch.setattr(sarima_model.pm, "auto_arima", capture_auto_arima)
    result = sarima_model.fit_sarima(
        pd.Series(np.arange(24.0)),
        forecast_horizon=6,
        seasonal_period=12,
    )

    assert observed["seasonal"] is False
    assert observed["m"] == 1
    assert result.status == ForecastFitStatus.NOT_ESTIMABLE
    assert result.fitted_configuration["fallback"] == "persistence"


def test_sarima_empty_series_returns_typed_fallback() -> None:
    """Empty input returns a complete typed result rather than raising."""
    result = sarima_model.fit_sarima(
        pd.Series(dtype=float),
        forecast_horizon=3,
        seasonal_period=12,
    )

    assert result.status == ForecastFitStatus.NOT_ESTIMABLE
    assert result.forecast == [0.0, 0.0, 0.0]
    assert result.is_fallback is True
