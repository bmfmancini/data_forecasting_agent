"""Unit tests for the Prophet forecast model adapter.

Prophet is a heavy optional dependency that is not installed in CI, so every
test patches ``forecasting.prophet_model.import_prophet`` with a lightweight
fake ``prophet`` module.  The fake Prophet class records the frames it is
handed and returns deterministic ``yhat``/``yhat_lower``/``yhat_upper``
columns so the adapter logic can be exercised end-to-end without cmdstan.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from forecasting import prophet_model


class _FakeProphet:
    """Deterministic stand-in for ``prophet.Prophet`` used by the adapter.

    Records the last history frame it was fit on and produces forecasts whose
    ``yhat`` echoes the last observed value (plus a small linear drift) and
    whose intervals bracket ``yhat``.  ``fit`` behaviour is configurable so
    tests can simulate fit failures on the train split only.
    """

    # Class-level knobs the tests flip.
    fail_first_fit: bool = False
    _fit_call_count: int = 0

    def __init__(self) -> None:
        self._history: pd.DataFrame | None = None
        self._make_future_freqs: list[str | None] = []

    def fit(self, history: pd.DataFrame) -> "_FakeProphet":
        """Store the history frame; optionally fail for the first fit only.

        The holdout fit is the first ``Prophet().fit(...)`` call the adapter
        makes; the full-series fit is the second.  Failing only the first
        therefore simulates a holdout-only failure.
        """
        _FakeProphet._fit_call_count += 1
        if _FakeProphet.fail_first_fit and _FakeProphet._fit_call_count == 1:
            raise RuntimeError("train fit failed")
        self._history = history
        return self

    def make_future_dataframe(
        self,
        periods: int,
        freq: str | None = None,
        include_history: bool = True,
    ) -> pd.DataFrame:
        """Record the freq and build a future ``ds`` frame."""
        del include_history  # Unused by the fake.
        self._make_future_freqs.append(freq)
        last_ds = self._history["ds"].iloc[-1]
        future = pd.date_range(start=last_ds, periods=periods + 1, freq=freq or "D")[1:]
        return pd.DataFrame({"ds": future})

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        """Return deterministic yhat / yhat_lower / yhat_upper columns."""
        last_y = float(self._history["y"].iloc[-1])
        n = len(future)
        drift = np.arange(1, n + 1, dtype=float) * 0.5
        yhat = np.full(n, last_y, dtype=float) + drift
        return pd.DataFrame(
            {
                "ds": future["ds"],
                "yhat": yhat,
                "yhat_lower": yhat - 1.0,
                "yhat_upper": yhat + 1.0,
            }
        )


def _fake_prophet_module() -> SimpleNamespace:
    """Build a fake ``prophet`` module exposing ``Prophet``."""
    return SimpleNamespace(Prophet=_FakeProphet)


@pytest.fixture(autouse=True)
def _reset_fake() -> None:
    """Reset fake state between tests."""
    _FakeProphet.fail_first_fit = False
    _FakeProphet._fit_call_count = 0
    yield
    _FakeProphet.fail_first_fit = False
    _FakeProphet._fit_call_count = 0


def _patch_prophet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``import_prophet`` to return the fake prophet module."""
    monkeypatch.setattr(prophet_model, "import_prophet", lambda: _fake_prophet_module())


def _monthly_series(n: int = 36) -> pd.Series:
    """A deterministic monthly series with trend + seasonality."""
    idx = pd.date_range("2020-01-01", periods=n, freq="MS")
    values = np.arange(n, dtype=float) + 10 * np.sin(np.arange(n) * np.pi / 6)
    return pd.Series(values, index=idx)


class TestFitProphetHappyPath:
    """Happy-path behaviour of ``fit_prophet``."""

    def test_returns_full_contract_with_correct_lengths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The result dict matches the standard model contract."""
        _patch_prophet(monkeypatch)
        series = _monthly_series()

        result = prophet_model.fit_prophet(series, forecast_horizon=6, freq="MS")

        assert set(result) == {
            "forecast",
            "lower_ci",
            "upper_ci",
            "rmse",
            "mae",
            "mape",
        }
        assert len(result["forecast"]) == 6
        assert len(result["lower_ci"]) == 6
        assert len(result["upper_ci"]) == 6

    def test_intervals_bracket_forecast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``lower_ci`` <= ``forecast`` <= ``upper_ci`` for every step."""
        _patch_prophet(monkeypatch)
        result = prophet_model.fit_prophet(_monthly_series(), 4, freq="MS")

        for lo, fc, hi in zip(
            result["lower_ci"], result["forecast"], result["upper_ci"]
        ):
            assert lo <= fc <= hi

    def test_metrics_are_finite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Holdout metrics come back finite for a healthy series."""
        _patch_prophet(monkeypatch)
        result = prophet_model.fit_prophet(_monthly_series(), 6, freq="MS")

        assert np.isfinite(result["rmse"])
        assert np.isfinite(result["mae"])
        assert np.isfinite(result["mape"])


class TestFitProphetShortSeries:
    """Short-series fallback to a persistence forecast."""

    def test_single_point_returns_persistence_forecast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fewer than 2 points should never call Prophet."""
        _patch_prophet(monkeypatch)
        series = pd.Series([42.0])

        result = prophet_model.fit_prophet(series, forecast_horizon=3)

        assert result["forecast"] == [42.0, 42.0, 42.0]
        assert result["lower_ci"] == [42.0, 42.0, 42.0]
        assert result["upper_ci"] == [42.0, 42.0, 42.0]
        assert result["rmse"] == 0.0
        assert result["mae"] == 0.0
        assert result["mape"] == 0.0

    def test_empty_series_returns_zero_forecast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty series falls back to a zero persistence forecast."""
        _patch_prophet(monkeypatch)
        result = prophet_model.fit_prophet(pd.Series(dtype=float), forecast_horizon=2)

        assert result["forecast"] == [0.0, 0.0]
        assert result["rmse"] == 0.0


class TestFitProphetNonDatetimeIndex:
    """A non-datetime index is converted to a synthetic date range."""

    def test_range_index_still_forecasts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_prophet(monkeypatch)
        series = pd.Series(np.arange(20, dtype=float))

        result = prophet_model.fit_prophet(series, forecast_horizon=3)

        assert len(result["forecast"]) == 3
        assert all(np.isfinite(result["forecast"]))


class TestFitProphetTrainFailure:
    """Resilience when the holdout fit fails but the full fit succeeds."""

    def test_holdout_failure_zeroes_metrics_but_keeps_forecast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the train-split fit raises, metrics zero out, forecast survives."""
        _patch_prophet(monkeypatch)
        # First fit (holdout) raises; the second (full-series) succeeds.
        _FakeProphet.fail_first_fit = True

        result = prophet_model.fit_prophet(_monthly_series(), 4, freq="MS")

        assert result["rmse"] == 0.0
        assert result["mae"] == 0.0
        assert result["mape"] == 0.0
        assert len(result["forecast"]) == 4


class TestFitProphetImportError:
    """When Prophet is unavailable, the adapter surfaces the ImportError."""

    def test_raises_when_prophet_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise() -> None:
            raise ImportError("prophet not installed")

        monkeypatch.setattr(prophet_model, "import_prophet", _raise)

        with pytest.raises(ImportError):
            prophet_model.fit_prophet(_monthly_series(), forecast_horizon=3)


class TestFitProphetFreqThreading:
    """The supplied freq reaches the future-frame construction."""

    def test_freq_is_threaded_to_future_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The freq kwarg is forwarded to ``_future_frame``."""
        _patch_prophet(monkeypatch)
        seen: list[str | None] = []
        original = prophet_model._future_frame

        def _spy(history: pd.DataFrame, periods: int, freq: str | None) -> pd.DataFrame:
            seen.append(freq)
            return original(history, periods, freq)

        monkeypatch.setattr(prophet_model, "_future_frame", _spy)

        prophet_model.fit_prophet(_monthly_series(), forecast_horizon=3, freq="MS")

        # The holdout fit and the full-series fit each call _future_frame.
        assert seen[-1] == "MS"
