"""Universal memory-governance regression tests."""

from __future__ import annotations

import asyncio
import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.stats import theilslopes

from exceptions import ForecastResourceError
from forecasting.exact_statistics import exact_theilslopes_bounded
from forecasting import pmdarima_compat
from services import job_service
from services.job_service import MemoryAdmissionController
from services.pipeline_service import _prepare_pipeline_input
from utils.memory import (
    effective_memory_capacity_mb,
    estimate_arima_workspace_mb,
    estimate_bounded_theil_sen_resident_mb,
    estimate_forecast_memory,
    estimate_theil_sen_workspace_mb,
    memory_snapshot,
)


@pytest.mark.parametrize("size", [8, 9, 32, 101])
@pytest.mark.parametrize("shape", ["random", "ties", "trend", "constant"])
def test_bounded_theil_sen_matches_scipy(size: int, shape: str) -> None:
    rng = np.random.default_rng(size)
    values = rng.normal(size=size)
    if shape == "ties":
        values = np.round(values, 1)
    elif shape == "trend":
        values = np.arange(size) * 1.25 + rng.normal(scale=0.01, size=size)
    elif shape == "constant":
        values = np.ones(size)
    expected = theilslopes(values, np.arange(size, dtype=float), alpha=0.95)
    actual = exact_theilslopes_bounded(values, workspace_mb=64)
    np.testing.assert_allclose(
        [actual.slope, actual.intercept, actual.low_slope, actual.high_slope],
        expected,
        rtol=0,
        atol=0,
    )


def test_resource_estimates_are_dimension_based_and_monotonic() -> None:
    assert estimate_theil_sen_workspace_mb(10_000) > estimate_theil_sen_workspace_mb(
        1_000
    )
    assert estimate_bounded_theil_sen_resident_mb(
        10_000
    ) > estimate_bounded_theil_sen_resident_mb(1_000)
    assert estimate_arima_workspace_mb(10_000, 12) > estimate_arima_workspace_mb(
        1_000, 12
    )
    assert (
        estimate_forecast_memory(10_000, 12, horizon=24, candidate_count=12).total_mb
        > estimate_forecast_memory(1_000, 12, horizon=6, candidate_count=8).total_mb
    )
    assert effective_memory_capacity_mb(2048, 999) == 2048
    snapshot = memory_snapshot()
    assert snapshot.current_rss_mb > 0
    assert snapshot.peak_rss_mb > 0


def test_memory_admission_rejects_job_larger_than_capacity() -> None:
    controller = MemoryAdmissionController(512)
    with pytest.raises(ForecastResourceError, match="exceeding"):
        asyncio.run(controller.acquire(513, "unused-job"))


def test_memory_admission_queues_until_weight_is_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(job_service, "_is_cancel_requested", lambda _job_id: False)
    monkeypatch.setattr(job_service, "_update_job_progress", lambda *_args: None)
    controller = MemoryAdmissionController(600)

    async def scenario() -> None:
        await controller.acquire(400, "first")
        waiter = asyncio.create_task(controller.acquire(300, "second"))
        await asyncio.sleep(0)
        assert not waiter.done()
        await controller.release(400)
        await waiter
        assert controller.used_mb == 300
        await controller.release(300)

    asyncio.run(scenario())


def test_spawned_forecast_entry_point_is_serializable() -> None:
    assert pickle.loads(pickle.dumps(job_service._execute_pipeline_process)) is (
        job_service._execute_pipeline_process
    )


@pytest.mark.parametrize(
    ("seasonal_period", "seasonal_order"),
    [(1, (0, 0, 0, 0)), (12, (1, 0, 1, 12))],
)
def test_low_memory_arima_discovery_is_normally_refit(
    monkeypatch: pytest.MonkeyPatch,
    seasonal_period: int,
    seasonal_order: tuple[int, int, int, int],
) -> None:
    calls: dict[str, object] = {}

    class FakeModel:
        order = (3, 1, 2)
        with_intercept = True

        def __init__(self) -> None:
            self.seasonal_order = seasonal_order

        def predict(self, n_periods: int, return_conf_int: bool = False):
            values = np.arange(n_periods, dtype=float)
            intervals = np.column_stack((values - 1, values + 1))
            return (values, intervals) if return_conf_int else values

        def resid(self):
            return np.array([0.1, -0.1])

    class FakeArimaConstructor:
        def __init__(self, **kwargs: object) -> None:
            calls["refit_kwargs"] = kwargs

        def fit(self, series: pd.Series) -> FakeModel:
            calls["refit_size"] = len(series)
            return FakeModel()

    class FakePM:
        ARIMA = FakeArimaConstructor

        @staticmethod
        def auto_arima(series: pd.Series, **kwargs: object) -> FakeModel:
            calls["discovery_kwargs"] = kwargs
            return FakeModel()

    monkeypatch.setattr(pmdarima_compat, "import_pmdarima", lambda: FakePM)
    import core.config as config

    monkeypatch.setattr(config, "ARIMA_LOW_MEMORY_THRESHOLD_MB", 1)
    series = pd.Series(np.arange(40, dtype=float))
    model = pmdarima_compat.fit_auto_arima_memory_aware(
        series,
        seasonal_period=seasonal_period,
        seasonal=seasonal_period > 1,
    )
    assert calls["discovery_kwargs"]["low_memory"] is True
    assert calls["refit_kwargs"] == {
        "order": (3, 1, 2),
        "seasonal_order": seasonal_order,
        "with_intercept": True,
        "suppress_warnings": True,
    }
    assert calls["refit_size"] == len(series)
    forecasts, intervals = model.predict(3, return_conf_int=True)
    assert np.isfinite(forecasts).all()
    assert np.isfinite(intervals).all()
    assert np.isfinite(model.resid()).all()


@pytest.mark.parametrize(
    ("frequency", "dates", "expected_modeled"),
    [
        ("D", ["2024-01-01", "2024-01-02", "2024-01-03"], 3),
        ("D", ["2024-01-01", "2024-01-03"], 3),
        ("MS", ["2024-01-01", "2024-03-01"], 3),
        ("W", ["2024-01-07", "2024-01-21"], 3),
    ],
)
def test_timestamp_expansion_provenance(
    frequency: str, dates: list[str], expected_modeled: int
) -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "value": range(len(dates))})
    prepared = _prepare_pipeline_input(
        frame,
        "date",
        "value",
        frequency,
        {"frequency": frequency, "missing_strategy": "interpolate"},
    )
    provenance = prepared.preparation_provenance
    assert provenance["observed_count"] == len(dates)
    assert provenance["modeled_count"] == expected_modeled
    assert provenance["inserted_timestamp_count"] == expected_modeled - len(dates)
    assert provenance["expansion_ratio"] == expected_modeled / len(dates)
