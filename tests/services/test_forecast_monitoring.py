"""Forecast monitoring must preserve issued values and enforce job ownership."""

import pytest

from core import config as settings
from core.database import init_database, transaction
from services.forecast_monitoring import monitor_forecast, save_snapshot
from services.job_service import _insert_job


@pytest.fixture
def issued_forecast(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "BACKEND_DB_PATH", str(tmp_path / "monitoring.db"))
    init_database()
    with transaction() as connection:
        connection.execute(
            "INSERT INTO api_users (id, username, api_key_hash) VALUES (1, 'owner', 'test')"
        )
    _insert_job(
        "job",
        {
            "file_id": "file",
            "date_col": "date",
            "value_col": "value",
            "forecast_horizon": 2,
            "forced_model": None,
            "user_prompt": None,
            "preflight_options": {},
        },
        1,
        1,
        "owner",
        False,
    )
    forecast = {
        "model_used": "EWMA",
        "forecast_dates": ["2025-01-01", "2025-01-02"],
        "forecast": [10.0, 10.0],
        "lower_ci": [8.0, 8.0],
        "upper_ci": [12.0, 12.0],
        "validation_design": {"monitoring_baselines": {"Naive": [8.0, 8.0]}},
    }
    save_snapshot("job", forecast)
    return forecast


def test_actuals_are_persisted_without_refitting(issued_forecast):
    observed = monitor_forecast("job", {"id": 1}, {"2025-01-01": 11.0})
    assert observed["n_observed"] == 1
    assert observed["n_pending"] == 1
    assert observed["metrics"]["mae"] == 1.0
    assert observed["interval_coverage"] == 1.0
    assert observed["skill_scores"]["mae_skill_vs_Naive"] == pytest.approx(2 / 3)
    assert monitor_forecast("job", {"id": 1}) == observed


def test_forecast_snapshot_cannot_be_replaced_after_actuals(issued_forecast):
    save_snapshot("job", {**issued_forecast, "forecast": [99.0, 99.0]})
    result = monitor_forecast("job", {"id": 1}, {"2025-01-01": 11.0})
    assert result["by_horizon"]["1"]["forecast"] == 10.0


def test_another_owner_cannot_read_or_record_actuals(issued_forecast):
    with pytest.raises(LookupError):
        monitor_forecast("job", {"id": 2}, {"2025-01-01": 99.0})
    assert monitor_forecast("job", {"id": 1})["n_observed"] == 0


def test_invalid_batch_does_not_partially_write_actuals(issued_forecast):
    with pytest.raises(ValueError):
        monitor_forecast("job", {"id": 1}, {"2025-01-01": 11.0, "2024-01-01": 9.0})
    assert monitor_forecast("job", {"id": 1})["n_observed"] == 0


def test_retention_deletes_forecast_and_actuals_together(issued_forecast):
    monitor_forecast("job", {"id": 1}, {"2025-01-01": 11.0})
    with transaction() as connection:
        connection.execute("DELETE FROM forecast_jobs WHERE job_id = 'job'")
        assert (
            connection.execute("SELECT COUNT(*) FROM forecast_snapshots").fetchone()[0]
            == 0
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM forecast_actuals").fetchone()[0]
            == 0
        )


def test_monitoring_api_records_scores_and_checks_ownership(
    issued_forecast, monkeypatch
):
    from fastapi.testclient import TestClient
    from auth.dependency import require_api_key
    from main import app

    monkeypatch.setitem(app.dependency_overrides, require_api_key, lambda: {"id": 1})
    client = TestClient(app)
    response = client.post("/jobs/job/actuals", json={"2025-01-01": 11.0})
    assert response.status_code == 200
    assert response.json()["metrics"]["mae"] == 1.0
    assert client.get("/jobs/job/monitoring").json()["n_observed"] == 1
    assert client.post("/jobs/job/actuals", json={"invalid": 12.0}).status_code == 422
    monkeypatch.setitem(app.dependency_overrides, require_api_key, lambda: {"id": 2})
    assert client.get("/jobs/job/monitoring").status_code == 404
    assert (
        client.post("/jobs/job/actuals", json={"2025-01-02": 12.0}).status_code == 404
    )


def test_comparison_requires_same_series_and_ownership(issued_forecast):
    from services.forecast_monitoring import monitor_forecasts

    result = monitor_forecasts(["job"], {"id": 1})
    assert result["alert_status"] == "insufficient_evidence"
    assert result["alerts"] == []
    with pytest.raises(LookupError):
        monitor_forecasts(["job"], {"id": 2})
    with pytest.raises(ValueError):
        monitor_forecasts(["job", "job"], {"id": 1})


def test_comparison_detects_persistent_bias_and_declining_coverage(issued_forecast):
    import pandas as pd
    from services.forecast_monitoring import monitor_forecasts

    ids = []
    for i in range(10):
        job_id = f"run-{i}"
        ids.append(job_id)
        _insert_job(
            job_id,
            {
                "file_id": "file",
                "date_col": "date",
                "value_col": "value",
                "forecast_horizon": 4,
                "forced_model": None,
                "user_prompt": None,
                "preflight_options": {},
            },
            1,
            1,
            "owner",
            False,
        )
        dates = (
            pd.date_range("2025-01-01", periods=40)[4 * i : 4 * i + 4]
            .strftime("%Y-%m-%d")
            .tolist()
        )
        snapshot = {
            **issued_forecast,
            "forecast_dates": dates,
            "forecast": [10.0] * 4,
            "lower_ci": [8.0] * 4,
            "upper_ci": [12.0] * 4,
            "validation_design": {"monitoring_baselines": {"Naive": [13.0] * 4}},
        }
        save_snapshot(job_id, snapshot)
        monitor_forecast(
            job_id, {"id": 1}, dict.fromkeys(dates, 10.0 if i < 5 else 15.0)
        )
    result = monitor_forecasts(ids[::-1], {"id": 1})
    assert result["alert_status"] == "evaluated"
    assert {a["kind"] for a in result["alerts"]} == {
        "persistent_bias",
        "declining_interval_coverage",
        "declining_baseline_skill",
    }
    assert result["recent"]["n_observed"] == 20
    with transaction() as connection:
        connection.execute(
            "UPDATE forecast_jobs SET value_col='different' WHERE job_id=?", (ids[0],)
        )
    with pytest.raises(ValueError, match="same series"):
        monitor_forecasts(ids, {"id": 1})


def test_comparison_api_is_read_only_and_rejects_unowned_jobs(
    issued_forecast, monkeypatch
):
    from fastapi.testclient import TestClient
    from auth.dependency import require_api_key
    from main import app

    monkeypatch.setitem(app.dependency_overrides, require_api_key, lambda: {"id": 1})
    client = TestClient(app)
    assert client.post("/monitoring/compare", json=["job"]).status_code == 200
    assert client.post("/monitoring/compare", json=[]).status_code == 422
    monkeypatch.setitem(app.dependency_overrides, require_api_key, lambda: {"id": 2})
    assert client.post("/monitoring/compare", json=["job"]).status_code == 404


def test_single_step_monitoring_accumulates_enough_dates(issued_forecast):
    import pandas as pd
    from services.forecast_monitoring import monitor_forecasts

    ids = []
    for i, date in enumerate(pd.date_range("2025-01-01", periods=40)):
        job_id = f"daily-{i}"
        ids.append(job_id)
        _insert_job(
            job_id,
            {
                "file_id": "file",
                "date_col": "date",
                "value_col": "value",
                "forecast_horizon": 1,
                "forced_model": None,
                "user_prompt": None,
                "preflight_options": {},
            },
            1,
            1,
            "owner",
            False,
        )
        stamp = date.isoformat()
        save_snapshot(
            job_id,
            {
                **issued_forecast,
                "forecast_dates": [stamp],
                "forecast": [10.0],
                "lower_ci": [8.0],
                "upper_ci": [12.0],
                "validation_design": {"frequency": "D"},
            },
        )
        monitor_forecast(job_id, {"id": 1}, {stamp: 11.0})
    assert (
        monitor_forecasts(ids[:10], {"id": 1})["alert_status"]
        == "insufficient_evidence"
    )
    result = monitor_forecasts(ids, {"id": 1})
    assert result["alert_status"] == "evaluated"
    assert result["recent"]["n_vintages"] == 20
    assert result["earlier"]["n_vintages"] == 20


def test_duplicate_origins_cannot_inflate_monitoring_evidence(issued_forecast):
    from services.forecast_monitoring import monitor_forecasts

    _insert_job(
        "duplicate",
        {
            "file_id": "file",
            "date_col": "date",
            "value_col": "value",
            "forecast_horizon": 2,
            "forced_model": None,
            "user_prompt": None,
            "preflight_options": {},
        },
        1,
        1,
        "owner",
        False,
    )
    save_snapshot("duplicate", issued_forecast)
    result = monitor_forecasts(["job", "duplicate"], {"id": 1})
    assert result["duplicate_origins_excluded"] == 1
    assert result["summary"]["n_vintages"] == 1
    assert result["alert_status"] == "insufficient_evidence"


def test_comparison_preserves_the_issued_quantile(issued_forecast):
    from services.forecast_monitoring import monitor_forecasts

    _insert_job(
        "quantile",
        {
            "file_id": "file",
            "date_col": "date",
            "value_col": "value",
            "forecast_horizon": 2,
            "forced_model": None,
            "user_prompt": None,
            "preflight_options": {},
        },
        1,
        1,
        "owner",
        False,
    )
    save_snapshot(
        "quantile",
        {**issued_forecast, "validation_design": {"decision_loss": {"quantile": 0.8}}},
    )
    monitor_forecast("quantile", {"id": 1}, {"2025-01-01": 15.0})
    result = monitor_forecasts(["quantile"], {"id": 1})
    assert result["summary"]["metrics"]["pinball"] == pytest.approx(4.0)
    with pytest.raises(ValueError, match="same forecast quantile"):
        monitor_forecasts(["job", "quantile"], {"id": 1})


def test_comparison_rejects_mixed_declared_and_undeclared_frequency(issued_forecast):
    from services.forecast_monitoring import monitor_forecasts

    # A one-step vintage cannot have its frequency inferred (<3 dates), so the
    # first job resolves to None even though its dates are daily.
    _insert_job(
        "undeclared",
        {
            "file_id": "file",
            "date_col": "date",
            "value_col": "value",
            "forecast_horizon": 1,
            "forced_model": None,
            "user_prompt": None,
            "preflight_options": {},
        },
        1,
        1,
        "owner",
        False,
    )
    save_snapshot(
        "undeclared",
        {
            **issued_forecast,
            "forecast_dates": ["2025-01-01"],
            "forecast": [10.0],
            "lower_ci": [8.0],
            "upper_ci": [12.0],
            "validation_design": {"monitoring_baselines": {"Naive": [8.0]}},
        },
    )
    monitor_forecast("undeclared", {"id": 1}, {"2025-01-01": 11.0})

    # The declared daily vintage must not pass against the None reference.
    _insert_job(
        "declared",
        {
            "file_id": "file",
            "date_col": "date",
            "value_col": "value",
            "forecast_horizon": 1,
            "forced_model": None,
            "user_prompt": None,
            "preflight_options": {},
        },
        1,
        1,
        "owner",
        False,
    )
    save_snapshot(
        "declared",
        {
            **issued_forecast,
            "forecast_dates": ["2025-01-02"],
            "forecast": [10.0],
            "lower_ci": [8.0],
            "upper_ci": [12.0],
            "validation_design": {
                "frequency": "D",
                "monitoring_baselines": {"Naive": [8.0]},
            },
        },
    )
    monitor_forecast("declared", {"id": 1}, {"2025-01-02": 11.0})

    with pytest.raises(ValueError, match="same forecast frequency"):
        monitor_forecasts(["undeclared", "declared"], {"id": 1})
    with pytest.raises(ValueError, match="same forecast frequency"):
        monitor_forecasts(["declared", "undeclared"], {"id": 1})
