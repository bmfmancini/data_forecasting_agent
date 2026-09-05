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
