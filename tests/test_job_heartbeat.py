"""Regression tests for durable long-running forecast activity signals."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def job_service(tmp_path, monkeypatch):
    """Load the job service against an isolated initialized database."""
    import importlib
    import core.config
    import core.database
    import services.job_service

    monkeypatch.setenv("BACKEND_DB_PATH", str(tmp_path / "heartbeat.db"))
    importlib.reload(core.config)
    importlib.reload(core.database)
    core.database.init_database()
    importlib.reload(services.job_service)
    return services.job_service


def _insert_job(job_service, status: str = "pending") -> None:
    from core.database import get_connection

    connection = get_connection()
    try:
        connection.execute(
            """
            INSERT INTO forecast_jobs (
                job_id, file_id, date_col, value_col, forecast_horizon,
                preflight_options, status, step
            ) VALUES ('heartbeat-job', 'file-1', 'date', 'value', 12, '{}', ?, 'Queued')
            """,
            (status,),
        )
        connection.commit()
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("heartbeat_at", "expected"),
    [
        ("2026-07-18 11:59:50", "active"),
        ("2026-07-18 11:59:10", "delayed"),
        ("2026-07-18 11:58:20", "stale"),
    ],
)
def test_liveness_classification_uses_heartbeat_age(
    job_service, heartbeat_at: str, expected: str
) -> None:
    record = {
        "status": "running",
        "queued_at": "2026-07-18 11:55:00",
        "started_at": "2026-07-18 11:56:00",
        "heartbeat_at": heartbeat_at,
        "progress_updated_at": "2026-07-18 11:57:00",
        "completed_at": None,
    }

    enriched = job_service._with_activity_metadata(
        record, now=datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    )

    assert enriched["liveness"] == expected
    assert enriched["elapsed_seconds"] == 240
    assert enriched["stage_age_seconds"] == 180


def test_pending_and_terminal_jobs_have_explicit_liveness(job_service) -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    pending = job_service._with_activity_metadata(
        {"status": "pending", "queued_at": "2026-07-18 11:59:00"}, now=now
    )
    done = job_service._with_activity_metadata(
        {
            "status": "done",
            "queued_at": "2026-07-18 11:00:00",
            "started_at": "2026-07-18 11:10:00",
            "completed_at": "2026-07-18 11:40:00",
        },
        now=now,
    )

    assert pending["liveness"] == "queued"
    assert done["liveness"] == "terminal"
    assert done["elapsed_seconds"] == 1800


def test_claim_and_progress_refresh_activity_timestamps(job_service) -> None:
    _insert_job(job_service)

    claimed = job_service._claim_job("heartbeat-job")
    assert claimed is not None
    running = job_service._job_record("heartbeat-job")
    assert running["heartbeat_at"] is not None
    assert running["progress_updated_at"] is not None

    job_service._update_job_progress("heartbeat-job", 60, "Evaluating ARIMA fold 1")
    updated = job_service.get_job_status_only("heartbeat-job")
    assert updated["progress"] == 60
    assert updated["step"] == "Evaluating ARIMA fold 1"
    assert updated["liveness"] == "active"


def test_parent_heartbeat_repeats_until_cancelled(job_service, monkeypatch) -> None:
    touches: list[str] = []
    monkeypatch.setattr(job_service.settings, "JOB_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(job_service, "_touch_job_heartbeat", touches.append)

    async def exercise() -> None:
        task = asyncio.create_task(job_service._maintain_job_heartbeat("job-1"))
        await asyncio.sleep(0.035)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    assert len(touches) >= 2
    assert set(touches) == {"job-1"}


def test_candidate_and_fold_callbacks_do_not_change_evaluation() -> None:
    from forecasting.backtesting import (
        BacktestConfig,
        FoldPrediction,
        evaluate_candidates,
    )

    activity: list[str] = []
    series = pd.Series(np.arange(1, 17, dtype=float))

    def naive(train, fold):
        return FoldPrediction(
            predictions=np.repeat(float(train.iloc[-1]), fold.horizon)
        )

    result = evaluate_candidates(
        series,
        {"Naive": naive},
        config=BacktestConfig(
            initial_train_size=8,
            horizon=2,
            max_origins=2,
            final_test_size=2,
        ),
        activity_callback=activity.append,
    )

    assert "Naive" in result
    assert any("candidate 1 of 1" in message for message in activity)
    assert any("validation fold" in message for message in activity)
    assert any("final holdout" in message for message in activity)
