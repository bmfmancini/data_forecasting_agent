"""Regression tests for durable, owner-scoped uploaded file storage."""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
import pytest

import core.config as settings
import services.file_service as file_service
import services.job_service as job_service
from auth.api_key_db import create_first_user, delete_api_user
from core.database import get_connection, init_database
from main import preflight_check
from schemas import AnalyzeRequest


@pytest.fixture
def storage_dir(tmp_path: Any, monkeypatch: Any) -> str:
    """Configure an isolated durable upload store."""
    path = str(tmp_path / "files")
    monkeypatch.setattr(settings, "FILE_STORAGE_DIR", path)
    monkeypatch.setattr(settings, "BACKEND_DB_PATH", str(tmp_path / "backend.db"))
    monkeypatch.setattr(settings, "MAX_INMEMORY_FILES", 50)
    monkeypatch.setattr(file_service, "MAX_FILES", 50)
    init_database()
    file_service.init_storage()
    return path


def test_files_are_owner_scoped_and_survive_index_reload(storage_dir: str) -> None:
    """A restart must retain metadata and reject another regular user."""
    owner = create_first_user("owner", "test-owner-key")
    owner_id = int(owner["id"])
    df = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=3), "value": [1, 2, 3]}
    )
    file_id = file_service.store_file(
        df, "date", "value", "D", "forecast.csv", owner_id=owner_id
    )

    assert file_service.get_file(file_id, {"id": 202, "is_admin": False}) is None
    assert (
        file_service.get_file(file_id, {"id": owner_id, "is_admin": False}) is not None
    )
    assert file_service.get_file(file_id, {"id": 202, "is_admin": True}) is not None

    # Simulate a process restart: only SQLite metadata and parquet remain.
    file_service._file_index.clear()  # pylint: disable=protected-access
    file_service.init_storage()
    restored = file_service.get_file(file_id, {"id": owner_id, "is_admin": False})
    assert restored is not None
    assert restored["filename"] == "forecast.csv"
    assert restored["df"]["value"].tolist() == [1, 2, 3]


def test_preflight_loads_the_value_column_selected_after_upload(
    storage_dir: str,
) -> None:
    """A dropdown override must be loaded instead of the detected value column."""
    del storage_dir
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=20, freq="D"),
            "Socrata Bounce Rate": range(20),
            "Combined Users": range(100, 120),
        }
    )
    file_id = file_service.store_file(
        frame,
        "Date",
        "Socrata Bounce Rate",
        "D",
        "traffic.csv",
    )

    stored = file_service.get_file(
        file_id,
        selected_date_col="Date",
        selected_value_col="Combined Users",
    )
    assert stored is not None
    assert stored["df"].columns.tolist() == ["Date", "Combined Users"]

    result = asyncio.run(
        preflight_check(
            AnalyzeRequest(
                file_id=file_id,
                forecast_horizon=3,
                date_col="Date",
                value_col="Combined Users",
            ),
            {},
        )
    )
    assert result.row_count == 20
    assert result.usable_observations == 20


def test_analysis_job_loads_its_selected_columns(monkeypatch: Any) -> None:
    """The background worker must use the same columns that passed preflight."""
    requested: dict[str, Any] = {}

    def fake_get_file(file_id: str, **kwargs: Any) -> None:
        requested.update({"file_id": file_id, **kwargs})
        return None

    monkeypatch.setattr(job_service, "get_file", fake_get_file)
    monkeypatch.setattr(job_service, "_set_job_error", lambda *_args: None)
    monkeypatch.setattr(job_service, "release_file", lambda *_args: None)

    asyncio.run(
        job_service._run_job(  # pylint: disable=protected-access
            "job-id",
            {
                "file_id": "file-id",
                "date_col": "Date",
                "value_col": "Combined Users",
            },
        )
    )

    assert requested == {
        "file_id": "file-id",
        "selected_date_col": "Date",
        "selected_value_col": "Combined Users",
    }


def test_jobs_are_owner_scoped(storage_dir: str) -> None:
    """Regular users cannot poll a different owner's job."""
    owner = create_first_user("owner", "test-owner-key")
    owner_id = int(owner["id"])
    df = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=3), "value": [1, 2, 3]}
    )
    file_id = file_service.store_file(
        df, "date", "value", "D", "forecast.csv", owner_id=owner_id
    )
    job_service._job_store.clear()  # pylint: disable=protected-access
    job_service.init_job_queue()
    job_id = job_service.create_job(
        file_id=file_id,
        date_col="date",
        value_col="value",
        forecast_horizon=3,
        forced_model=None,
        user_prompt=None,
        preflight_options=None,
        owner_id=owner_id,
    )

    assert job_service.get_job(job_id, {"id": 202, "is_admin": False}) is None
    assert job_service.get_job_status_only(job_id, {"id": owner_id, "is_admin": False})
    assert job_service.get_job(job_id, {"id": 202, "is_admin": True}) is not None


def test_non_admin_user_job_limit_queues_additional_work(storage_dir: str) -> None:
    """A user's second job remains pending until their first job completes."""
    owner = create_first_user("owner", "test-owner-key")
    owner_id = int(owner["id"])
    dataframe = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=3), "value": [1, 2, 3]}
    )
    first_file_id = file_service.store_file(
        dataframe, "date", "value", "D", "first.csv", owner_id=owner_id
    )
    second_file_id = file_service.store_file(
        dataframe, "date", "value", "D", "second.csv", owner_id=owner_id
    )
    job_service.init_job_queue()
    job_service.update_job_settings(1, 30, True)
    first_job_id = job_service.create_job(
        first_file_id,
        "date",
        "value",
        3,
        None,
        None,
        None,
        owner_id,
        application_user_id=42,
        application_username="forecast-user",
    )
    second_job_id = job_service.create_job(
        second_file_id,
        "date",
        "value",
        3,
        None,
        None,
        None,
        owner_id,
        application_user_id=42,
        application_username="forecast-user",
    )

    assert (
        job_service._claim_job(first_job_id) is not None
    )  # pylint: disable=protected-access
    assert (
        job_service._claim_job(second_job_id) is None
    )  # pylint: disable=protected-access
    pending = job_service.get_job_status_only(second_job_id)
    assert pending is not None
    assert pending["status"] == "pending"


def test_clear_terminal_jobs_preserves_active_work(storage_dir: str) -> None:
    """Manual queue cleanup deletes terminal jobs without affecting pending work."""
    connection = get_connection()
    try:
        for job_id, status in (
            ("done-job", "done"),
            ("error-job", "error"),
            ("pending-job", "pending"),
        ):
            connection.execute(
                """
                INSERT INTO forecast_jobs (job_id, file_id, date_col, value_col,
                    forecast_horizon, status, step, completed_at)
                VALUES (?, 'file', 'date', 'value', 3, ?, 'test', datetime('now'))
                """,
                (job_id, status),
            )
        connection.commit()
    finally:
        connection.close()

    assert job_service.clear_terminal_jobs() == 2
    jobs = {job["job_id"]: job for job in job_service.list_recent_jobs()}
    assert "done-job" not in jobs
    assert "error-job" not in jobs
    assert jobs["pending-job"]["status"] == "pending"


def test_user_with_uploaded_files_cannot_be_deleted(storage_dir: str) -> None:
    """Foreign-key ownership must be enforced before user deletion."""
    owner = create_first_user("owner", "test-owner-key")
    df = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=3), "value": [1, 2, 3]}
    )
    file_service.store_file(
        df, "date", "value", "D", "forecast.csv", owner_id=int(owner["id"])
    )

    with pytest.raises(ValueError, match="owns uploaded files"):
        delete_api_user(int(owner["id"]))


def test_reserved_file_is_not_evicted(storage_dir: str, monkeypatch: Any) -> None:
    """A queued or running job's input cannot be evicted by a new upload."""
    owner = create_first_user("owner", "test-owner-key")
    owner_id = int(owner["id"])
    df = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=3), "value": [1, 2, 3]}
    )
    file_id = file_service.store_file(
        df, "date", "value", "D", "first.csv", owner_id=owner_id
    )
    assert file_service.reserve_file(file_id)
    monkeypatch.setattr(file_service, "MAX_FILES", 1)

    with pytest.raises(RuntimeError, match="currently used"):
        file_service.store_file(
            df, "date", "value", "D", "second.csv", owner_id=owner_id
        )

    file_service.release_file(file_id)
