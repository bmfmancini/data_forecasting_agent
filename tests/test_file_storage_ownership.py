"""Regression tests for durable, owner-scoped uploaded file storage."""

from __future__ import annotations

import os
import sys
from typing import Any

import pandas as pd
import pytest

_backend_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data_forecaster", "backend")
)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import core.config as settings  # noqa: E402
import services.file_service as file_service  # noqa: E402
import services.job_service as job_service  # noqa: E402
from auth.api_key_db import create_first_user, delete_api_user  # noqa: E402
from core.database import init_database  # noqa: E402


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
