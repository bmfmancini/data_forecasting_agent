"""Tests for durable result persistence and job limits."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "data_forecaster", "backend")
)

import pytest


@pytest.fixture()
def temp_env(monkeypatch):
    """Set up a temporary database and file storage directory."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    file_storage = os.path.join(tmpdir, "file_store")
    monkeypatch.setenv("BACKEND_DB_PATH", db_path)
    monkeypatch.setenv("FILE_STORAGE_DIR", file_storage)
    import importlib
    import core.config
    import core.database

    importlib.reload(core.config)
    importlib.reload(core.database)
    core.database.init_database()
    yield {"db_path": db_path, "file_storage": file_storage}


@pytest.fixture()
def job_service(temp_env):
    """Provide the job_service module with a fresh temporary environment."""
    import importlib
    import services.job_service

    importlib.reload(services.job_service)
    return services.job_service


class TestDurableResults:
    """Tests for _persist_result, _load_result, and _delete_result_file."""

    def test_persist_and_load_result(self, job_service):
        """Results can be persisted to disk and loaded back."""
        result = {"report": "test report", "forecast": {"model": "ARIMA"}}
        job_service._persist_result("job-1", result)
        loaded = job_service._load_result("job-1")
        assert loaded is not None
        assert loaded["report"] == "test report"

    def test_persist_result_fsyncs_file_and_directory(self, job_service, monkeypatch):
        """Durable writes sync both the result file and its parent directory."""
        synced: list[int] = []
        real_fsync = job_service.os.fsync

        def recording_fsync(fd: int) -> None:
            synced.append(fd)
            real_fsync(fd)

        monkeypatch.setattr(job_service.os, "fsync", recording_fsync)

        job_service._persist_result("job-1", {"report": "ready"})

        assert len(synced) == 2

    def test_load_result_returns_none_for_missing(self, job_service):
        """Loading a non-existent result returns None."""
        assert job_service._load_result("nonexistent") is None

    def test_delete_result_file(self, job_service):
        """Deleting a result file removes it from disk."""
        job_service._persist_result("job-1", {"data": "test"})
        job_service._delete_result_file("job-1")
        assert job_service._load_result("job-1") is None

    def test_delete_result_file_idempotent(self, job_service):
        """Deleting a non-existent result file does not raise."""
        job_service._delete_result_file("nonexistent")


class TestQueuedJobLimit:
    """Tests for the per-user queued job admission cap."""

    def _insert_pending_job(self, job_service, job_id: str, app_user_id: int) -> None:
        from core.database import get_connection

        conn = get_connection()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO api_users (id, username, api_key_hash) "
                "VALUES (1, 'test', 'hash')"
            )
            conn.execute(
                """
                INSERT INTO forecast_jobs (
                    job_id, backend_owner_id, application_user_id,
                    application_username, application_user_is_admin, file_id,
                    date_col, value_col, forecast_horizon, forced_model,
                    user_prompt, preflight_options, report_name,
                    source_filename, custom_settings_json, cancel_requested,
                    status, step
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    1,
                    app_user_id,
                    "user",
                    0,
                    "f",
                    "d",
                    "v",
                    12,
                    None,
                    None,
                    "{}",
                    "Test",
                    "t.csv",
                    "[]",
                    0,
                    "pending",
                    "Q",
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _insert_via_service(
        self,
        job_service,
        job_id: str,
        app_user_id: int | None = 1,
        is_admin: bool = False,
    ) -> None:
        """Insert through the atomic admission path."""
        from core.database import get_connection

        connection = get_connection()
        try:
            connection.execute(
                "INSERT OR IGNORE INTO api_users (id, username, api_key_hash) "
                "VALUES (1, 'test', 'hash')"
            )
            connection.commit()
        finally:
            connection.close()
        job_service._insert_job(
            job_id,
            {
                "file_id": "f",
                "date_col": "d",
                "value_col": "v",
                "forecast_horizon": 12,
                "forced_model": None,
                "user_prompt": None,
                "preflight_options": {},
            },
            1,
            app_user_id,
            "user",
            is_admin,
            "Test",
            "t.csv",
            "[]",
        )

    def test_queued_limit_rejects(self, job_service):
        """QueuedJobLimitError is raised when pending count reaches the cap."""
        for i in range(5):
            self._insert_pending_job(job_service, f"job-{i}", app_user_id=1)
        with pytest.raises(job_service.QueuedJobLimitError):
            self._insert_via_service(job_service, "job-rejected")

    def test_queued_limit_allows_below_cap(self, job_service):
        """No error when pending count is below the cap."""
        for i in range(4):
            self._insert_pending_job(job_service, f"job-{i}", app_user_id=1)
        self._insert_via_service(job_service, "job-allowed")

        from core.database import get_connection

        conn = get_connection()
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM forecast_jobs WHERE status = 'pending'"
            ).fetchone()[0]
        finally:
            conn.close()
        assert count == 5

    def test_admin_bypasses_queued_limit(self, job_service):
        """Admins bypass the queued limit check in create_job."""
        for i in range(5):
            self._insert_pending_job(job_service, f"job-{i}", app_user_id=1)
        self._insert_via_service(job_service, "job-admin", is_admin=True)

    def test_missing_application_user_id_does_not_bypass_limit(self, job_service):
        """Legacy jobs without a frontend user ID remain admission-limited."""
        for i in range(5):
            self._insert_via_service(job_service, f"job-{i}", app_user_id=None)

        with pytest.raises(job_service.QueuedJobLimitError):
            self._insert_via_service(
                job_service,
                "job-rejected",
                app_user_id=None,
            )

    def test_concurrent_admission_cannot_exceed_limit(self, job_service):
        """Concurrent submissions serialize their count-and-insert decision."""
        from core.database import get_connection

        conn = get_connection()
        try:
            conn.execute(
                "UPDATE forecast_job_settings "
                "SET max_queued_jobs_per_user = 1 WHERE singleton = 1"
            )
            conn.execute(
                "INSERT OR IGNORE INTO api_users (id, username, api_key_hash) "
                "VALUES (1, 'test', 'hash')"
            )
            conn.commit()
        finally:
            conn.close()

        def submit(job_id: str) -> str:
            try:
                self._insert_via_service(job_service, job_id)
                return "accepted"
            except job_service.QueuedJobLimitError:
                return "rejected"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(submit, ("job-a", "job-b")))

        assert sorted(outcomes) == ["accepted", "rejected"]


def test_backend_migrates_pre_queue_schema(tmp_path, monkeypatch) -> None:
    """Backend initialization upgrades databases created before queue changes."""
    db_path = tmp_path / "legacy-backend.db"
    connection = sqlite3.connect(db_path)
    connection.executescript("""
        CREATE TABLE forecast_jobs (
            job_id TEXT PRIMARY KEY,
            backend_owner_id INTEGER,
            application_user_id INTEGER,
            application_username TEXT NOT NULL DEFAULT '',
            application_user_is_admin INTEGER NOT NULL DEFAULT 0,
            file_id TEXT NOT NULL,
            date_col TEXT NOT NULL,
            value_col TEXT NOT NULL,
            forecast_horizon INTEGER NOT NULL,
            forced_model TEXT,
            user_prompt TEXT,
            preflight_options TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL,
            progress INTEGER NOT NULL DEFAULT 0,
            step TEXT NOT NULL,
            error TEXT,
            queued_at TEXT NOT NULL DEFAULT (datetime('now')),
            started_at TEXT,
            completed_at TEXT
        );
        CREATE TABLE forecast_job_settings (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            max_running_jobs_per_user INTEGER NOT NULL DEFAULT 1,
            retention_days INTEGER,
            cleanup_enabled INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        INSERT INTO forecast_job_settings
            (singleton, max_running_jobs_per_user, retention_days, cleanup_enabled)
        VALUES (1, 1, 30, 1);
        """)
    connection.close()

    import core.config as backend_settings
    import core.database as backend_database

    monkeypatch.setattr(backend_settings, "BACKEND_DB_PATH", str(db_path))
    backend_database.init_database()

    connection = sqlite3.connect(db_path)
    try:
        job_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(forecast_jobs)")
        }
        settings_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(forecast_job_settings)")
        }
    finally:
        connection.close()

    assert {
        "report_name",
        "source_filename",
        "custom_settings_json",
        "cancel_requested",
        "heartbeat_at",
        "progress_updated_at",
    } <= job_columns
    assert "max_queued_jobs_per_user" in settings_columns
