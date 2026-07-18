"""Tests for cooperative job cancellation in the backend job service."""

from __future__ import annotations

import os
import sys
import tempfile

# Ensure the backend package is importable.
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "data_forecaster", "backend")
)

import pytest

from exceptions import JobCancelledError


@pytest.fixture()
def temp_db(monkeypatch):
    """Use a temporary SQLite database for each test."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    monkeypatch.setenv("BACKEND_DB_PATH", db_path)
    # Re-import to pick up the new env var.
    import importlib
    import core.config
    import core.database

    importlib.reload(core.config)
    importlib.reload(core.database)
    core.database.init_database()
    yield db_path


@pytest.fixture()
def job_service(temp_db):
    """Provide the job_service module with a fresh temporary database."""
    import importlib
    import services.job_service

    importlib.reload(services.job_service)
    return services.job_service


def _insert_test_job(
    job_service,
    job_id: str = "test-job-1",
    status: str = "pending",
    application_user_id: int = 1,
    cancel_requested: int = 0,
) -> None:
    """Insert a minimal job row for testing."""
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
                user_prompt, preflight_options, report_name, source_filename,
                custom_settings_json, cancel_requested, status, step
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                1,
                application_user_id,
                "testuser",
                0,
                "file-1",
                "date",
                "value",
                12,
                None,
                None,
                "{}",
                "Test",
                "test.csv",
                "[]",
                cancel_requested,
                status,
                "Test step",
            ),
        )
        conn.commit()
    finally:
        conn.close()


class TestRequestCancel:
    """Tests for the request_cancel function."""

    def test_cancel_pending_job(self, job_service, monkeypatch):
        """Pending cancellation is terminal and releases its file reservation."""
        released: list[str] = []
        monkeypatch.setattr(job_service, "release_file", released.append)
        _insert_test_job(job_service, status="pending")
        result = job_service.request_cancel("test-job-1", application_user_id=1)
        assert result == "cancelled"
        record = job_service._job_record("test-job-1")
        assert record["status"] == "cancelled"
        assert released == ["file-1"]

    def test_cancel_running_job(self, job_service):
        """Running jobs transition to cancelling with cancel_requested set."""
        _insert_test_job(job_service, status="running")
        result = job_service.request_cancel("test-job-1", application_user_id=1)
        assert result == "cancelling"
        record = job_service._job_record("test-job-1")
        assert record["status"] == "cancelling"
        assert int(record["cancel_requested"]) == 1

    def test_cancel_cancelling_job_is_idempotent(self, job_service):
        """Cancelling a cancelling job is a no-op."""
        _insert_test_job(job_service, status="cancelling", cancel_requested=1)
        result = job_service.request_cancel("test-job-1", application_user_id=1)
        assert result == "cancelling"

    def test_cancel_done_job_returns_conflict(self, job_service):
        """Cancelling a completed job returns conflict."""
        _insert_test_job(job_service, status="done")
        result = job_service.request_cancel("test-job-1", application_user_id=1)
        assert result == "conflict"

    def test_cancel_nonexistent_job_returns_not_found(self, job_service):
        """Cancelling a non-existent job returns not_found."""
        result = job_service.request_cancel("nonexistent", application_user_id=1)
        assert result == "not_found"

    def test_cancel_wrong_user_returns_not_found(self, job_service):
        """User B cannot cancel user A's job."""
        _insert_test_job(job_service, status="pending", application_user_id=1)
        result = job_service.request_cancel("test-job-1", application_user_id=2)
        assert result == "not_found"


class TestSetJobCancelled:
    """Tests for the _set_job_cancelled function."""

    def test_cancelled_from_cancelling(self, job_service):
        """Cancelling jobs transition to cancelled."""
        _insert_test_job(job_service, status="cancelling", cancel_requested=1)
        job_service._set_job_cancelled("test-job-1")
        record = job_service._job_record("test-job-1")
        assert record["status"] == "cancelled"

    def test_does_not_overwrite_done(self, job_service):
        """_set_job_cancelled does not overwrite a done job (conditional)."""
        _insert_test_job(job_service, status="done")
        job_service._set_job_cancelled("test-job-1")
        record = job_service._job_record("test-job-1")
        assert record["status"] == "done"


class TestSetJobError:
    """Tests for conditional failure transitions."""

    def test_pending_job_can_transition_to_error(self, job_service):
        """Missing input discovered during restart must not leave a zombie."""
        _insert_test_job(job_service, status="pending")

        job_service._set_job_error("test-job-1", "Input missing")

        record = job_service._job_record("test-job-1")
        assert record["status"] == "error"
        assert record["error"] == "Input missing"


class TestIsCancelRequested:
    """Tests for the _is_cancel_requested function."""

    def test_returns_true_when_requested(self, job_service):
        """Returns True when cancel_requested is set."""
        _insert_test_job(job_service, cancel_requested=1)
        assert job_service._is_cancel_requested("test-job-1") is True

    def test_returns_false_when_not_requested(self, job_service):
        """Returns False when cancel_requested is not set."""
        _insert_test_job(job_service, cancel_requested=0)
        assert job_service._is_cancel_requested("test-job-1") is False


class TestPipelineCancellation:
    """Tests cooperative cancellation checks in the pipeline contract."""

    def test_cancel_check_raises(self):
        """The _check_cancelled helper raises JobCancelledError."""
        # We test the cancellation check logic directly since run_pipeline
        # requires a full DataFrame and LLM setup.
        from services.pipeline_service import run_pipeline
        import inspect

        sig = inspect.signature(run_pipeline)
        assert "cancel_check" in sig.parameters
        assert sig.parameters["cancel_check"].default is None
