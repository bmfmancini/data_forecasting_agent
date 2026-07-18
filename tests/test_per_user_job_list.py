"""Tests for per-user job listing and owner-scoped access."""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "data_forecaster", "backend")
)

import pytest

@pytest.fixture()
def temp_db(monkeypatch):
    """Use a temporary SQLite database for each test."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    monkeypatch.setenv("BACKEND_DB_PATH", db_path)
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


def _insert_job(
    job_service,
    job_id: str,
    application_user_id: int | None,
    application_username: str | None = "user",
    status: str = "pending",
    backend_owner_id: int = 1,
) -> None:
    """Insert a minimal job row for testing."""
    from core.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO api_users (id, username, api_key_hash) "
            "VALUES (?, ?, 'hash')",
            (backend_owner_id, f"owner-{backend_owner_id}"),
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
                backend_owner_id,
                application_user_id,
                application_username,
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
                0,
                status,
                "Test step",
            ),
        )
        conn.commit()
    finally:
        conn.close()


class TestListJobsForUser:
    """Tests for list_jobs_for_user."""

    def test_returns_only_users_jobs(self, job_service):
        """list_jobs_for_user returns only the specified user's jobs."""
        _insert_job(
            job_service, "job-a1", application_user_id=1, application_username="alice"
        )
        _insert_job(
            job_service, "job-a2", application_user_id=1, application_username="alice"
        )
        _insert_job(
            job_service, "job-b1", application_user_id=2, application_username="bob"
        )
        jobs = job_service.list_jobs_for_user(1, backend_owner_id=1)
        assert len(jobs) == 2
        job_ids = [j["job_id"] for j in jobs]
        assert "job-a1" in job_ids
        assert "job-a2" in job_ids
        assert "job-b1" not in job_ids

    def test_returns_empty_for_user_with_no_jobs(self, job_service):
        """Returns empty list for a user with no jobs."""
        _insert_job(job_service, "job-a1", application_user_id=1)
        jobs = job_service.list_jobs_for_user(99, backend_owner_id=1)
        assert jobs == []

    def test_direct_api_jobs_are_listed_without_application_user(self, job_service):
        """Direct API jobs are scoped by backend owner without frontend metadata."""
        _insert_job(job_service, "job-direct", application_user_id=None)
        _insert_job(job_service, "job-frontend", application_user_id=1)

        jobs = job_service.list_jobs_for_user(None, backend_owner_id=1)

        assert [job["job_id"] for job in jobs] == ["job-direct"]

    def test_includes_report_name(self, job_service):
        """The returned jobs include the report_name field."""
        _insert_job(job_service, "job-1", application_user_id=1)
        jobs = job_service.list_jobs_for_user(1, backend_owner_id=1)
        assert jobs[0]["report_name"] == "Test"

    def test_filters_same_application_user_by_backend_owner(self, job_service):
        """Overlapping frontend user IDs remain isolated by backend owner."""
        _insert_job(
            job_service,
            "job-owner-1",
            application_user_id=1,
            backend_owner_id=1,
        )
        _insert_job(
            job_service,
            "job-owner-2",
            application_user_id=1,
            backend_owner_id=2,
        )

        jobs = job_service.list_jobs_for_user(1, backend_owner_id=1)

        assert [job["job_id"] for job in jobs] == ["job-owner-1"]


def test_clear_terminal_jobs_for_user_preserves_other_and_active_jobs(
    job_service,
) -> None:
    """User cleanup deletes only their terminal queue records."""
    _insert_job(job_service, "mine-done", application_user_id=1, status="done")
    _insert_job(job_service, "mine-error", application_user_id=1, status="error")
    _insert_job(
        job_service, "mine-cancelled", application_user_id=1, status="cancelled"
    )
    _insert_job(job_service, "mine-active", application_user_id=1, status="running")
    _insert_job(job_service, "other-done", application_user_id=2, status="done")
    _insert_job(
        job_service,
        "other-owner-done",
        application_user_id=1,
        status="done",
        backend_owner_id=2,
    )

    deleted = job_service.clear_terminal_jobs_for_user(1, backend_owner_id=1)

    assert deleted == 3
    assert job_service.get_job("mine-done", application_user_id=1) is None
    assert job_service.get_job("mine-error", application_user_id=1) is None
    assert job_service.get_job("mine-cancelled", application_user_id=1) is None
    assert job_service.get_job("mine-active", application_user_id=1) is not None
    assert job_service.get_job("other-done", application_user_id=2) is not None
    assert job_service.get_job("other-owner-done", application_user_id=1) is not None


class TestOwnerScopedGetJob:
    """Tests for get_job with application_user_id scoping."""

    def test_user_can_get_own_job(self, job_service):
        """User can retrieve their own job."""
        _insert_job(job_service, "job-1", application_user_id=1, status="done")
        job = job_service.get_job("job-1", application_user_id=1)
        assert job is not None
        assert job["job_id"] == "job-1"

    def test_user_cannot_get_other_users_job(self, job_service):
        """User B cannot retrieve user A's job even with same backend_owner_id."""
        _insert_job(job_service, "job-1", application_user_id=1, status="done")
        job = job_service.get_job("job-1", application_user_id=2)
        assert job is None

    def test_get_job_status_only_owner_scoped(self, job_service):
        """get_job_status_only respects application_user_id scoping."""
        _insert_job(job_service, "job-1", application_user_id=1, status="running")
        status = job_service.get_job_status_only("job-1", application_user_id=1)
        assert status is not None
        assert status["status"] == "running"
        status_b = job_service.get_job_status_only("job-1", application_user_id=2)
        assert status_b is None

    def test_frontend_job_requires_application_identity(self, job_service):
        """Even a backend admin cannot omit identity on a frontend-owned job."""
        _insert_job(job_service, "job-1", application_user_id=1, status="done")

        job = job_service.get_job(
            "job-1",
            requester={"id": 1, "is_admin": True},
        )

        assert job is None


def test_application_user_id_is_declared_as_header() -> None:
    """FastAPI binds the optional delegated application-user header."""
    from main import app

    parameters = app.openapi()["paths"]["/jobs/mine"]["get"]["parameters"]
    parameter_locations = {(item["name"], item["in"]) for item in parameters}

    assert ("X-Application-User-ID", "header") in parameter_locations


def test_full_job_response_includes_finalization_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full job API contract includes the persisted metadata snapshot."""
    import main

    monkeypatch.setattr(
        main,
        "get_job",
        lambda *_args, **_kwargs: {
            "status": "done",
            "progress": 100,
            "step": "Complete",
            "result": {"report": "Ready"},
            "error": None,
            "report_name": "Quarterly forecast",
            "source_filename": "quarterly.csv",
            "forecast_horizon": 6,
            "custom_settings_json": '[{"name": "currency"}]',
        },
    )

    response = main.get_job_status(
        "job-1",
        _user={"id": 1},
        app_user_id=1,
    )

    assert response.source_filename == "quarterly.csv"
    assert response.forecast_horizon == 6
    assert response.custom_settings_json == '[{"name": "currency"}]'
