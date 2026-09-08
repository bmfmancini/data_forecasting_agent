"""Traditional Forecasting mode — pipeline boundary, worker re-check, persistence.

The effective execution mode is decided in three places, and each must agree:
the HTTP layer at submission, the pipeline boundary before any stage runs,
and the job worker at execution time (an administrator may disable AI after
a job was queued).  These tests exercise the latter two and the mode's
persistence across restarts and migrations.
"""

from __future__ import annotations

import asyncio
import sqlite3
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import core.config as settings
import services.job_service as job_service
import services.pipeline_service as pipeline_service
from core.database import get_connection, init_database, transaction
from core.system_settings_store import is_llm_enabled, set_llm_enabled
from services.file_service import store_file


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolated_backend(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh DB, file store, and queue per test; in-memory cache cleared."""
    from pathlib import Path

    monkeypatch.setattr(settings, "BACKEND_DB_PATH", str(tmp_path / "backend.db"))
    monkeypatch.setattr(settings, "FILE_STORAGE_DIR", str(tmp_path / "files"))
    Path(tmp_path / "files").mkdir()
    monkeypatch.setattr(settings, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(job_service, "JOB_QUEUE", asyncio.Queue())
    job_service._job_store.clear()
    init_database()
    with transaction() as connection:
        connection.execute(
            "INSERT INTO api_users (id, username, api_key_hash) "
            "VALUES (1, 'owner', 'test')"
        )
    yield
    job_service._job_store.clear()


def _series_df() -> pd.DataFrame:
    """A clean monthly series."""
    n = 48
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="MS"),
            "value": 100.0 + 0.5 * np.arange(n) + rng.normal(scale=1.0, size=n),
        }
    )


def _queue_row(job_id: str) -> dict[str, Any]:
    """Read the persisted job row back as a dict (the worker's job payload)."""
    connection = get_connection()
    try:
        row = connection.execute(
            "SELECT * FROM forecast_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return dict(row)
    finally:
        connection.close()


@pytest.fixture
def stored_file() -> str:
    """A stored file available to the job worker."""
    return store_file(_series_df(), "date", "value", "MS", "data.csv", owner_id=1)


def _insert_pending_job(
    job_id: str,
    stored_file: str,
    forced_model: str | None,
    traditional_mode: bool = False,
) -> dict[str, Any]:
    """Persist a pending job row and return it as the worker would read it."""
    job_service._insert_job(
        job_id,
        {
            "file_id": stored_file,
            "date_col": "date",
            "value_col": "value",
            "forecast_horizon": 3,
            "forced_model": forced_model,
            "traditional_mode": traditional_mode,
            "user_prompt": None,
            "preflight_options": {},
        },
        1,
        1,
        "tester",
        False,
    )
    return _queue_row(job_id)


def _fake_pipeline_result() -> SimpleNamespace:
    """A pipeline result shaped for the worker's persistence steps."""
    forecast = {
        "model_used": "ARIMA",
        "forecast_dates": ["2024-01-01", "2024-02-01", "2024-03-01"],
        "forecast": [110.0, 111.0, 112.0],
        "lower_ci": [100.0, 101.0, 102.0],
        "upper_ci": [120.0, 121.0, 122.0],
        "validation_design": {"decision_loss": {"resolved": "mase"}},
    }
    return SimpleNamespace(
        forecast=SimpleNamespace(model_dump=lambda mode="json": forecast),
        model_dump=lambda: {"forecast": forecast},
    )


# ── Pipeline boundary ─────────────────────────────────────────────────────────


class TestPipelineBoundary:
    """Traditional mode without an explicit model is rejected before stages."""

    def test_boundary_rejection_runs_no_stage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stages = MagicMock()
        monkeypatch.setattr(pipeline_service, "_run_statistical_stages", stages)

        with pytest.raises(ValueError, match="Traditional Forecasting requires"):
            pipeline_service.run_pipeline(
                _series_df(),
                file_id="f",
                date_col="date",
                value_col="value",
                freq="MS",
                forecast_horizon=3,
                traditional_mode=True,
            )

        stages.assert_not_called()


# ── Job worker re-check ───────────────────────────────────────────────────────


class TestWorkerRecheck:
    """The worker re-resolves the effective mode from the global setting."""

    @staticmethod
    def _run_job(job_id: str, job: dict[str, Any]) -> None:
        asyncio.run(job_service._run_job(job_id, job))

    @staticmethod
    def _job_row(job_id: str) -> dict[str, Any]:
        connection = get_connection()
        try:
            row = connection.execute(
                "SELECT status, error FROM forecast_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            return dict(row)
        finally:
            connection.close()

    def test_queued_auto_job_fails_loudly_when_ai_disabled(
        self, monkeypatch: pytest.MonkeyPatch, stored_file: str
    ) -> None:
        """An Auto job queued while AI was enabled fails with an explanation
        when the administrator has disabled AI by execution time — a model
        is never silently selected."""
        set_llm_enabled(False)
        run_pipeline = MagicMock()
        monkeypatch.setattr(pipeline_service, "run_pipeline", run_pipeline)

        job = _insert_pending_job("job-auto", stored_file, forced_model=None)
        self._run_job("job-auto", job)

        run_pipeline.assert_not_called()
        row = self._job_row("job-auto")
        assert row["status"] == "error"
        assert "AI features were disabled by an administrator" in row["error"]
        assert "Traditional Forecasting requires" in row["error"]

    def test_forced_model_job_runs_traditional_when_globally_disabled(
        self, monkeypatch: pytest.MonkeyPatch, stored_file: str
    ) -> None:
        """A forced-model job submitted in AI mode is *executed* traditional
        after a deployment-wide disable — the effective mode, not the
        request's intent."""
        set_llm_enabled(False)
        captured: dict[str, Any] = {}

        def fake_run_pipeline(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _fake_pipeline_result()

        monkeypatch.setattr(pipeline_service, "run_pipeline", fake_run_pipeline)
        monkeypatch.setattr(job_service, "index_analysis_results", lambda *a, **k: None)

        job = _insert_pending_job("job-forced", stored_file, forced_model="ARIMA")
        self._run_job("job-forced", job)

        assert captured["traditional_mode"] is True
        assert captured["forced_model"] == "ARIMA"
        assert self._job_row("job-forced")["status"] == "done"

    def test_ai_job_stays_ai_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch, stored_file: str
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_run_pipeline(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _fake_pipeline_result()

        monkeypatch.setattr(pipeline_service, "run_pipeline", fake_run_pipeline)
        monkeypatch.setattr(job_service, "index_analysis_results", lambda *a, **k: None)

        job = _insert_pending_job("job-ai", stored_file, forced_model=None)
        self._run_job("job-ai", job)

        assert captured["traditional_mode"] is False
        assert self._job_row("job-ai")["status"] == "done"

    def test_queued_traditional_job_stays_traditional_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch, stored_file: str
    ) -> None:
        """The per-run flag survives to execution even with AI enabled."""
        captured: dict[str, Any] = {}

        def fake_run_pipeline(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _fake_pipeline_result()

        monkeypatch.setattr(pipeline_service, "run_pipeline", fake_run_pipeline)
        monkeypatch.setattr(job_service, "index_analysis_results", lambda *a, **k: None)

        job = _insert_pending_job(
            "job-trad", stored_file, forced_model="ARIMA", traditional_mode=True
        )
        self._run_job("job-trad", job)

        assert captured["traditional_mode"] is True


# ── Mode persistence ──────────────────────────────────────────────────────────


class TestModePersistence:
    """The mode travels with the job, across restarts and migrations."""

    def test_create_job_persists_traditional_mode(self, stored_file: str) -> None:
        job_id = job_service.create_job(
            file_id=stored_file,
            date_col="date",
            value_col="value",
            forecast_horizon=3,
            forced_model="ARIMA",
            user_prompt=None,
            preflight_options={},
            traditional_mode=True,
            owner_id=1,
            application_user_id=1,
            application_username="tester",
        )

        connection = get_connection()
        try:
            row = connection.execute(
                "SELECT traditional_mode, status FROM forecast_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        finally:
            connection.close()
        assert row["traditional_mode"] == 1
        assert row["status"] == "pending"
        assert job_service._job_store[job_id]["request"]["traditional_mode"] is True

    def test_init_job_queue_restores_pending_traditional_job(
        self, stored_file: str
    ) -> None:
        """A pending traditional job re-queued after a restart keeps its mode."""
        _insert_pending_job(
            "job-restored", stored_file, forced_model="ARIMA", traditional_mode=True
        )

        job_service.init_job_queue()

        assert job_service.JOB_QUEUE.qsize() == 1
        row = _queue_row("job-restored")
        assert row["traditional_mode"] == 1
        assert row["status"] == "pending"


# ── Global setting store ─────────────────────────────────────────────────────


class TestSystemSettingStore:
    """The deployment-wide switch is readable on every install shape."""

    def test_defaults_to_enabled(self) -> None:
        assert is_llm_enabled() is True

    def test_round_trip(self) -> None:
        set_llm_enabled(False)
        assert is_llm_enabled() is False
        set_llm_enabled(True)
        assert is_llm_enabled() is True

    def test_pre_migration_database_stays_enabled(self, tmp_path) -> None:
        """A database without the system_settings table (pre-migration) must
        never silently disable AI."""
        db_path = str(tmp_path / "pre-feature.db")
        connection = sqlite3.connect(db_path)
        try:
            connection.execute("CREATE TABLE llm_config (singleton INTEGER)")
            connection.commit()
        finally:
            connection.close()

        assert is_llm_enabled(db_path) is True


class TestMigrations:
    """The additive migrations are idempotent and apply to pre-feature DBs."""

    @staticmethod
    def _columns(table: str) -> set[str]:
        connection = get_connection()
        try:
            return {
                str(row["name"])
                for row in connection.execute(f"PRAGMA table_info({table})")
            }
        finally:
            connection.close()

    def test_fresh_database_has_columns(self) -> None:
        assert "llm_enabled" in self._columns("system_settings")
        assert "traditional_mode" in self._columns("forecast_jobs")

    def test_running_twice_is_idempotent(self) -> None:
        init_database()
        init_database()
        assert "llm_enabled" in self._columns("system_settings")
        assert "traditional_mode" in self._columns("forecast_jobs")

    def test_pre_feature_database_is_migrated(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A database created before the feature gets the columns added."""
        db_path = str(tmp_path / "old.db")
        connection = sqlite3.connect(db_path)
        try:
            # Minimal pre-feature shapes: system_settings without llm_enabled,
            # forecast_jobs without traditional_mode.
            connection.executescript(
                """
                CREATE TABLE system_settings (
                    singleton      INTEGER PRIMARY KEY CHECK (singleton = 1),
                    setup_complete INTEGER NOT NULL DEFAULT 0,
                    worker_mode    TEXT    NOT NULL DEFAULT 'standalone',
                    updated_at     TEXT    NOT NULL DEFAULT (datetime('now'))
                );
                INSERT INTO system_settings (singleton) VALUES (1);
                CREATE TABLE forecast_jobs (
                    job_id                    TEXT PRIMARY KEY,
                    backend_owner_id          INTEGER,
                    application_user_id       INTEGER,
                    application_username      TEXT NOT NULL DEFAULT '',
                    application_user_is_admin INTEGER NOT NULL DEFAULT 0,
                    file_id                   TEXT NOT NULL,
                    date_col                  TEXT NOT NULL,
                    value_col                  TEXT NOT NULL,
                    forecast_horizon          INTEGER NOT NULL,
                    forced_model              TEXT,
                    user_prompt               TEXT,
                    preflight_options         TEXT NOT NULL DEFAULT '{}',
                    status                    TEXT NOT NULL,
                    progress                  INTEGER NOT NULL DEFAULT 0,
                    step                      TEXT NOT NULL,
                    queued_at                 TEXT NOT NULL DEFAULT (datetime('now'))
                );
                """
            )
            connection.commit()
        finally:
            connection.close()

        monkeypatch.setattr(settings, "BACKEND_DB_PATH", db_path)
        init_database()

        connection = sqlite3.connect(db_path)
        try:
            settings_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(system_settings)")
            }
            job_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(forecast_jobs)")
            }
            enabled = connection.execute(
                "SELECT llm_enabled FROM system_settings WHERE singleton = 1"
            ).fetchone()[0]
        finally:
            connection.close()

        assert "llm_enabled" in settings_columns
        assert "traditional_mode" in job_columns
        # The pre-existing singleton row picks up the column default: on.
        assert enabled == 1
        assert is_llm_enabled(db_path) is True