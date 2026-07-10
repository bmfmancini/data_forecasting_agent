"""Persistent scheduling and execution for forecast analysis jobs."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from collections.abc import Callable
from typing import Any, cast

import core.config as settings
from core.database import get_connection
from core.logging_config import get_logger
from schemas import AnalysisResponse
from services.file_service import get_file, release_file, reserve_file
from services.pipeline_service import run_pipeline
from services.rag_service import get_rag_kb

logger = get_logger(__name__)

MAX_JOBS: int = settings.MAX_INMEMORY_JOBS
_job_store: dict[str, dict[str, Any]] = {}
_job_store_lock = threading.Lock()
JOB_QUEUE: asyncio.Queue[str] = cast(asyncio.Queue, None)


def init_job_queue() -> None:
    """Initialise the queue and restore jobs left pending after a restart."""
    global JOB_QUEUE  # pylint: disable=global-statement
    JOB_QUEUE = asyncio.Queue()
    connection = get_connection()
    try:
        connection.execute(
            """
            UPDATE forecast_jobs
            SET status = 'error', error = 'Backend restarted during processing.',
                step = 'Interrupted by backend restart.', completed_at = datetime('now')
            WHERE status = 'running'
            """
        )
        pending_rows = connection.execute(
            "SELECT job_id, file_id FROM forecast_jobs WHERE status = 'pending' "
            "ORDER BY queued_at, rowid"
        ).fetchall()
        connection.commit()
    finally:
        connection.close()

    for row in pending_rows:
        job_id = str(row["job_id"])
        if reserve_file(str(row["file_id"])):
            JOB_QUEUE.put_nowait(job_id)
        else:
            _set_job_error(job_id, "Input file is no longer available.")


def is_queue_ready() -> bool:
    """Return whether the scheduler queue has been initialised."""
    return JOB_QUEUE is not None


def get_job_settings() -> dict[str, Any]:
    """Return the current persistent job scheduler settings."""
    connection = get_connection()
    try:
        row = connection.execute(
            """
            SELECT max_running_jobs_per_user, retention_days, cleanup_enabled
            FROM forecast_job_settings WHERE singleton = 1
            """
        ).fetchone()
        return dict(row) if row else {
            "max_running_jobs_per_user": 1,
            "retention_days": 30,
            "cleanup_enabled": 1,
        }
    finally:
        connection.close()


def update_job_settings(
    max_running_jobs_per_user: int,
    retention_days: int | None,
    cleanup_enabled: bool,
) -> dict[str, Any]:
    """Persist scheduler settings and return their normalized representation."""
    connection = get_connection()
    try:
        connection.execute(
            """
            UPDATE forecast_job_settings
            SET max_running_jobs_per_user = ?, retention_days = ?,
                cleanup_enabled = ?, updated_at = datetime('now')
            WHERE singleton = 1
            """,
            (max_running_jobs_per_user, retention_days, int(cleanup_enabled)),
        )
        connection.commit()
    finally:
        connection.close()
    return get_job_settings()


def cleanup_terminal_jobs() -> int:
    """Delete expired completed or failed jobs without touching active work."""
    settings_row = get_job_settings()
    if not settings_row["cleanup_enabled"] or settings_row["retention_days"] is None:
        return 0
    connection = get_connection()
    try:
        cursor = connection.execute(
            """
            DELETE FROM forecast_jobs
            WHERE status IN ('done', 'error')
              AND completed_at < datetime('now', ?)
            """,
            (f"-{int(settings_row['retention_days'])} days",),
        )
        connection.commit()
        return cursor.rowcount
    finally:
        connection.close()


def clear_terminal_jobs() -> int:
    """Delete completed and failed job records and release cached results."""
    connection = get_connection()
    try:
        terminal_rows = connection.execute(
            "SELECT job_id FROM forecast_jobs WHERE status IN ('done', 'error')"
        ).fetchall()
        cursor = connection.execute(
            "DELETE FROM forecast_jobs WHERE status IN ('done', 'error')"
        )
        connection.commit()
    finally:
        connection.close()
    with _job_store_lock:
        for row in terminal_rows:
            _job_store.pop(str(row["job_id"]), None)
    return cursor.rowcount


def create_job(
    file_id: str,
    date_col: str,
    value_col: str,
    forecast_horizon: int,
    forced_model: str | None,
    user_prompt: str | None,
    preflight_options: dict[str, Any] | None,
    owner_id: int | None = None,
    application_user_id: int | None = None,
    application_username: str | None = None,
    application_user_is_admin: bool = False,
) -> str:
    """Create and persist a pending job before placing it on the scheduler queue."""
    job_id = str(uuid.uuid4())
    request_data = {
        "file_id": file_id,
        "date_col": date_col,
        "value_col": value_col,
        "forecast_horizon": forecast_horizon,
        "forced_model": forced_model,
        "user_prompt": user_prompt,
        "preflight_options": preflight_options or {},
    }
    _insert_job(
        job_id,
        request_data,
        owner_id,
        application_user_id,
        application_username or "Unknown user",
        application_user_is_admin,
    )
    if not reserve_file(file_id):
        _set_job_error(job_id, f"File ID '{file_id}' not found in store.")
        raise ValueError(f"File ID '{file_id}' not found in store.")
    with _job_store_lock:
        _evict_cached_terminal_jobs()
        _job_store[job_id] = {
            "owner_id": owner_id,
            "request": request_data,
            "result": None,
            "error": None,
        }
    JOB_QUEUE.put_nowait(job_id)
    logger.info("Job enqueued: job_id=%s file_id=%s", job_id, file_id)
    return job_id


def _insert_job(
    job_id: str,
    request_data: dict[str, Any],
    owner_id: int | None,
    application_user_id: int | None,
    application_username: str,
    application_user_is_admin: bool,
) -> None:
    """Insert a queued job record."""
    connection = get_connection()
    try:
        connection.execute(
            """
            INSERT INTO forecast_jobs (
                job_id, backend_owner_id, application_user_id,
                application_username, application_user_is_admin, file_id, date_col,
                value_col, forecast_horizon, forced_model, user_prompt,
                preflight_options, status, step
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                job_id, owner_id, application_user_id, application_username,
                int(application_user_is_admin), request_data["file_id"],
                request_data["date_col"], request_data["value_col"],
                request_data["forecast_horizon"], request_data["forced_model"],
                request_data["user_prompt"], json.dumps(request_data["preflight_options"]),
                "Queued — waiting for an available slot…",
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _evict_cached_terminal_jobs() -> None:
    """Bound in-memory results while retaining complete job metadata in SQLite."""
    while len(_job_store) >= MAX_JOBS:
        oldest_job_id = next(iter(_job_store))
        _job_store.pop(oldest_job_id)


def _claim_job(job_id: str) -> dict[str, Any] | None:
    """Atomically mark a pending job running when its user has capacity."""
    connection = get_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM forecast_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None or row["status"] != "pending":
            connection.rollback()
            return None
        settings_row = connection.execute(
            "SELECT max_running_jobs_per_user FROM forecast_job_settings "
            "WHERE singleton = 1"
        ).fetchone()
        if not int(row["application_user_is_admin"]):
            running_count = connection.execute(
                """
                SELECT COUNT(*) AS count FROM forecast_jobs
                WHERE application_user_id IS ? AND status = 'running'
                """,
                (row["application_user_id"],),
            ).fetchone()["count"]
            if running_count >= settings_row["max_running_jobs_per_user"]:
                connection.rollback()
                return None
        connection.execute(
            """
            UPDATE forecast_jobs
            SET status = 'running', step = 'Analysis in progress.',
                started_at = datetime('now')
            WHERE job_id = ?
            """,
            (job_id,),
        )
        connection.commit()
        return dict(row)
    finally:
        connection.close()


def _job_record(job_id: str) -> dict[str, Any] | None:
    """Return a job record from durable storage."""
    connection = get_connection()
    try:
        row = connection.execute(
            "SELECT * FROM forecast_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        connection.close()


def get_job(job_id: str, requester: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Return a job's current state when the requester owns it or is an admin."""
    record = _job_record(job_id)
    if record is None:
        return None
    if requester and not requester.get("is_admin") and record["backend_owner_id"] != requester.get("id"):
        return None
    cached = _job_store.get(job_id, {})
    return {**record, "result": cached.get("result")}


def get_job_status_only(
    job_id: str, requester: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Return the lightweight job state used by polling."""
    job = get_job(job_id, requester)
    if job is None:
        return None
    return {"status": job["status"], "progress": job["progress"], "step": job["step"]}


def list_recent_jobs(limit: int = 25) -> list[dict[str, Any]]:
    """Return the most recent queue records for administrators."""
    connection = get_connection()
    try:
        rows = connection.execute(
            """
            SELECT job_id, application_username, status, progress, step,
                   forecast_horizon, forced_model, queued_at, started_at,
                   completed_at, error
            FROM forecast_jobs ORDER BY queued_at DESC, rowid DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def _update_job_progress(job_id: str, pct: int, step: str) -> None:
    """Persist progress updates from the pipeline execution thread."""
    connection = get_connection()
    try:
        connection.execute(
            "UPDATE forecast_jobs SET progress = ?, step = ? WHERE job_id = ?",
            (pct, step, job_id),
        )
        connection.commit()
    finally:
        connection.close()


def _set_job_error(job_id: str, error: str) -> None:
    """Mark a job terminally failed."""
    connection = get_connection()
    try:
        connection.execute(
            """
            UPDATE forecast_jobs
            SET status = 'error', step = 'Pipeline failed.', error = ?,
                completed_at = datetime('now')
            WHERE job_id = ?
            """,
            (error, job_id),
        )
        connection.commit()
    finally:
        connection.close()


async def job_worker() -> None:
    """Process queue items while respecting each non-admin user's running cap."""
    while True:
        job_id = await JOB_QUEUE.get()
        try:
            job = _claim_job(job_id)
            if job is None:
                JOB_QUEUE.put_nowait(job_id)
                await asyncio.sleep(0.1)
                continue
            await _run_job(job_id, job)
        finally:
            JOB_QUEUE.task_done()


async def _run_job(job_id: str, job: dict[str, Any]) -> None:
    """Run one claimed job and persist its terminal state."""
    stored = get_file(str(job["file_id"]))
    if stored is None:
        _set_job_error(job_id, "Uploaded file not found.")
        release_file(str(job["file_id"]))
        return

    def run_pipeline_sync() -> AnalysisResponse:
        return run_pipeline(
            df=stored["df"], file_id=str(job["file_id"]),
            date_col=str(job["date_col"]), value_col=str(job["value_col"]),
            freq=stored["freq"], forecast_horizon=int(job["forecast_horizon"]),
            forced_model=job["forced_model"], user_prompt=job["user_prompt"],
            preflight_options=json.loads(str(job["preflight_options"])),
            chroma_persist_dir=settings.CHROMA_PERSIST_DIR,
            progress_callback=lambda pct, step: _update_job_progress(job_id, pct, step),
        )

    try:
        result = await asyncio.to_thread(run_pipeline_sync)
        _complete_job(job_id)
        with _job_store_lock:
            _job_store.setdefault(job_id, {})["result"] = result.model_dump()
        try:
            await asyncio.to_thread(
                index_analysis_results, str(job["file_id"]), job["backend_owner_id"],
                result, settings.CHROMA_PERSIST_DIR,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Analysis result indexing failed for job_id=%s", job_id)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Pipeline failed for job_id=%s", job_id)
        _set_job_error(job_id, str(exc))
    finally:
        release_file(str(job["file_id"]))


def _complete_job(job_id: str) -> None:
    """Mark a job complete in durable storage."""
    connection = get_connection()
    try:
        connection.execute(
            """
            UPDATE forecast_jobs
            SET status = 'done', progress = 100, step = 'Analysis complete.',
                completed_at = datetime('now')
            WHERE job_id = ?
            """,
            (job_id,),
        )
        connection.commit()
    finally:
        connection.close()


def index_analysis_results(
    file_id: str, owner_id: int | None, analysis_result: AnalysisResponse,
    chroma_persist_dir: str,
) -> None:
    """Index completed analysis results without making indexing job-critical."""
    rag_kb = get_rag_kb(chroma_persist_dir)
    summary = (
        f"Forecasting Analysis Results for {file_id}:\n"
        f"Summary Report: {analysis_result.report}\n"
        f"Model Selection: {analysis_result.model_selection.selected_model}\n"
        f"Statistical Insights: {analysis_result.statistical.model_dump_json()}\n"
    )
    if hasattr(rag_kb, "add_texts"):
        rag_kb.add_texts(
            texts=[summary],
            metadatas=[{"file_id": file_id, "owner_id": str(owner_id or ""), "type": "analysis_result"}],
        )
