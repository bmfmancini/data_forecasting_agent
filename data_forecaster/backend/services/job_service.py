"""Persistent scheduling and execution for forecast analysis jobs."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sqlite3
import threading
import uuid
from typing import Any, cast

import core.config as settings
from core.database import get_connection
from core.logging_config import get_logger
from exceptions import JobCancelledError
from schemas import AnalysisResponse
from services.file_service import get_file, release_file, reserve_file

logger = get_logger(__name__)

MAX_JOBS: int = settings.MAX_INMEMORY_JOBS
_job_store: dict[str, dict[str, Any]] = {}
_job_store_lock = threading.Lock()
JOB_QUEUE: asyncio.Queue[str] = cast(asyncio.Queue, None)

# Directory where completed analysis results are persisted as JSON so they
# survive backend restarts and in-memory cache eviction.
_RESULTS_DIR: str = os.path.join(settings.FILE_STORAGE_DIR, "results")

# Statuses that are considered terminal (no further transitions expected).
_TERMINAL_STATUSES: tuple[str, ...] = ("done", "error", "cancelled")

# Statuses that count toward the per-user running concurrency cap.  A
# ``cancelling`` job still occupies its slot until the worker acknowledges.
_RUNNING_STATUSES: tuple[str, ...] = ("running", "cancelling")


def init_job_queue() -> None:
    """Initialise the queue and restore jobs left pending after a restart."""
    global JOB_QUEUE  # pylint: disable=global-statement
    JOB_QUEUE = asyncio.Queue()
    _ensure_results_dir()
    connection = get_connection()
    try:
        connection.execute("""
            UPDATE forecast_jobs
            SET status = 'error', error = 'Backend restarted during processing.',
                step = 'Interrupted by backend restart.', completed_at = datetime('now')
            WHERE status IN ('running', 'cancelling')
            """)
        pending_rows = connection.execute(
            "SELECT job_id, file_id FROM forecast_jobs WHERE status = 'pending' "
            "ORDER BY queued_at, rowid"
        ).fetchall()
        connection.commit()
    finally:
        connection.close()

    # Clean up orphaned result files for jobs that are no longer ``done``
    # (e.g. interrupted by the restart above).
    _cleanup_orphaned_result_files()

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
        row = connection.execute("""
            SELECT max_running_jobs_per_user, max_queued_jobs_per_user,
                   retention_days, cleanup_enabled
            FROM forecast_job_settings WHERE singleton = 1
            """).fetchone()
        return (
            dict(row)
            if row
            else {
                "max_running_jobs_per_user": 1,
                "max_queued_jobs_per_user": 5,
                "retention_days": 30,
                "cleanup_enabled": 1,
            }
        )
    finally:
        connection.close()


def update_job_settings(
    max_running_jobs_per_user: int,
    max_queued_jobs_per_user: int,
    retention_days: int | None,
    cleanup_enabled: bool,
) -> dict[str, Any]:
    """Persist scheduler settings and return their normalized representation."""
    connection = get_connection()
    try:
        connection.execute(
            """
            UPDATE forecast_job_settings
            SET max_running_jobs_per_user = ?, max_queued_jobs_per_user = ?,
                retention_days = ?, cleanup_enabled = ?, updated_at = datetime('now')
            WHERE singleton = 1
            """,
            (
                max_running_jobs_per_user,
                max_queued_jobs_per_user,
                retention_days,
                int(cleanup_enabled),
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return get_job_settings()


def cleanup_terminal_jobs() -> int:
    """Delete expired terminal jobs without touching active work.

    Terminal statuses are ``done``, ``error``, and ``cancelled``.  The
    durable result file on disk is deleted alongside the job row.
    """
    settings_row = get_job_settings()
    if not settings_row["cleanup_enabled"] or settings_row["retention_days"] is None:
        return 0
    connection = get_connection()
    try:
        rows = connection.execute(
            """
            SELECT job_id FROM forecast_jobs
            WHERE status IN ('done', 'error', 'cancelled')
              AND completed_at < datetime('now', ?)
            """,
            (f"-{int(settings_row['retention_days'])} days",),
        ).fetchall()
        cursor = connection.execute(
            """
            DELETE FROM forecast_jobs
            WHERE status IN ('done', 'error', 'cancelled')
              AND completed_at < datetime('now', ?)
            """,
            (f"-{int(settings_row['retention_days'])} days",),
        )
        connection.commit()
    finally:
        connection.close()
    for row in rows:
        _delete_result_file(str(row["job_id"]))
    return cursor.rowcount


def clear_terminal_jobs() -> int:
    """Delete terminal job records, cached results, and durable result files.

    Terminal statuses are ``done``, ``error``, and ``cancelled``.
    """
    connection = get_connection()
    try:
        terminal_rows = connection.execute(
            "SELECT job_id FROM forecast_jobs "
            "WHERE status IN ('done', 'error', 'cancelled')"
        ).fetchall()
        cursor = connection.execute(
            "DELETE FROM forecast_jobs "
            "WHERE status IN ('done', 'error', 'cancelled')"
        )
        connection.commit()
    finally:
        connection.close()
    with _job_store_lock:
        for row in terminal_rows:
            _job_store.pop(str(row["job_id"]), None)
    for row in terminal_rows:
        _delete_result_file(str(row["job_id"]))
    return cursor.rowcount


class QueuedJobLimitError(ValueError):
    """Raised when a non-admin user exceeds the per-user queued-job cap."""


def create_job(  # pylint: disable=too-many-arguments,too-many-positional-arguments
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
    report_name: str = "",
    source_filename: str = "",
    custom_settings: list[dict[str, str]] | None = None,
) -> str:
    """Create and persist a pending job before placing it on the scheduler queue.

    Args:
        file_id: Identifier of the uploaded data file.
        date_col: Name of the date column.
        value_col: Name of the value column.
        forecast_horizon: Number of future periods to forecast.
        forced_model: Optional model override.
        user_prompt: Optional extra instructions for the report agent.
        preflight_options: Optional preflight configuration dict.
        owner_id: Backend API user ID of the submitter.
        application_user_id: Frontend application user ID.
        application_username: Frontend application username.
        application_user_is_admin: Whether the frontend user is an admin.
        report_name: Display name for the job in the user queue.
        source_filename: Original uploaded CSV filename.
        custom_settings: User-selected report settings to persist with the job.

    Returns:
        The newly created job ID.

    Raises:
        QueuedJobLimitError: When a non-admin user has too many pending jobs.
        ValueError: When the uploaded file cannot be reserved.
    """
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
    if not reserve_file(file_id):
        raise ValueError(f"File ID '{file_id}' not found in store.")
    try:
        _insert_job(
            job_id,
            request_data,
            owner_id,
            application_user_id,
            application_username or "Unknown user",
            application_user_is_admin,
            report_name,
            source_filename,
            json.dumps(custom_settings or []),
        )
    except (QueuedJobLimitError, sqlite3.Error):
        release_file(file_id)
        raise
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


def _enforce_queued_limit(
    connection: sqlite3.Connection,
    backend_owner_id: int | None,
    application_user_id: int | None,
) -> None:
    """Raise ``QueuedJobLimitError`` when the user has too many pending jobs.

    Args:
        connection: Open connection holding an immediate write transaction.
        backend_owner_id: Backend API user ID that owns the queue.
        application_user_id: Frontend application user ID to check.

    Raises:
        QueuedJobLimitError: When pending count is at or above the configured cap.
    """
    settings_row = connection.execute(
        "SELECT max_queued_jobs_per_user FROM forecast_job_settings "
        "WHERE singleton = 1"
    ).fetchone()
    max_queued = int(settings_row["max_queued_jobs_per_user"])
    pending_count = connection.execute(
        """
        SELECT COUNT(*) AS count FROM forecast_jobs
        WHERE backend_owner_id IS ? AND application_user_id IS ?
          AND status = 'pending'
        """,
        (backend_owner_id, application_user_id),
    ).fetchone()["count"]
    if int(pending_count) >= max_queued:
        raise QueuedJobLimitError(
            "You have too many queued forecasts. Wait for one to start or "
            "cancel one before submitting another."
        )


def _insert_job(
    job_id: str,
    request_data: dict[str, Any],
    owner_id: int | None,
    application_user_id: int | None,
    application_username: str,
    application_user_is_admin: bool,
    report_name: str,
    source_filename: str,
    custom_settings_json: str,
) -> None:
    """Atomically enforce queued capacity and insert a queued job record."""
    connection = get_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        if not application_user_is_admin:
            _enforce_queued_limit(
                connection,
                owner_id,
                application_user_id,
            )
        connection.execute(
            """
            INSERT INTO forecast_jobs (
                job_id, backend_owner_id, application_user_id,
                application_username, application_user_is_admin, file_id, date_col,
                value_col, forecast_horizon, forced_model, user_prompt,
                preflight_options, report_name, source_filename,
                custom_settings_json, status, step
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                job_id,
                owner_id,
                application_user_id,
                application_username,
                int(application_user_is_admin),
                request_data["file_id"],
                request_data["date_col"],
                request_data["value_col"],
                request_data["forecast_horizon"],
                request_data["forced_model"],
                request_data["user_prompt"],
                json.dumps(request_data["preflight_options"]),
                report_name,
                source_filename,
                custom_settings_json,
                "Queued — waiting for an available slot…",
            ),
        )
        connection.commit()
    except (QueuedJobLimitError, sqlite3.Error):
        connection.rollback()
        raise
    finally:
        connection.close()


def _evict_cached_terminal_jobs() -> None:
    """Bound in-memory results while retaining complete job metadata in SQLite."""
    while len(_job_store) >= MAX_JOBS:
        oldest_job_id = next(iter(_job_store))
        _job_store.pop(oldest_job_id)


def _claim_job(job_id: str) -> dict[str, Any] | None:
    """Atomically mark a pending job running when its user has capacity.

    A job that was cancelled while pending (``cancel_requested = 1``) is
    transitioned to ``cancelled`` here and ``None`` is returned so the worker
    skips it.
    """
    connection = get_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM forecast_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None or row["status"] != "pending":
            connection.rollback()
            return None
        # If the job was cancelled while pending, finalise the cancellation
        # atomically and skip execution.
        if int(row["cancel_requested"]):
            connection.execute(
                """
                UPDATE forecast_jobs
                SET status = 'cancelled', step = 'Cancelled.',
                    completed_at = datetime('now')
                WHERE job_id = ? AND status = 'pending'
                """,
                (job_id,),
            )
            connection.commit()
            release_file(str(row["file_id"]))
            return None
        settings_row = connection.execute(
            "SELECT max_running_jobs_per_user FROM forecast_job_settings "
            "WHERE singleton = 1"
        ).fetchone()
        if not int(row["application_user_is_admin"]):
            placeholders = ",".join("?" for _ in _RUNNING_STATUSES)
            running_count = connection.execute(
                f"""
                SELECT COUNT(*) AS count FROM forecast_jobs
                WHERE backend_owner_id IS ? AND application_user_id IS ?
                  AND status IN ({placeholders})
                """,
                (
                    row["backend_owner_id"],
                    row["application_user_id"],
                    *_RUNNING_STATUSES,
                ),
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


def _is_owner_authorized(
    record: dict[str, Any],
    requester: dict[str, Any] | None,
    application_user_id: int | None = None,
) -> bool:
    """Return ``True`` when the requester is allowed to access the job.

    Frontend-owned jobs always require their matching ``application_user_id``
    because multiple app users may share the same ``backend_owner_id``.
    Backend-only jobs without an application user retain backend-owner/admin
    authorization.

    Args:
        record: The job record from durable storage.
        requester: The authenticated backend API user dict (may be ``None``
            when API-key auth is disabled).
        application_user_id: Optional frontend application user ID to scope by.

    Returns:
        ``True`` if the requester is authorized for the job's ownership scope.
    """
    record_application_user_id = record.get("application_user_id")
    if record_application_user_id is not None and application_user_id is None:
        return False
    if requester is None:
        # Auth disabled — allow if no app-user scoping is requested, or if
        # the app-user ID matches.
        if application_user_id is None:
            return True
        return record.get("application_user_id") == application_user_id
    if requester.get("is_admin") and application_user_id is None:
        # Backend-only job accessed through an administrator credential.
        return True
    # Backend-owner check.
    if record.get("backend_owner_id") != requester.get("id"):
        return False
    # App-user scoping (multiple app users share one backend credential).
    if application_user_id is not None:
        return record.get("application_user_id") == application_user_id
    return True


def get_job(
    job_id: str,
    requester: dict[str, Any] | None = None,
    application_user_id: int | None = None,
) -> dict[str, Any] | None:
    """Return a job's current state when the requester owns it or is an admin.

    Args:
        job_id: The job identifier.
        requester: The authenticated backend API user dict.
        application_user_id: Optional frontend application user ID for
            owner-scoped access.  When provided, the job is only returned if
            its ``application_user_id`` matches.

    Returns:
        The job record with a ``result`` key, or ``None`` if not found or
        not authorized.
    """
    record = _job_record(job_id)
    if record is None:
        return None
    if not _is_owner_authorized(record, requester, application_user_id):
        return None
    cached = _job_store.get(job_id, {})
    result = cached.get("result")
    if result is None and record["status"] == "done":
        result = _load_result(job_id)
    return {**record, "result": result}


def get_job_status_only(
    job_id: str,
    requester: dict[str, Any] | None = None,
    application_user_id: int | None = None,
) -> dict[str, Any] | None:
    """Return the lightweight job state used by polling.

    Args:
        job_id: The job identifier.
        requester: The authenticated backend API user dict.
        application_user_id: Optional frontend application user ID for
            owner-scoped access.

    Returns:
        A dict with ``status``, ``progress``, and ``step`` keys, or ``None``.
    """
    job = get_job(job_id, requester, application_user_id)
    if job is None:
        return None
    return {"status": job["status"], "progress": job["progress"], "step": job["step"]}


def list_jobs_for_user(
    application_user_id: int,
    backend_owner_id: int | None,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Return the most recent jobs belonging to a specific application user.

    Args:
        application_user_id: Frontend application user ID to scope by.
        backend_owner_id: Backend API user ID that owns the jobs.
        limit: Maximum number of jobs to return.

    Returns:
        A list of job dicts with user-queue fields, newest first.
    """
    connection = get_connection()
    try:
        rows = connection.execute(
            """
            SELECT job_id, report_name, status, progress, step,
                   forecast_horizon, forced_model, queued_at, started_at,
                   completed_at, error
            FROM forecast_jobs
            WHERE backend_owner_id IS ? AND application_user_id = ?
            ORDER BY queued_at DESC, rowid DESC LIMIT ?
            """,
            (backend_owner_id, application_user_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


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
            "UPDATE forecast_jobs SET progress = ?, step = ? "
            "WHERE job_id = ? AND status = 'running'",
            (pct, step, job_id),
        )
        connection.commit()
    finally:
        connection.close()


def _set_job_error(job_id: str, error: str) -> None:
    """Mark a job terminally failed.

    The transition is conditional on the job being in a non-terminal state
    (``pending``, ``running``, or ``cancelling``) to prevent racing with
    completion or cancellation.
    """
    connection = get_connection()
    try:
        connection.execute(
            """
            UPDATE forecast_jobs
            SET status = 'error', step = 'Pipeline failed.', error = ?,
                completed_at = datetime('now')
            WHERE job_id = ? AND status IN ('pending', 'running', 'cancelling')
            """,
            (error, job_id),
        )
        connection.commit()
    finally:
        connection.close()


# ── Cooperative cancellation ──────────────────────────────────────────────────


def request_cancel(
    job_id: str,
    requester: dict[str, Any] | None = None,
    application_user_id: int | None = None,
) -> str:
    """Request cancellation of a job.

    Cancellation is cooperative, not immediate:

    - **Pending jobs** transition atomically from ``pending`` to ``cancelled``.
    - **Running jobs** set ``cancel_requested = 1`` and transition to
      ``cancelling``.  The job continues to count as running until the
      worker acknowledges by transitioning to ``cancelled`` after the
      pipeline raises :class:`JobCancelledError` at the next stage boundary.
    - **Cancelling jobs** are a no-op (idempotent).
    - **Terminal jobs** (``done``, ``error``, ``cancelled``) return
      ``'conflict'``.

    Args:
        job_id: The job identifier.
        requester: The authenticated backend API user dict.
        application_user_id: Optional frontend application user ID for
            owner-scoped cancellation.

    Returns:
        One of ``'cancelled'``, ``'cancelling'``, ``'conflict'``, or
        ``'not_found'``.
    """
    connection = get_connection()
    file_id_to_release: str | None = None
    cancel_result = "not_found"
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM forecast_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            connection.rollback()
            cancel_result = "not_found"
        else:
            record = dict(row)
            if not _is_owner_authorized(record, requester, application_user_id):
                connection.rollback()
                cancel_result = "not_found"
            elif record["status"] == "pending":
                cursor = connection.execute(
                    """
                    UPDATE forecast_jobs
                    SET status = 'cancelled', step = 'Cancelled.',
                        completed_at = datetime('now')
                    WHERE job_id = ? AND status = 'pending'
                    """,
                    (job_id,),
                )
                connection.commit()
                cancel_result = "cancelled"
                if cursor.rowcount:
                    file_id_to_release = str(record["file_id"])
            elif record["status"] == "running":
                connection.execute(
                    """
                    UPDATE forecast_jobs
                    SET cancel_requested = 1, status = 'cancelling',
                        step = 'Cancelling…'
                    WHERE job_id = ? AND status = 'running'
                    """,
                    (job_id,),
                )
                connection.commit()
                cancel_result = "cancelling"
            elif record["status"] == "cancelling":
                connection.rollback()
                cancel_result = "cancelling"
            else:
                connection.rollback()
                cancel_result = "conflict"
    finally:
        connection.close()
    if file_id_to_release is not None:
        release_file(file_id_to_release)
    return cancel_result


def _is_cancel_requested(job_id: str) -> bool:
    """Return ``True`` when ``cancel_requested`` is set on the job row."""
    connection = get_connection()
    try:
        row = connection.execute(
            "SELECT cancel_requested FROM forecast_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        return bool(row and int(row["cancel_requested"]))
    finally:
        connection.close()


def _set_job_cancelled(job_id: str) -> None:
    """Transition a cancelling/running job to ``cancelled``.

    The transition is conditional on the job being in ``cancelling`` or
    ``running`` state to prevent racing with completion.
    """
    connection = get_connection()
    try:
        connection.execute(
            """
            UPDATE forecast_jobs
            SET status = 'cancelled', step = 'Cancelled.',
                completed_at = datetime('now')
            WHERE job_id = ? AND status IN ('cancelling', 'running')
            """,
            (job_id,),
        )
        connection.commit()
    finally:
        connection.close()


# ── Durable result persistence ────────────────────────────────────────────────


def _ensure_results_dir() -> None:
    """Create the results directory if it does not exist."""
    os.makedirs(_RESULTS_DIR, exist_ok=True)


def _result_file_path(job_id: str) -> str:
    """Return the on-disk path for a job's durable result JSON."""
    return os.path.join(_RESULTS_DIR, f"{job_id}.json")


def _persist_result(job_id: str, result: dict[str, Any]) -> None:
    """Write the completed analysis result to disk as JSON.

    Args:
        job_id: The job identifier.
        result: The ``AnalysisResponse.model_dump()`` dict.
    """
    _ensure_results_dir()
    path = _result_file_path(job_id)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)
    directory_fd = os.open(_RESULTS_DIR, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _load_result(job_id: str) -> dict[str, Any] | None:
    """Load a durable result from disk, or ``None`` if not found."""
    path = _result_file_path(job_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to load durable result for job_id=%s", job_id)
        return None


def _delete_result_file(job_id: str) -> None:
    """Delete the durable result file for a job, ignoring missing files."""
    path = _result_file_path(job_id)
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _cleanup_orphaned_result_files() -> None:
    """Delete result files for jobs that are not in the ``done`` state.

    Called on backend restart to remove results orphaned by jobs interrupted
    during processing.
    """
    if not os.path.isdir(_RESULTS_DIR):
        return
    connection = get_connection()
    try:
        done_ids = {
            str(row["job_id"])
            for row in connection.execute(
                "SELECT job_id FROM forecast_jobs WHERE status = 'done'"
            ).fetchall()
        }
    finally:
        connection.close()
    for filename in os.listdir(_RESULTS_DIR):
        if not filename.endswith(".json"):
            continue
        job_id = filename[:-5]
        if job_id not in done_ids:
            _delete_result_file(job_id)


async def job_worker() -> None:
    """Process queue items while respecting each non-admin user's running cap.

    When ``_claim_job`` returns ``None`` the job may have been cancelled while
    pending (in which case it is already marked ``cancelled``) or the user is
    at capacity (in which case it stays ``pending`` and must be re-queued).
    We distinguish the two by checking the job status after the claim fails.
    """
    while True:
        job_id = await JOB_QUEUE.get()
        try:
            job = _claim_job(job_id)
            if job is None:
                # Re-queue only if the job is still pending (capacity issue).
                # If it was cancelled or is terminal, do not re-queue.
                record = _job_record(job_id)
                if record and record["status"] == "pending":
                    JOB_QUEUE.put_nowait(job_id)
                    await asyncio.sleep(0.1)
                continue
            await _run_job(job_id, job)
        finally:
            JOB_QUEUE.task_done()


async def _run_job(job_id: str, job: dict[str, Any]) -> None:
    """Run one claimed job and persist its terminal state.

    Cancellation is cooperative: the pipeline receives a ``cancel_check``
    callback that raises :class:`JobCancelledError` at stage boundaries when
    ``cancel_requested`` is set.  If the pipeline completes before noticing
    the cancellation request, completion wins (the result is valid).
    """
    stored = get_file(str(job["file_id"]))
    if stored is None:
        _set_job_error(job_id, "Uploaded file not found.")
        release_file(str(job["file_id"]))
        return

    def run_pipeline_sync() -> AnalysisResponse:
        return _run_pipeline(
            df=stored["df"],
            file_id=str(job["file_id"]),
            date_col=str(job["date_col"]),
            value_col=str(job["value_col"]),
            freq=stored["freq"],
            forecast_horizon=int(job["forecast_horizon"]),
            forced_model=job["forced_model"],
            user_prompt=job["user_prompt"],
            preflight_options=json.loads(str(job["preflight_options"])),
            chroma_persist_dir=settings.CHROMA_PERSIST_DIR,
            progress_callback=lambda pct, step: _update_job_progress(job_id, pct, step),
            cancel_check=lambda: _is_cancel_requested(job_id),
        )

    try:
        result = await asyncio.to_thread(run_pipeline_sync)
        # Persist the result to disk *before* marking the job done so that
        # ``done`` always implies the result is available durably.
        result_dict = result.model_dump()
        _persist_result(job_id, result_dict)
        _complete_job(job_id)
        with _job_store_lock:
            _job_store.setdefault(job_id, {})["result"] = result_dict
        # Skip RAG indexing if the job was cancelled but the pipeline
        # completed first (the result is still valid and persisted).
        if not _is_cancel_requested(job_id):
            try:
                await asyncio.to_thread(
                    index_analysis_results,
                    str(job["file_id"]),
                    job["backend_owner_id"],
                    result,
                    settings.CHROMA_PERSIST_DIR,
                )
            except (RuntimeError, TypeError, ValueError, OSError):
                logger.exception(
                    "Analysis result indexing failed for job_id=%s", job_id
                )
    except JobCancelledError:
        logger.info("Job cancelled by user: job_id=%s", job_id)
        _set_job_cancelled(job_id)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Pipeline failed for job_id=%s", job_id)
        _set_job_error(job_id, str(exc))
    finally:
        release_file(str(job["file_id"]))


def _complete_job(job_id: str) -> None:
    """Mark a job complete in durable storage.

    The transition is conditional on the job being ``running`` or
    ``cancelling`` so that completion wins over a cancellation request that
    arrived while the pipeline was finishing.
    """
    connection = get_connection()
    try:
        connection.execute(
            """
            UPDATE forecast_jobs
            SET status = 'done', progress = 100, step = 'Analysis complete.',
                completed_at = datetime('now')
            WHERE job_id = ? AND status IN ('running', 'cancelling')
            """,
            (job_id,),
        )
        connection.commit()
    finally:
        connection.close()


def _run_pipeline(**kwargs: Any) -> AnalysisResponse:
    """Load and execute the forecasting pipeline only when a job is running."""
    run_pipeline = importlib.import_module("services.pipeline_service").run_pipeline
    return run_pipeline(**kwargs)


def index_analysis_results(
    file_id: str,
    owner_id: int | None,
    analysis_result: AnalysisResponse,
    chroma_persist_dir: str,
) -> None:
    """Index completed analysis results without making indexing job-critical."""
    get_rag_kb = importlib.import_module("services.rag_service").get_rag_kb
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
            metadatas=[
                {
                    "file_id": file_id,
                    "owner_id": str(owner_id or ""),
                    "type": "analysis_result",
                }
            ],
        )
