"""Persistence helpers for user-owned final forecast reports."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from db.db import get_db, query_db


class ReportLimitError(ValueError):
    """Raised when a user has reached the configured report limit."""


_VISUAL_FIELDS: tuple[str, ...] = (
    "chart_historical",
    "chart_stl",
    "chart_acf_pacf",
    "chart_forecast",
    "chart_model_comparison",
    "chart_historical_png",
    "chart_stl_png",
    "chart_forecast_png",
    "chart_model_comparison_png",
)


def _report_limit(connection: sqlite3.Connection) -> int:
    """Return the configured positive per-user report limit."""
    row = connection.execute(
        "SELECT value FROM app_config WHERE key = 'max_reports_per_user'"
    ).fetchone()
    try:
        limit = int(row["value"]) if row else 10
    except (KeyError, TypeError, ValueError):
        limit = 10
    return max(limit, 1)


def _report_title(filename: str) -> str:
    """Build a stable display title from a source filename."""
    base_name = filename.rsplit(".", 1)[0] if "." in filename else filename
    prefix = "Forecast Report — "
    usable_name = base_name.strip() or "data"
    return f"{prefix}{usable_name[: 200 - len(prefix)].rstrip()}"


def save_report(
    user_id: int,
    result: dict[str, Any],
    source_filename: str,
    forecast_horizon: int | None,
    custom_settings: list[dict[str, str]] | None = None,
    job_id: str | None = None,
    report_title: str | None = None,
) -> int:
    """Atomically save a final report if its owner remains below the cap.

    When ``job_id`` is provided, the save is **idempotent**: if a report
    with the same ``job_id`` already exists, its ID is returned without
    creating a duplicate or re-checking the report limit.  This makes the
    operation safe for repeated polling or multi-tab finalization attempts.

    Args:
        user_id: Authenticated application user who owns the report.
        result: Completed backend result containing final report artifacts.
        source_filename: Original dataset filename retained as display metadata.
        forecast_horizon: Requested number of forecast periods.
        custom_settings: Optional user-selected report settings to persist as JSON.
        job_id: Optional backend job ID for idempotent linking.
        report_title: Resolved job title; falls back to the dataset-based title.

    Returns:
        The newly created or existing report ID.

    Raises:
        ReportLimitError: When the owner has reached the configured limit and
            no report for this ``job_id`` exists yet.
    """
    visual_assets = {field: result.get(field) for field in _VISUAL_FIELDS}
    analysis_result = {
        key: value
        for key, value in result.items()
        if key not in _VISUAL_FIELDS
        and key not in {"report", "executive_report", "llm_fallback"}
    }
    executive_report = result.get("executive_report")
    model_used = (result.get("forecast") or {}).get("model_used")
    resolved_title = str(report_title or "").strip() or _report_title(source_filename)
    if len(resolved_title) > 200:
        raise ValueError("Report title must be 200 characters or fewer.")
    connection = get_db()
    transaction_started = False
    try:
        connection.execute("BEGIN IMMEDIATE")
        transaction_started = True

        # Idempotency: if a report for this job_id already exists, return it.
        if job_id is not None:
            existing = connection.execute(
                "SELECT id FROM forecast_reports WHERE job_id = ? AND user_id = ?",
                (job_id, user_id),
            ).fetchone()
            if existing is not None:
                connection.rollback()
                transaction_started = False
                return int(existing["id"])

        stored_count = connection.execute(
            "SELECT COUNT(*) AS count FROM forecast_reports WHERE user_id = ?",
            (user_id,),
        ).fetchone()["count"]
        if int(stored_count) >= _report_limit(connection):
            connection.rollback()
            transaction_started = False
            raise ReportLimitError("Report limit reached.")
        cursor = connection.execute(
            """
            INSERT INTO forecast_reports (
                user_id, job_id, title, source_filename, model_used,
                forecast_horizon, report_markdown, executive_report_json,
                visual_assets_json, analysis_result_json, custom_settings_json,
                llm_fallback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                job_id,
                resolved_title,
                source_filename,
                str(model_used) if model_used else None,
                forecast_horizon,
                str(result.get("report") or "Report not available."),
                json.dumps(executive_report) if executive_report is not None else None,
                json.dumps(visual_assets),
                json.dumps(analysis_result),
                json.dumps(custom_settings or []),
                int(bool(result.get("llm_fallback", False))),
            ),
        )
        connection.commit()
        transaction_started = False
        return int(cursor.lastrowid)
    except Exception:
        if transaction_started:
            connection.rollback()
        raise


def get_report_by_job_id(job_id: str, user_id: int) -> dict[str, Any] | None:
    """Return lightweight report linkage for an owner-scoped job ID.

    Args:
        job_id: The backend job ID to look up.
        user_id: The authenticated application user ID (owner scope).

    Returns:
        A dict containing the report ID, or ``None`` if not found or not owned.
    """
    row = query_db(
        "SELECT id FROM forecast_reports WHERE job_id = ? AND user_id = ?",
        (job_id, user_id),
        one=True,
    )
    return row if isinstance(row, dict) else None


def get_report_ids_by_job_ids(job_ids: list[str], user_id: int) -> dict[str, int]:
    """Return report IDs for multiple owner-scoped jobs in one query.

    Args:
        job_ids: Backend job IDs whose report linkage should be checked.
        user_id: Authenticated application user ID that owns the reports.

    Returns:
        A mapping from backend job ID to frontend report ID.
    """
    unique_job_ids = list(dict.fromkeys(job_id for job_id in job_ids if job_id))
    if not unique_job_ids:
        return {}
    placeholders = ",".join("?" for _ in unique_job_ids)
    rows = query_db(
        f"""
        SELECT job_id, id FROM forecast_reports
        WHERE user_id = ? AND job_id IN ({placeholders})
        """,
        (user_id, *unique_job_ids),
    )
    if not isinstance(rows, list):
        return {}
    return {str(row["job_id"]): int(row["id"]) for row in rows}


def get_report_for_user(report_id: int, user_id: int) -> dict[str, Any] | None:
    """Return one report only when it belongs to the requesting user."""
    row = query_db(
        """
        SELECT id, job_id, title, source_filename, model_used, forecast_horizon,
               report_markdown, executive_report_json, visual_assets_json,
               analysis_result_json, custom_settings_json, llm_fallback, created_at
        FROM forecast_reports WHERE id = ? AND user_id = ?
        """,
        (report_id, user_id),
        one=True,
    )
    return _decode_report(row) if isinstance(row, dict) else None


def _decode_report(row: dict[str, Any]) -> dict[str, Any]:
    """Decode stored JSON fields into the rendering result shape."""
    report = dict(row)
    full_result = (
        json.loads(report.pop("analysis_result_json"))
        if report.get("analysis_result_json")
        else {}
    )
    report["executive_report"] = (
        json.loads(report.pop("executive_report_json"))
        if report.get("executive_report_json")
        else None
    )
    visual_assets = json.loads(report.pop("visual_assets_json"))
    report["custom_settings"] = (
        json.loads(report.pop("custom_settings_json"))
        if report.get("custom_settings_json")
        else []
    )
    report["report"] = report.pop("report_markdown")
    # Saved metadata remains authoritative, while the complete result supplies
    # the validation, statistics, model, forecast, and reasoning tab payloads.
    full_result.update(visual_assets)
    full_result.update(report)
    return full_result


def list_reports_for_user(user_id: int) -> list[dict[str, Any]]:
    """List lightweight report metadata for one user, newest first."""
    rows = query_db(
        """
        SELECT id, title, source_filename, model_used, forecast_horizon, created_at
        FROM forecast_reports WHERE user_id = ? ORDER BY created_at DESC, id DESC
        """,
        (user_id,),
    )
    return rows if isinstance(rows, list) else []


def report_usage_for_user(user_id: int) -> tuple[int, int]:
    """Return a user's stored-report count and the current global limit."""
    connection = get_db()
    count = connection.execute(
        "SELECT COUNT(*) AS count FROM forecast_reports WHERE user_id = ?", (user_id,)
    ).fetchone()["count"]
    return int(count), _report_limit(connection)


def delete_report_for_user(report_id: int, user_id: int) -> bool:
    """Delete one report only when it belongs to the requesting user."""
    connection = get_db()
    cursor = connection.execute(
        "DELETE FROM forecast_reports WHERE id = ? AND user_id = ?",
        (report_id, user_id),
    )
    connection.commit()
    return cursor.rowcount == 1


def rename_report_for_user(report_id: int, user_id: int, title: str) -> bool:
    """Rename one report only when it belongs to the requesting user."""
    connection = get_db()
    cursor = connection.execute(
        "UPDATE forecast_reports SET title = ? WHERE id = ? AND user_id = ?",
        (title, report_id, user_id),
    )
    connection.commit()
    return cursor.rowcount == 1


def list_report_owners() -> list[dict[str, Any]]:
    """List application users who currently own at least one report."""
    rows = query_db("""
        SELECT u.id, u.username, COUNT(fr.id) AS report_count
        FROM users u JOIN forecast_reports fr ON fr.user_id = u.id
        GROUP BY u.id, u.username ORDER BY u.username COLLATE NOCASE
        """)
    return rows if isinstance(rows, list) else []


def delete_report_for_admin(report_id: int) -> bool:
    """Delete one report as an administrator."""
    connection = get_db()
    cursor = connection.execute(
        "DELETE FROM forecast_reports WHERE id = ?", (report_id,)
    )
    connection.commit()
    return cursor.rowcount == 1


def delete_all_reports_for_admin(user_id: int) -> int:
    """Delete every report owned by the selected user."""
    connection = get_db()
    cursor = connection.execute(
        "DELETE FROM forecast_reports WHERE user_id = ?", (user_id,)
    )
    connection.commit()
    return cursor.rowcount
