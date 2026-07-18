"""Tests for frontend report persistence helpers."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

import pytest

from data_forecaster.frontend.services import report_service


def _report_db(path: Path) -> sqlite3.Connection:
    schema = (
        Path(__file__).resolve().parents[1]
        / "data_forecaster"
        / "frontend"
        / "db"
        / "schema.sql"
    )
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(schema.read_text(encoding="utf-8"))
    connection.execute("INSERT OR IGNORE INTO roles (id, name) VALUES (1, 'admin')")
    connection.execute("""
        INSERT INTO users (id, username, password_hash, role_id)
        VALUES (1, 'alice', 'hash', 1)
        """)
    connection.execute(
        "INSERT OR IGNORE INTO app_config (key, value) "
        "VALUES ('max_reports_per_user', '10')"
    )
    connection.commit()
    return connection


def test_save_report_rolls_back_serialization_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any exception after BEGIN IMMEDIATE should roll back the transaction."""
    connection = _report_db(tmp_path / "reports.db")
    monkeypatch.setattr(report_service, "get_db", lambda: connection)

    def fail_dumps(_value: Any) -> str:
        raise TypeError("not serializable")

    monkeypatch.setattr(report_service.json, "dumps", fail_dumps)

    with pytest.raises(TypeError, match="not serializable"):
        report_service.save_report(
            user_id=1,
            result={"report": "Report"},
            source_filename="forecast.csv",
            forecast_horizon=3,
        )

    assert not connection.in_transaction
    count = connection.execute("SELECT COUNT(*) FROM forecast_reports").fetchone()[0]
    assert count == 0


def test_batch_report_linkage_uses_one_lightweight_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queue linkage fetches only IDs for all visible jobs in one query."""
    calls: list[tuple[str, tuple[Any, ...]]] = []

    def fake_query(sql: str, args: tuple[Any, ...] = (), **_kwargs: Any) -> Any:
        calls.append((sql, args))
        return [
            {"job_id": "job-1", "id": 10},
            {"job_id": "job-2", "id": 11},
        ]

    monkeypatch.setattr(report_service, "query_db", fake_query)

    result = report_service.get_report_ids_by_job_ids(
        ["job-1", "job-2", "job-1"],
        user_id=7,
    )

    assert result == {"job-1": 10, "job-2": 11}
    assert len(calls) == 1
    assert "visual_assets_json" not in calls[0][0]
    assert calls[0][1] == (7, "job-1", "job-2")


def test_save_report_is_idempotent_by_job_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated finalization returns one report without consuming more quota."""
    connection = _report_db(tmp_path / "idempotent-reports.db")
    monkeypatch.setattr(report_service, "get_db", lambda: connection)
    result = {"report": "Ready", "forecast": {"model_used": "ARIMA"}}

    first_id = report_service.save_report(
        user_id=1,
        result=result,
        source_filename="forecast.csv",
        forecast_horizon=3,
        job_id="job-1",
    )
    second_id = report_service.save_report(
        user_id=1,
        result=result,
        source_filename="forecast.csv",
        forecast_horizon=3,
        job_id="job-1",
    )

    count = connection.execute("SELECT COUNT(*) FROM forecast_reports").fetchone()[0]
    assert first_id == second_id
    assert count == 1


def test_custom_report_title_persists_and_rename_remains_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Creation title is persisted and the existing rename flow still wins."""
    connection = _report_db(tmp_path / "titled-reports.db")
    monkeypatch.setattr(report_service, "get_db", lambda: connection)

    report_id = report_service.save_report(
        user_id=1,
        result={"report": "Ready"},
        source_filename="sales.csv",
        forecast_horizon=3,
        job_id="job-title",
        report_title="Q4 report",
    )
    created = connection.execute(
        "SELECT title, created_at FROM forecast_reports WHERE id = ?", (report_id,)
    ).fetchone()

    assert created["title"] == "Q4 report"
    assert created["created_at"]
    assert report_service.rename_report_for_user(report_id, 1, "Board report")
    renamed = connection.execute(
        "SELECT title FROM forecast_reports WHERE id = ?", (report_id,)
    ).fetchone()
    assert renamed["title"] == "Board report"


def test_report_title_falls_back_and_rejects_over_length_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _report_db(tmp_path / "title-validation.db")
    monkeypatch.setattr(report_service, "get_db", lambda: connection)

    report_id = report_service.save_report(
        user_id=1,
        result={"report": "Ready"},
        source_filename="sales.csv",
        forecast_horizon=3,
        report_title="   ",
    )
    row = connection.execute(
        "SELECT title FROM forecast_reports WHERE id = ?", (report_id,)
    ).fetchone()
    assert row["title"] == "Forecast Report — sales"

    with pytest.raises(ValueError, match="200"):
        report_service.save_report(
            user_id=1,
            result={"report": "Ready"},
            source_filename="sales.csv",
            forecast_horizon=3,
            report_title="x" * 201,
        )


def test_saved_report_restores_complete_analysis_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saved reports retain the data required by every analysis tab."""
    connection = _report_db(tmp_path / "complete-reports.db")
    monkeypatch.setattr(report_service, "get_db", lambda: connection)
    result = {
        "report": "Complete report",
        "validation": {"reasoning_steps": ["Validated dates"]},
        "statistical": {"reasoning_steps": ["Detected seasonality"]},
        "model_selection": {
            "selected_model": "ARIMA",
            "reasoning_steps": ["Lowest validation error"],
        },
        "forecast": {
            "model_used": "ARIMA",
            "forecast_dates": ["2026-08-01"],
            "forecast": [42.0],
            "reasoning_steps": ["Generated one period"],
        },
        "report_reasoning": ["Summarized the forecast"],
        "chart_forecast": {"data": []},
    }

    report_id = report_service.save_report(
        user_id=1,
        result=result,
        source_filename="forecast.csv",
        forecast_horizon=1,
        job_id="job-complete",
    )
    row = connection.execute(
        """
        SELECT id, job_id, title, source_filename, model_used, forecast_horizon,
               report_markdown, executive_report_json, visual_assets_json,
               analysis_result_json, custom_settings_json, llm_fallback, created_at
        FROM forecast_reports WHERE id = ?
        """,
        (report_id,),
    ).fetchone()

    restored = report_service._decode_report(dict(row))

    assert restored["validation"] == result["validation"]
    assert restored["statistical"] == result["statistical"]
    assert restored["model_selection"] == result["model_selection"]
    assert restored["forecast"] == result["forecast"]
    assert restored["report_reasoning"] == result["report_reasoning"]
    assert restored["chart_forecast"] == result["chart_forecast"]
