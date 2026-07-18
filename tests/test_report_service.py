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
    connection.execute(
        """
        INSERT INTO users (id, username, password_hash, role_id)
        VALUES (1, 'alice', 'hash', 1)
        """
    )
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
