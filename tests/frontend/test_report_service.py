"""Tests for frontend report persistence helpers."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

import pytest

from data_forecaster.frontend.services import report_service


def _report_db(path: Path) -> sqlite3.Connection:
    schema = (
        Path(__file__).resolve().parents[2]
        / "data_forecaster"
        / "frontend"
        / "db"
        / "schema.sql"
    )
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(schema.read_text(encoding="utf-8"))
    connection.execute("INSERT INTO roles (id, name) VALUES (1, 'admin')")
    connection.execute(
        """
        INSERT INTO users (id, username, password_hash, role_id)
        VALUES (1, 'alice', 'hash', 1)
        """
    )
    connection.execute(
        "INSERT INTO app_config (key, value) VALUES ('max_reports_per_user', '10')"
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


def test_owned_section_edits_persist_without_changing_original(tmp_path, monkeypatch):
    from data_forecaster.frontend.services.report_editing import report_sections, EditConflict
    connection = _report_db(tmp_path / "reports.db")
    monkeypatch.setattr(report_service, "get_db", lambda: connection)
    original = "## Summary\n\nOriginal words."
    report_id = report_service.save_report(1, {"report": original}, "data.csv", 3)
    version = report_sections({"report": original})[0]["version"]
    assert not report_service.edit_report_section_for_user(report_id, 999, "0", "remove", version)
    assert report_service.edit_report_section_for_user(report_id, 1, "0", "save", version, "Summary", "My wording")
    with pytest.raises(EditConflict):
        report_service.edit_report_section_for_user(report_id, 1, "0", "save", version, "Summary", "Stale wording")
    row = dict(connection.execute("SELECT * FROM forecast_reports WHERE id = ?", (report_id,)).fetchone())
    decoded = report_service._decode_report(row)
    assert decoded["original_report"] == original
    assert "My wording" in decoded["report"]
    assert "Original words" not in decoded["report"]
    assert not connection.in_transaction
