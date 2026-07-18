"""Tests for report title resolution, persistence handoff, and presentation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import ValidationError

from data_forecaster.frontend.services.report_identity import (
    ReportTitleValidationError,
    normalize_report_title,
    report_download_filename,
    resolve_report_identity,
)
from schemas import AnalyzeRequest

_ROUTES_SOURCE = Path("data_forecaster/frontend/blueprints/main/routes.py")
_SETUP_TEMPLATE = Path("data_forecaster/frontend/templates/main/forecast_setup.html")


def test_report_title_normalization_uses_custom_and_default_values() -> None:
    assert normalize_report_title("  Q4 report  ", "sales.csv") == "Q4 report"
    assert normalize_report_title("   ", "sales.csv") == ("Forecast Report — sales")
    assert normalize_report_title("", "") == "Forecast Report — data"
    assert len(normalize_report_title("", f"{'x' * 250}.csv")) == 200


def test_report_title_normalization_rejects_over_length_value() -> None:
    with pytest.raises(ReportTitleValidationError, match="200"):
        normalize_report_title("x" * 201, "sales.csv")


def test_setup_and_analyze_routes_wire_report_title_through_shared_helper() -> None:
    routes_source = _ROUTES_SOURCE.read_text(encoding="utf-8")
    setup_template = _SETUP_TEMPLATE.read_text(encoding="utf-8")

    assert 'id="inp-report-title"' in setup_template
    assert 'maxlength="200"' in setup_template
    assert "setup_state.report_title" in setup_template
    assert 'session["report_title"] = raw_title' in routes_source
    assert "report_name = normalize_report_title(" in routes_source
    assert '"report_name": report_name' in routes_source


def test_backend_request_schema_rejects_over_length_report_name() -> None:
    with pytest.raises(ValidationError):
        AnalyzeRequest(
            file_id="file-1",
            forecast_horizon=3,
            report_name="x" * 201,
        )


def test_both_finalization_paths_use_durable_job_title() -> None:
    routes_source = _ROUTES_SOURCE.read_text(encoding="utf-8")

    assert routes_source.count('job_record.get("report_name")') == 2
    assert routes_source.count("report_title=report_title") == 2


def test_saved_title_overrides_creation_title_without_changing_other_metadata() -> None:
    identity = resolve_report_identity(
        {
            "title": "Renamed report",
            "created_at": "2026-07-19 03:00:00",
            "executive_report": {
                "metadata": {
                    "title": "Original report",
                    "prepared_by": "alice",
                    "generated_at": "2026-07-18T01:02:00+00:00",
                }
            },
        },
        "sales.csv",
        "bob",
    )

    assert identity == {
        "title": "Renamed report",
        "prepared_by": "alice",
        "creation_date": "July 18, 2026 at 01:02 UTC",
    }


def test_legacy_identity_uses_owner_and_saved_date_fallbacks() -> None:
    identity = resolve_report_identity(
        {
            "title": "Legacy report",
            "created_at": "2026-07-17 22:30:00",
            "executive_report": {"metadata": {}},
        },
        "sales.csv",
        "alice",
    )

    assert identity["title"] == "Legacy report"
    assert identity["prepared_by"] == "alice"
    assert identity["creation_date"] == "July 17, 2026 at 22:30 UTC"


def test_report_download_filename_is_bounded_and_path_safe() -> None:
    filename = report_download_filename("../Q4 / Montréal report " + "x" * 200)

    assert filename.endswith(".pdf")
    assert "/" not in filename
    assert "\\" not in filename
    assert len(filename) <= 104


def test_report_template_escapes_identity_values() -> None:
    environment = Environment(
        loader=FileSystemLoader("data_forecaster/frontend/templates"),
        autoescape=select_autoescape(("html",)),
    )
    environment.globals.update(
        csrf_token=lambda: "",
        current_user=SimpleNamespace(
            is_authenticated=False, is_admin=False, username=""
        ),
        get_flashed_messages=lambda **_kwargs: [],
        request=SimpleNamespace(endpoint="", blueprint=""),
        session={},
        url_for=lambda *_args, **_kwargs: "#",
    )

    html = environment.get_template("main/report.html").render(
        er=None,
        segments=[],
        llm_fallback=False,
        export_url="#",
        custom_settings=[],
        report_identity={
            "title": "<script>alert(1)</script>",
            "prepared_by": "Alice & Bob",
            "creation_date": "July 18, 2026 at 01:02 UTC",
        },
    )

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "Alice &amp; Bob" in html
    assert "July 18, 2026 at 01:02 UTC" in html
