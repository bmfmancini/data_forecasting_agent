"""Tests for durable job identity propagation into the report pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

_BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
if str(_BACKEND_ROOT) in sys.path:
    sys.path.remove(str(_BACKEND_ROOT))
sys.path.insert(0, str(_BACKEND_ROOT))
sys.modules.pop("services", None)

from services import job_service


def test_worker_passes_durable_title_and_username_to_pipeline() -> None:
    kwargs = job_service._pipeline_kwargs_from_job(
        "job-1",
        {
            "file_id": "file-1",
            "date_col": "date",
            "value_col": "value",
            "forecast_horizon": 3,
            "forced_model": None,
            "user_prompt": None,
            "preflight_options": "{}",
            "report_name": "Q4 report",
            "application_username": "alice",
        },
        {"df": object(), "freq": "MS"},
    )

    assert kwargs["report_title"] == "Q4 report"
    assert kwargs["prepared_by"] == "alice"
