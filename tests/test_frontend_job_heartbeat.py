"""Frontend regressions for long-running job reassurance and reconnecting."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import requests
from flask import Flask, session


def test_transient_status_timeout_preserves_active_session(monkeypatch) -> None:
    frontend_root = Path(__file__).resolve().parents[1] / "data_forecaster" / "frontend"
    saved_services = {
        name: module
        for name, module in sys.modules.items()
        if name == "services" or name.startswith("services.")
    }
    for name in saved_services:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(frontend_root))
    try:
        from blueprints.main import routes
    finally:
        sys.path.remove(str(frontend_root))

    class TimeoutClient:
        def get_job_status_lightweight(self, *_args, **_kwargs):
            raise requests.exceptions.Timeout("test timeout")

    app = Flask(__name__)
    app.secret_key = "test-secret"
    monkeypatch.setattr(routes, "get_api_client", lambda: TimeoutClient())
    monkeypatch.setattr(routes, "current_user", SimpleNamespace(id=7))

    try:
        with app.test_request_context("/api/jobs/status"):
            session["job_id"] = "long-job"
            session["job_running"] = True
            session["job_progress"] = 60
            session["job_step"] = "Evaluating SARIMA"

            response = routes.api_job_status.__wrapped__()
            payload = response.get_json()

            assert response.status_code == 504
            assert payload["transient"] is True
            assert payload["done"] is False
            assert session["job_id"] == "long-job"
            assert session["job_running"] is True
    finally:
        for name in list(sys.modules):
            if name == "services" or name.startswith("services."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_services)


def test_reassurance_markup_and_polling_states_are_present() -> None:
    root = Path(__file__).resolve().parents[1] / "data_forecaster" / "frontend"
    progress_html = (root / "templates/main/forecast_progress.html").read_text()
    jobs_html = (root / "templates/main/jobs.html").read_text()
    polling_js = (root / "static/js/polling.js").read_text()
    jobs_js = (root / "static/js/jobs.js").read_text()

    assert 'id="job-heartbeat"' in progress_html
    assert 'id="heartbeat-elapsed"' in progress_html
    assert 'id="poll-reconnecting"' in progress_html
    assert 'id="jobs-connection-status"' in jobs_html
    for state in ("active", "delayed", "stale", "terminal"):
        assert state in polling_js
    assert "MAX_BACKOFF_MS" in polling_js
    assert "forecast continues in the background" in polling_js
    assert "setConnectionStatus(true)" in jobs_js
