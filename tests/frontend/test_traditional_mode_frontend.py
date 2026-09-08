"""Traditional Forecasting mode — frontend behaviour tests.

Covers the setup wizard's "Use Traditional Forecasting without an LLM"
branch (no provider test or save), the deployment-wide disable in the
chat/setup routes, the per-run toggle plumbing into the analyze payload,
mode labelling from the *actual* job result, and the saved-report
round-trip of ``traditional_mode``.

All HTTP calls to the FastAPI backend are mocked by patching the
``requests`` module attributes used by ``services.api_client``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "data_forecaster" / "backend"
FRONTEND_ROOT = REPO_ROOT / "data_forecaster" / "frontend"
if str(FRONTEND_ROOT) in sys.path:
    sys.path.remove(str(FRONTEND_ROOT))
sys.path.insert(0, str(FRONTEND_ROOT))
sys.modules.pop("services", None)

import config as frontend_config  # noqa: E402
from app import create_app  # noqa: E402
from db.db import execute_db  # noqa: E402
from services import api_client as frontend_api_client  # noqa: E402
import blueprints.main.routes as main_routes  # noqa: E402

sys.modules.pop("services", None)
if str(FRONTEND_ROOT) in sys.path:
    sys.path.remove(str(FRONTEND_ROOT))
if str(BACKEND_ROOT) in sys.path:
    sys.path.remove(str(BACKEND_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(1, str(FRONTEND_ROOT))

_BACKEND_URL = "http://backend:8000"


class _FakeResponse:
    """Minimal stand-in for :class:`requests.Response`."""

    def __init__(self, status_code: int, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.fixture
def backend_state() -> dict[str, Any]:
    """Mutable fake-backend state shared with the request stubs."""
    return {
        "setup_complete": False,
        "llm_config": {
            "provider": "gemini",
            "model": "gemini-1.5-flash",
            "base_url": None,
            "temperature": 0.1,
            "api_key_set": False,
            "configured": False,
            "llm_enabled": True,
        },
        "put_enabled_calls": [],
        "put_config_calls": [],
        "post_test_calls": [],
        "analyze_calls": [],
        "enabled_error": False,
    }


@pytest.fixture
def mock_backend(
    monkeypatch: pytest.MonkeyPatch,
    backend_state: dict[str, Any],
) -> dict[str, Any]:
    """Patch the requests module used by the API client with a fake backend."""

    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        if url.endswith("/setup/status"):
            return _FakeResponse(
                200,
                {
                    "setup_complete": backend_state["setup_complete"],
                    "admin_exists": True,
                    "llm_configured": backend_state["llm_config"]["configured"],
                    "llm_enabled": backend_state["llm_config"]["llm_enabled"],
                    "models_enabled": 5,
                },
            )
        if "/status" in url and "/jobs/" in url:
            return _FakeResponse(
                200, {"status": "done", "progress": 100, "step": "Complete"}
            )
        if "/jobs/" in url:
            return _FakeResponse(
                200,
                {
                    "result": backend_state.get(
                        "job_result",
                        {
                            "traditional_mode": True,
                            "llm_fallback": False,
                            "forecast": {"model_used": "ARIMA"},
                        },
                    )
                },
            )
        if url.endswith("/config/llm"):
            return _FakeResponse(200, backend_state["llm_config"])
        return _FakeResponse(404, {"detail": "Not found"})

    def fake_post(
        url: str, json: dict[str, Any] | None = None, **kwargs: Any
    ) -> _FakeResponse:
        if url.endswith("/config/llm/test"):
            backend_state["post_test_calls"].append(json or {})
            return _FakeResponse(200, {"ok": True, "message": "Test passed."})
        if url.endswith("/analyze"):
            backend_state["analyze_calls"].append(json or {})
            return _FakeResponse(202, {"job_id": "job-1", "status": "pending"})
        return _FakeResponse(404, {"detail": "Not found"})

    def fake_put(
        url: str, json: dict[str, Any] | None = None, **kwargs: Any
    ) -> _FakeResponse:
        if url.endswith("/config/llm/enabled"):
            backend_state["put_enabled_calls"].append(json or {})
            if backend_state["enabled_error"]:
                return _FakeResponse(400, {"detail": "Switch rejected."})
            backend_state["llm_config"]["llm_enabled"] = bool(
                (json or {}).get("enabled")
            )
            return _FakeResponse(200, backend_state["llm_config"])
        if url.endswith("/config/llm"):
            backend_state["put_config_calls"].append(json or {})
            backend_state["llm_config"]["configured"] = True
            return _FakeResponse(200, backend_state["llm_config"])
        return _FakeResponse(404, {"detail": "Not found"})

    monkeypatch.setattr(frontend_api_client.requests, "get", fake_get)
    monkeypatch.setattr(frontend_api_client.requests, "post", fake_post)
    monkeypatch.setattr(frontend_api_client.requests, "put", fake_put)
    return backend_state


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a testing Flask app with an isolated database."""
    monkeypatch.setattr(
        frontend_config.TestingConfig, "DATABASE", str(tmp_path / "frontend.db")
    )
    application = create_app("testing")
    application.config["BACKEND_URL"] = _BACKEND_URL
    return application


@pytest.fixture
def client(app, mock_backend: dict[str, Any]):
    """Test client with the fake backend active."""
    return app.test_client()


@pytest.fixture
def admin_client(app, mock_backend: dict[str, Any]):
    """Test client logged in as the seeded admin with setup complete."""
    mock_backend["setup_complete"] = True
    with app.app_context():
        execute_db("UPDATE users SET must_change_password = 0 WHERE username = 'admin'")
    test_client = app.test_client()
    with test_client.session_transaction() as sess:
        sess["_user_id"] = "1"
        sess["user_session_version"] = 0
    return test_client


# ── Setup wizard: Traditional Forecasting branch ─────────────────────────────


class TestWizardTraditionalBranch:
    """The checkbox finishes step 2 with no provider test, save, or key."""

    def test_selection_advances_without_provider_configuration(
        self, client, backend_state: dict[str, Any]
    ) -> None:
        client.post("/setup/backend", data={"base_url": _BACKEND_URL})
        resp = client.post("/setup/llm", data={"traditional_forecasting": "y"})

        assert resp.status_code == 302
        assert "/setup/auth" in resp.headers["Location"]
        # The switch was flipped off — and no provider test or save ran.
        assert backend_state["put_enabled_calls"] == [{"enabled": False}]
        assert backend_state["post_test_calls"] == []
        assert backend_state["put_config_calls"] == []
        assert backend_state["llm_config"]["configured"] is False
        with client.session_transaction() as sess:
            assert sess.get("setup_llm_ok") is True

    def test_backend_failure_keeps_user_on_step(
        self, client, backend_state: dict[str, Any]
    ) -> None:
        backend_state["enabled_error"] = True
        client.post("/setup/backend", data={"base_url": _BACKEND_URL})
        resp = client.post("/setup/llm", data={"traditional_forecasting": "y"})

        assert resp.status_code == 200
        assert b"Could not save Traditional Forecasting selection" in resp.data
        with client.session_transaction() as sess:
            assert "setup_llm_ok" not in sess

    def test_unchecked_form_still_requires_model(
        self, client, backend_state: dict[str, Any]
    ) -> None:
        """Without the checkbox, provider validators apply as before."""
        client.post("/setup/backend", data={"base_url": _BACKEND_URL})
        resp = client.post(
            "/setup/llm", data={"provider": "gemini", "temperature": "0.1"}
        )

        assert resp.status_code == 400
        assert backend_state["post_test_calls"] == []
        with client.session_transaction() as sess:
            assert "setup_llm_ok" not in sess


class TestLLMProviderFormValidation:
    """Provider validators are skipped *only* while the CSRF pass still runs."""

    def test_traditional_selection_skips_provider_validators(self, app) -> None:
        from werkzeug.datastructures import MultiDict

        from blueprints.setup.forms import LLMProviderForm

        with app.test_request_context(method="POST"):
            form = LLMProviderForm(
                formdata=MultiDict({"traditional_forecasting": "y"})
            )
            assert form.validate() is True
            assert not form.model.errors
            # The skipped validators are restored afterwards.
            assert form.model.validators

    def test_unchecked_form_requires_model(self, app) -> None:
        from werkzeug.datastructures import MultiDict

        from blueprints.setup.forms import LLMProviderForm

        with app.test_request_context(method="POST"):
            form = LLMProviderForm(formdata=MultiDict({"provider": "gemini"}))
            assert form.validate() is False
            assert form.model.errors

    def test_super_validate_still_runs_when_skipping(
        self, app, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CSRF lives in ``FlaskForm.validate``; the override must call it
        even when provider validators are bypassed."""
        from werkzeug.datastructures import MultiDict
        from flask_wtf import FlaskForm

        import blueprints.setup.forms as setup_forms

        calls: list[int] = []
        original = FlaskForm.validate

        def spy(self, extra_validators=None):
            calls.append(1)
            return original(self, extra_validators)

        monkeypatch.setattr(FlaskForm, "validate", spy)
        with app.test_request_context(method="POST"):
            form = setup_forms.LLMProviderForm(
                formdata=MultiDict({"traditional_forecasting": "y"})
            )
            form.validate()

        assert calls == [1]


# ── Deployment-wide disable in the app routes ────────────────────────────────


class TestGlobalDisableRoutes:
    """Chat and the per-run toggle disappear when AI is disabled."""

    def test_chat_redirects_when_disabled(
        self, admin_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(main_routes, "get_llm_enabled", lambda: False)
        resp = admin_client.get("/chat")

        assert resp.status_code == 302
        assert "/forecast-setup" in resp.headers["Location"]

    def test_chat_renders_when_enabled(
        self, admin_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(main_routes, "get_llm_enabled", lambda: True)
        resp = admin_client.get("/chat")

        assert resp.status_code == 200

    def test_setup_page_hides_toggle_and_auto_when_disabled(
        self, admin_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(main_routes, "get_llm_enabled", lambda: False)
        resp = admin_client.get("/forecast-setup")

        assert resp.status_code == 200
        assert b"chk-traditional" not in resp.data
        assert b"Auto (AI selects)" not in resp.data

    def test_setup_page_shows_toggle_and_auto_when_enabled(
        self, admin_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(main_routes, "get_llm_enabled", lambda: True)
        resp = admin_client.get("/forecast-setup")

        assert resp.status_code == 200
        assert b"chk-traditional" in resp.data
        assert b"Auto (AI selects)" in resp.data
        # The exact product copy for the per-run toggle.
        assert (
            b"Run without AI assistance. Choose a forecast model manually; "
            b"reports use standard templates." in resp.data
        )

    def test_setup_page_renders_saved_traditional_choice(
        self, admin_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A checked per-run toggle persists across page loads."""
        monkeypatch.setattr(main_routes, "get_llm_enabled", lambda: True)
        with admin_client.session_transaction() as sess:
            sess["traditional_mode"] = True
        resp = admin_client.get("/forecast-setup")

        assert b"chk-traditional" in resp.data
        assert b"checked" in resp.data.split(b"chk-traditional")[1]


# ── Per-run toggle plumbing ──────────────────────────────────────────────────


class TestAnalyzePayloadPlumbing:
    """The toggle flows through setup-state and the analyze payload."""

    def test_setup_state_persists_traditional_mode(self, admin_client) -> None:
        resp = admin_client.post(
            "/api/setup-state", json={"traditional_mode": True}
        )
        assert resp.status_code == 200
        with admin_client.session_transaction() as sess:
            assert sess.get("traditional_mode") is True

    def test_analyze_payload_carries_traditional_mode(
        self, admin_client, backend_state: dict[str, Any]
    ) -> None:
        with admin_client.session_transaction() as sess:
            sess["upload_info"] = {"file_id": "file-1", "filename": "data.csv"}
        resp = admin_client.post(
            "/api/analyze",
            json={
                "date_col": "date",
                "value_col": "value",
                "forecast_horizon": 6,
                "model_choice": "ARIMA",
                "traditional_mode": True,
            },
        )

        assert resp.status_code == 202
        assert len(backend_state["analyze_calls"]) == 1
        payload = backend_state["analyze_calls"][0]
        assert payload["traditional_mode"] is True
        assert payload["forced_model"] == "ARIMA"

    def test_analyze_payload_defaults_to_ai_mode(
        self, admin_client, backend_state: dict[str, Any]
    ) -> None:
        with admin_client.session_transaction() as sess:
            sess["upload_info"] = {"file_id": "file-1", "filename": "data.csv"}
        resp = admin_client.post(
            "/api/analyze",
            json={
                "date_col": "date",
                "value_col": "value",
                "forecast_horizon": 6,
                "model_choice": "Auto (AI selects)",
            },
        )

        assert resp.status_code == 202
        payload = backend_state["analyze_calls"][0]
        assert payload["traditional_mode"] is False
        assert payload["forced_model"] is None

    def test_custom_settings_label_traditional_runs(self, app) -> None:
        from flask import session

        with app.test_request_context():
            session["traditional_mode"] = True
            settings = main_routes._custom_settings_from_session()
        assert {
            "label": "Forecast mode",
            "value": "Traditional Forecasting",
        } in settings


# ── Mode labelling from the actual result ────────────────────────────────────


class TestDoneJobModeLabeling:
    """The session label comes from the result, never the live setting."""

    def test_done_job_records_traditional_mode(
        self, admin_client, backend_state: dict[str, Any]
    ) -> None:
        backend_state["job_result"] = {
            "traditional_mode": True,
            "llm_fallback": False,
            "forecast": {"model_used": "ARIMA"},
        }
        with admin_client.session_transaction() as sess:
            sess["job_id"] = "job-1"
            sess["upload_info"] = {"filename": "data.csv"}

        resp = admin_client.get("/api/jobs/status")

        assert resp.status_code == 200
        with admin_client.session_transaction() as sess:
            assert sess.get("traditional_mode") is True
            assert sess.get("llm_fallback") is False

    def test_done_job_keeps_ai_label_after_midrun_global_disable(
        self, admin_client, backend_state: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A run that started in AI mode and fell back stays AI-labelled even
        when the deployment-wide switch is now off."""
        monkeypatch.setattr(main_routes, "get_llm_enabled", lambda: False)
        backend_state["job_result"] = {
            "traditional_mode": False,
            "llm_fallback": True,
            "forecast": {"model_used": "ARIMA"},
        }
        with admin_client.session_transaction() as sess:
            sess["job_id"] = "job-1"
            sess["upload_info"] = {"filename": "data.csv"}

        resp = admin_client.get("/api/jobs/status")

        assert resp.status_code == 200
        with admin_client.session_transaction() as sess:
            assert sess.get("traditional_mode") is False
            assert sess.get("llm_fallback") is True


# ── Saved reports round-trip ──────────────────────────────────────────────────


class TestSavedReportModePersistence:
    """The stored label is independent of the current global setting."""

    @staticmethod
    def _report_db(path: Path):
        import sqlite3

        schema = (
            REPO_ROOT / "data_forecaster" / "frontend" / "db" / "schema.sql"
        ).read_text(encoding="utf-8")
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        connection.executescript(schema)
        connection.execute("INSERT INTO roles (id, name) VALUES (1, 'admin')")
        connection.execute(
            "INSERT INTO users (id, username, password_hash, role_id) "
            "VALUES (1, 'alice', 'hash', 1)"
        )
        connection.execute(
            "INSERT INTO app_config (key, value) "
            "VALUES ('max_reports_per_user', '10')"
        )
        connection.commit()
        return connection

    def test_save_get_round_trip_preserves_traditional_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data_forecaster.frontend.services import report_service

        connection = self._report_db(tmp_path / "reports.db")
        monkeypatch.setattr(report_service, "get_db", lambda: connection)

        def fake_query_db(sql: str, args: Any = (), one: bool = False) -> Any:
            cursor = connection.execute(sql, tuple(args))
            if one:
                row = cursor.fetchone()
                return dict(row) if row is not None else None
            return [dict(row) for row in cursor.fetchall()]

        monkeypatch.setattr(report_service, "query_db", fake_query_db)

        report_id = report_service.save_report(
            1,
            {
                "traditional_mode": True,
                "llm_fallback": False,
                "forecast": {"model_used": "ARIMA"},
                "report": "# markdown",
            },
            "data.csv",
            6,
        )

        stored = report_service.get_report_for_user(report_id, 1)
        assert stored is not None
        assert stored["traditional_mode"] == 1
        assert stored["llm_fallback"] == 0

    def test_ai_runs_store_their_own_label(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An AI-mode run is never relabelled by a later global disable."""
        from data_forecaster.frontend.services import report_service

        connection = self._report_db(tmp_path / "reports.db")
        monkeypatch.setattr(report_service, "get_db", lambda: connection)

        def fake_query_db(sql: str, args: Any = (), one: bool = False) -> Any:
            cursor = connection.execute(sql, tuple(args))
            if one:
                row = cursor.fetchone()
                return dict(row) if row is not None else None
            return [dict(row) for row in cursor.fetchall()]

        monkeypatch.setattr(report_service, "query_db", fake_query_db)

        report_id = report_service.save_report(
            1,
            {
                "traditional_mode": False,
                "llm_fallback": True,
                "forecast": {"model_used": "ARIMA"},
                "report": "# markdown",
            },
            "data.csv",
            6,
        )

        stored = report_service.get_report_for_user(report_id, 1)
        assert stored["traditional_mode"] == 0
        assert stored["llm_fallback"] == 1


# ── Template and client-side contracts ────────────────────────────────────────


class TestModeDisplayContracts:
    """Source-level contracts for the banner and the setup-page script."""

    def test_render_passes_traditional_mode_to_template(self) -> None:
        """``render_analysis_report`` forwards the execution mode."""
        source = (
            REPO_ROOT
            / "data_forecaster"
            / "frontend"
            / "services"
            / "report_rendering.py"
        ).read_text(encoding="utf-8")
        assert "traditional_mode=bool(result.get(" in source

    def test_report_banner_never_double_renders(self) -> None:
        """The traditional banner is an ``elif`` of the fallback warning."""
        source = (
            REPO_ROOT
            / "data_forecaster"
            / "frontend"
            / "templates"
            / "main"
            / "report.html"
        ).read_text(encoding="utf-8")
        assert "{% elif traditional_mode %}" in source
        # The fallback warning comes first: a genuine failure wins over the
        # neutral traditional message and the two never render together.
        assert source.index("{% if llm_fallback %}") < source.index(
            "{% elif traditional_mode %}"
        )
        assert "Traditional Forecasting — generated without AI assistance." in source

    def test_app_js_blocks_auto_when_traditional(self) -> None:
        """The setup-page script blocks Auto + traditional submissions."""
        source = (
            REPO_ROOT
            / "data_forecaster"
            / "frontend"
            / "static"
            / "js"
            / "app.js"
        ).read_text(encoding="utf-8")
        assert "traditionalMode" in source
        assert (
            "Traditional Forecasting requires selecting a specific forecast "
            "model" in source
        )