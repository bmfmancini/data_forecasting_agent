"""Tests for the setup wizard blueprint, setup gating, and admin config pages.

All HTTP calls to the FastAPI backend are mocked by patching the
``requests`` module attributes used by ``services.api_client``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "data_forecaster" / "backend"
FRONTEND_ROOT = REPO_ROOT / "data_forecaster" / "frontend"
if str(FRONTEND_ROOT) in sys.path:
    sys.path.remove(str(FRONTEND_ROOT))
sys.path.insert(0, str(FRONTEND_ROOT))
sys.modules.pop("services", None)

from app import create_app  # noqa: E402
from db.crypto import decrypt  # noqa: E402
from db.db import execute_db, query_db  # noqa: E402
from services import api_client as frontend_api_client  # noqa: E402

sys.modules.pop("services", None)
if str(FRONTEND_ROOT) in sys.path:
    sys.path.remove(str(FRONTEND_ROOT))
if str(BACKEND_ROOT) in sys.path:
    sys.path.remove(str(BACKEND_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(1, str(FRONTEND_ROOT))

_BACKEND_URL = "http://backend:8000"

_MODELS: list[dict[str, Any]] = [
    {"name": "Holt-Winters", "display_name": "Holt-Winters", "enabled": True},
    {"name": "ARIMA", "display_name": "ARIMA", "enabled": True},
    {"name": "SARIMA", "display_name": "SARIMA", "enabled": True},
    {"name": "EWMA", "display_name": "EWMA", "enabled": True},
    {"name": "Prophet", "display_name": "Prophet", "enabled": True},
]


class _FakeResponse:
    """Minimal stand-in for :class:`requests.Response`."""

    def __init__(
        self, status_code: int, payload: dict[str, Any] | None = None
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.fixture
def backend_state() -> dict[str, Any]:
    """Mutable fake-backend state shared with the request stubs."""
    return {
        "setup_complete": False,
        "bootstrapped": False,
        "models": [dict(model) for model in _MODELS],
        "llm_config": {
            "provider": "gemini",
            "model": "gemini-1.5-flash",
            "base_url": None,
            "temperature": 0.1,
            "api_key_set": False,
            "configured": False,
        },
        "last_model_error": False,
        "llm_test": {
            "ok": True,
            "url_reachable": True,
            "credentials_valid": True,
            "llm_responded": True,
            "message": "LLM connection test passed.",
            "response": "pong",
        },
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
                    "admin_exists": backend_state["bootstrapped"],
                    "llm_configured": backend_state["llm_config"]["configured"],
                    "models_enabled": sum(
                        1 for m in backend_state["models"] if m["enabled"]
                    ),
                },
            )
        if url.endswith("/models"):
            return _FakeResponse(200, {"models": backend_state["models"]})
        if url.endswith("/config/llm"):
            return _FakeResponse(200, backend_state["llm_config"])
        return _FakeResponse(404, {"detail": "Not found"})

    def fake_post(
        url: str, json: dict[str, Any] | None = None, **kwargs: Any
    ) -> _FakeResponse:
        if url.endswith("/config/llm/test"):
            return _FakeResponse(200, backend_state["llm_test"])
        if url.endswith("/setup/bootstrap"):
            if backend_state["bootstrapped"]:
                return _FakeResponse(409, {"detail": "Setup already completed."})
            backend_state["bootstrapped"] = True
            backend_state["setup_complete"] = True
            return _FakeResponse(
                200,
                {"user": {"username": (json or {})["username"]},
                 "setup_complete": True},
            )
        return _FakeResponse(404, {"detail": "Not found"})

    def fake_put(
        url: str, json: dict[str, Any] | None = None, **kwargs: Any
    ) -> _FakeResponse:
        body = json or {}
        if "/models/" in url:
            if backend_state["last_model_error"] and body.get("enabled") is False:
                return _FakeResponse(
                    400, {"detail": "Cannot disable the last enabled model."}
                )
            name = url.rsplit("/", 1)[-1]
            for model in backend_state["models"]:
                if model["name"] == name:
                    model["enabled"] = bool(body.get("enabled"))
            return _FakeResponse(200, {"models": backend_state["models"]})
        if url.endswith("/config/llm"):
            config = backend_state["llm_config"]
            for key, value in body.items():
                if key != "api_key":
                    config[key] = value
            if body.get("api_key"):
                config["api_key_set"] = True
            config["configured"] = True
            return _FakeResponse(200, config)
        return _FakeResponse(404, {"detail": "Not found"})

    monkeypatch.setattr(frontend_api_client.requests, "get", fake_get)
    monkeypatch.setattr(frontend_api_client.requests, "post", fake_post)
    monkeypatch.setattr(frontend_api_client.requests, "put", fake_put)
    return backend_state


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a testing Flask app with an isolated database."""
    monkeypatch.setattr(
        "config.TestingConfig.DATABASE", str(tmp_path / "frontend.db")
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
        execute_db(
            "UPDATE users SET must_change_password = 0 WHERE username = 'admin'"
        )
    test_client = app.test_client()
    with test_client.session_transaction() as sess:
        sess["_user_id"] = "1"
        sess["user_session_version"] = 0
    return test_client


class TestSetupGating:
    """The app redirects to /setup until the backend reports completion."""

    def test_redirects_to_wizard_when_incomplete(self, client) -> None:
        resp = client.get("/")
        assert resp.status_code == 302
        assert resp.headers["Location"].startswith("/setup")

    def test_allows_app_when_complete(self, client, backend_state) -> None:
        backend_state["setup_complete"] = True
        resp = client.get("/")
        assert resp.status_code == 302
        assert not resp.headers["Location"].startswith("/setup")

    def test_wizard_redirects_to_login_when_complete(
        self, client, backend_state
    ) -> None:
        backend_state["setup_complete"] = True
        resp = client.get("/setup/")
        assert resp.status_code == 302
        assert "/auth/login" in resp.headers["Location"]


class TestSetupGateWorkerConsistency:
    """The gate must resolve the backend URL from the DB, not just config.

    Gunicorn runs multiple workers; only the worker that handled the
    wizard's backend step has ``BACKEND_URL`` in its in-process config.
    Reading config alone made workers disagree about setup state, which
    caused the wizard/login redirect loop.
    """

    def test_gate_uses_db_url_when_config_is_stale(
        self, app, client, backend_state
    ) -> None:
        """A worker with empty in-memory config must still honour the DB URL."""
        client.post("/setup/backend", data={"base_url": _BACKEND_URL})

        # Simulate another gunicorn worker: in-process config is stale.
        app.config["BACKEND_URL"] = ""
        backend_state["setup_complete"] = True

        resp = client.get("/")
        assert resp.status_code == 302
        assert not resp.headers["Location"].startswith("/setup")

    def test_wizard_exits_when_db_url_set_and_backend_complete(
        self, app, client, backend_state
    ) -> None:
        """With the URL in the DB, every worker must agree setup is done."""
        client.post("/setup/backend", data={"base_url": _BACKEND_URL})

        app.config["BACKEND_URL"] = ""
        backend_state["setup_complete"] = True

        resp = client.get("/setup/")
        assert resp.status_code == 302
        assert "/auth/login" in resp.headers["Location"]

    def test_wizard_backend_step_prefills_db_url_on_stale_worker(
        self, app, client
    ) -> None:
        """Step 1 must pre-fill the URL from the DB on every worker."""
        client.post("/setup/backend", data={"base_url": _BACKEND_URL})

        app.config["BACKEND_URL"] = ""

        resp = client.get("/setup/backend")
        assert resp.status_code == 200
        assert _BACKEND_URL.encode() in resp.data

    def test_admin_step_preserves_db_base_url_on_stale_worker(
        self, app, client
    ) -> None:
        """Step 5 must not wipe the stored base_url on a worker without config.

        Previously step 5 read ``BACKEND_URL`` from in-process config; on a
        worker that had not handled step 1 that value was empty, so saving
        credentials overwrote the DB's ``base_url`` with an empty string and
        the wizard looped after every restart.
        """
        client.post("/setup/backend", data={"base_url": _BACKEND_URL})
        client.post(
            "/setup/llm",
            data={
                "provider": "gemini",
                "model": "gemini-1.5-flash",
                "temperature": "0.1",
            },
        )
        client.post("/setup/auth", data={"confirm": "y"})
        client.post(
            "/setup/models",
            data={"model_enabled": [m["name"] for m in _MODELS]},
        )

        # Simulate another gunicorn worker handling the final step.
        app.config["BACKEND_URL"] = ""
        app.config["API_VERIFY_SSL"] = False

        resp = client.post(
            "/setup/admin",
            data={"username": "frontend", "api_key": "test-secret-key-123"},
        )
        assert resp.status_code == 302
        assert "/setup/done" in resp.headers["Location"]

        with app.app_context():
            row = query_db(
                "SELECT base_url FROM api_credentials WHERE label = 'default'",
                one=True,
            )
        assert row is not None
        assert row["base_url"] == _BACKEND_URL


class TestSetupWizard:
    """Wizard step rendering and the full bootstrap flow."""

    def test_backend_step_renders(self, client) -> None:
        resp = client.get("/setup/backend")
        assert resp.status_code == 200
        assert b"Backend API Base URL" in resp.data

    def test_bootstrap_submit_stores_encrypted_credentials(
        self, app, client
    ) -> None:
        resp = client.post(
            "/setup/backend",
            data={"base_url": _BACKEND_URL},
        )
        assert resp.status_code == 302
        assert "/setup/llm" in resp.headers["Location"]

        resp = client.post(
            "/setup/llm",
            data={
                "provider": "gemini",
                "model": "gemini-1.5-flash",
                "temperature": "0.1",
            },
        )
        assert resp.status_code == 302
        assert "/setup/auth" in resp.headers["Location"]

        resp = client.post("/setup/auth", data={"confirm": "y"})
        assert resp.status_code == 302
        assert "/setup/models" in resp.headers["Location"]

        resp = client.post(
            "/setup/models",
            data={"model_enabled": [m["name"] for m in _MODELS]},
        )
        assert resp.status_code == 302
        assert "/setup/admin" in resp.headers["Location"]

        resp = client.post(
            "/setup/admin",
            data={"username": "frontend", "api_key": "test-secret-key-123"},
        )
        assert resp.status_code == 302
        assert "/setup/done" in resp.headers["Location"]

        with app.app_context():
            row = query_db(
                """
                SELECT base_url, encrypted_username, encrypted_password
                FROM api_credentials
                WHERE label = 'default'
                """,
                one=True,
            )
        assert row is not None
        assert row["base_url"] == _BACKEND_URL
        assert decrypt(str(row["encrypted_username"])) == "frontend"
        assert decrypt(str(row["encrypted_password"])) == "test-secret-key-123"
        # The plaintext key must never be stored.
        assert "test-secret-key-123" not in str(row["encrypted_password"])

    def test_llm_step_never_persists_key_locally(
        self, app, client, backend_state
    ) -> None:
        client.post("/setup/backend", data={"base_url": _BACKEND_URL})
        resp = client.post(
            "/setup/llm",
            data={
                "provider": "gemini",
                "model": "gemini-1.5-flash",
                "api_key": "super-secret-llm-key",
                "temperature": "0.1",
            },
        )
        assert resp.status_code == 302

        with app.app_context():
            row = query_db(
                "SELECT encrypted_username, encrypted_password "
                "FROM api_credentials WHERE label = 'default'",
                one=True,
            )
        assert row is not None
        assert row["encrypted_username"] is None
        assert row["encrypted_password"] is None
        # The key was forwarded to the backend instead.
        assert backend_state["llm_config"]["api_key_set"] is True

    def test_llm_step_blocks_progress_when_validation_fails(
        self, client, backend_state
    ) -> None:
        client.post("/setup/backend", data={"base_url": _BACKEND_URL})
        backend_state["llm_test"] = {
            "ok": False,
            "url_reachable": True,
            "credentials_valid": False,
            "llm_responded": False,
            "message": "The LLM URL or API key was rejected. Check both values.",
            "response": None,
        }

        resp = client.post(
            "/setup/llm",
            data={
                "provider": "gemini",
                "model": "gemini-1.5-flash",
                "api_key": "invalid-key",
                "temperature": "0.1",
            },
        )

        assert resp.status_code == 200
        assert b"URL reachable: Yes" in resp.data
        assert b"URL and API key valid: No" in resp.data
        assert backend_state["llm_config"]["configured"] is False
        with client.session_transaction() as sess:
            assert "setup_llm_ok" not in sess

    def test_models_step_rejects_unchecking_all(self, client) -> None:
        client.post("/setup/backend", data={"base_url": _BACKEND_URL})
        client.post(
            "/setup/llm",
            data={
                "provider": "gemini",
                "model": "gemini-1.5-flash",
                "temperature": "0.1",
            },
        )
        client.post("/setup/auth", data={"confirm": "y"})
        resp = client.post("/setup/models", data={})
        assert resp.status_code == 200
        assert b"At least one model must remain enabled." in resp.data


class TestAdminConfigPages:
    """Admin LLM config and model registry pages."""

    def test_llm_config_page_renders_masked(self, admin_client) -> None:
        resp = admin_client.get("/admin/llm-config")
        assert resp.status_code == 200
        assert b"gemini-1.5-flash" in resp.data
        assert b"Not set" in resp.data
        assert b"leave blank to keep current" in resp.data
        assert b"Test LLM" in resp.data

    def test_llm_test_displays_response_without_saving(
        self, admin_client, backend_state
    ) -> None:
        resp = admin_client.post(
            "/admin/llm-config",
            data={
                "provider": "gemini",
                "model": "candidate-model",
                "api_key": "candidate-key",
                "temperature": "0.1",
                "test_llm": "Test LLM",
            },
        )

        assert resp.status_code == 200
        assert b"LLM response:" in resp.data
        assert b"pong" in resp.data
        assert backend_state["llm_config"]["model"] == "gemini-1.5-flash"

    def test_llm_save_is_blocked_when_validation_fails(
        self, admin_client, backend_state
    ) -> None:
        backend_state["llm_test"] = {
            "ok": False,
            "url_reachable": False,
            "credentials_valid": False,
            "llm_responded": False,
            "message": "The LLM URL could not be reached.",
            "response": None,
        }

        resp = admin_client.post(
            "/admin/llm-config",
            data={
                "provider": "ollama_cloud",
                "model": "candidate-model",
                "base_url": "https://unreachable.example",
                "api_key": "candidate-key",
                "temperature": "0.1",
                "submit": "Save LLM Configuration",
            },
        )

        assert resp.status_code == 200
        assert b"The LLM URL could not be reached." in resp.data
        assert backend_state["llm_config"]["model"] == "gemini-1.5-flash"

    def test_models_page_renders(self, admin_client) -> None:
        resp = admin_client.get("/admin/models")
        assert resp.status_code == 200
        assert b"Holt-Winters" in resp.data
        assert b"Prophet" in resp.data

    def test_models_page_surfaces_last_model_error(
        self, admin_client, backend_state
    ) -> None:
        backend_state["last_model_error"] = True
        resp = admin_client.post(
            "/admin/models",
            data={"model_enabled": ["ARIMA"]},
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"Cannot disable the last enabled model." in resp.data
