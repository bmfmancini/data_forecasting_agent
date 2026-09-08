"""API-level tests for the deployment-wide "Enable AI features" switch.

PUT /config/llm/enabled is the single write endpoint; every LLM-dependent
behaviour (analyze gating, chat, health probes) must degrade cleanly when
the switch is off while the provider-configuration endpoints stay open —
they are required for the enable-later workflow on installs that started
LLM-free.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import core.config as settings
import services.job_service as job_service
from core import secret_store
from core.database import get_connection, init_database
from core.system_settings_store import set_llm_enabled
import main
from main import app
from services.file_service import store_file
from services.llm_validation_service import LLMValidationResult


@pytest.fixture(autouse=True)
def _isolated_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh DB + secrets dir per test; auth disabled for simplicity."""
    monkeypatch.setattr(settings, "BACKEND_DB_PATH", str(tmp_path / "backend.db"))
    monkeypatch.setattr(settings, "API_KEY_ENABLED", False)
    monkeypatch.setattr(settings, "SECRETS_DIR", str(tmp_path / "secrets"))
    monkeypatch.setattr(settings, "FILE_STORAGE_DIR", str(tmp_path / "files"))
    Path(tmp_path / "files").mkdir()
    monkeypatch.setattr(job_service, "JOB_QUEUE", asyncio.Queue())
    job_service._job_store.clear()
    secret_store.reset_cache()
    init_database()
    yield
    secret_store.reset_cache()
    job_service._job_store.clear()


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client (lifespan not run — the queue is patched above)."""
    return TestClient(app)


def _stored_file() -> str:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=36, freq="MS"),
            "value": [float(i) for i in range(36)],
        }
    )
    return store_file(df, "date", "value", "MS", "data.csv")


# ── The switch itself ─────────────────────────────────────────────────────────


class TestLLMEnabledSwitch:
    """PUT /config/llm/enabled round-trips and surfaces everywhere."""

    def test_put_round_trip(self, client: TestClient) -> None:
        off = client.put("/config/llm/enabled", json={"enabled": False})
        assert off.status_code == 200
        assert off.json()["llm_enabled"] is False

        get = client.get("/config/llm")
        assert get.json()["llm_enabled"] is False

        on = client.put("/config/llm/enabled", json={"enabled": True})
        assert on.status_code == 200
        assert on.json()["llm_enabled"] is True

    def test_setup_status_reports_the_switch(self, client: TestClient) -> None:
        status = client.get("/setup/status")
        assert status.status_code == 200
        assert status.json()["llm_enabled"] is True

        set_llm_enabled(False)
        status = client.get("/setup/status")
        assert status.json()["llm_enabled"] is False


# ── /analyze gating ───────────────────────────────────────────────────────────


class TestAnalyzeGating:
    """Auto model selection is unavailable whenever the effective mode is
    traditional — rejected at submission with an actionable message."""

    def test_per_run_traditional_requires_model(self, client: TestClient) -> None:
        response = client.post(
            "/analyze",
            json={"file_id": "any", "forecast_horizon": 3, "traditional_mode": True},
        )
        assert response.status_code == 400
        assert "Traditional Forecasting requires" in response.json()["detail"]
        assert "auto selection is disabled" in response.json()["detail"]

    def test_global_disable_blocks_auto_request(self, client: TestClient) -> None:
        """A plain AI-mode request with no forced model is still blocked
        when the deployment-wide switch is off (users cannot override)."""
        set_llm_enabled(False)
        response = client.post(
            "/analyze",
            json={"file_id": "any", "forecast_horizon": 3},
        )
        assert response.status_code == 400
        assert "Traditional Forecasting requires" in response.json()["detail"]

    def test_traditional_submission_creates_traditional_job(
        self, client: TestClient
    ) -> None:
        file_id = _stored_file()
        response = client.post(
            "/analyze",
            json={
                "file_id": file_id,
                "forecast_horizon": 3,
                "forced_model": "ARIMA",
                "traditional_mode": True,
            },
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]

        connection = get_connection()
        try:
            row = connection.execute(
                "SELECT traditional_mode, status FROM forecast_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        finally:
            connection.close()
        assert row["traditional_mode"] == 1

    def test_globally_disabled_submission_stores_traditional_job(
        self, client: TestClient
    ) -> None:
        """Even an AI-mode request is *stored* with the effective mode."""
        set_llm_enabled(False)
        file_id = _stored_file()
        response = client.post(
            "/analyze",
            json={
                "file_id": file_id,
                "forecast_horizon": 3,
                "forced_model": "Holt-Winters",
            },
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]

        connection = get_connection()
        try:
            row = connection.execute(
                "SELECT traditional_mode FROM forecast_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        finally:
            connection.close()
        assert row["traditional_mode"] == 1


# ── Chat and health while disabled ────────────────────────────────────────────


class TestDisabledChatAndHealth:
    """Chat is refused and health does not probe the provider while off."""

    def test_chat_returns_503_when_disabled(self, client: TestClient) -> None:
        set_llm_enabled(False)
        response = client.post("/chat", json={"query": "hello"})
        assert response.status_code == 503
        assert "AI chat is disabled" in response.json()["detail"]

    def test_llm_health_skips_provider_probe_when_disabled(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ollama_probe = AsyncMock(return_value=True)
        gemini_probe = AsyncMock(return_value=True)
        monkeypatch.setattr(main, "_check_ollama_reachable", ollama_probe)
        monkeypatch.setattr(main, "_check_gemini_reachable", gemini_probe)

        set_llm_enabled(False)
        response = client.get("/llm-health")

        data = response.json()
        assert data == {
            "llm_enabled": False,
            "llm_configured": False,
            "llm_reachable": False,
        }
        ollama_probe.assert_not_awaited()
        gemini_probe.assert_not_awaited()


# ── Enable-later workflow ─────────────────────────────────────────────────────


class TestEnableLaterWorkflow:
    """Provider configuration stays fully usable while AI is disabled."""

    def test_provider_test_endpoint_works_while_disabled(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An administrator can test and save a provider before flipping
        the switch — the test endpoint is not gated."""
        validator = AsyncMock(
            return_value=LLMValidationResult(
                ok=True,
                url_reachable=True,
                credentials_valid=True,
                llm_responded=True,
                message="LLM connection test passed.",
            )
        )
        monkeypatch.setattr(main, "validate_llm_configuration", validator)
        set_llm_enabled(False)

        response = client.post(
            "/config/llm/test",
            json={"provider": "gemini", "model": "gemini-2.0", "api_key": "k"},
        )

        assert response.status_code == 200
        assert response.json()["ok"] is True
        validator.assert_awaited_once()

    def test_provider_config_save_works_while_disabled(
        self, client: TestClient
    ) -> None:
        set_llm_enabled(False)
        response = client.put(
            "/config/llm",
            json={"provider": "gemini", "model": "gemini-2.0", "api_key": "k"},
        )
        assert response.status_code == 200

        config = client.get("/config/llm").json()
        assert config["configured"] is True
        assert config["llm_enabled"] is False

    def test_llm_health_probes_after_reenabling(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After re-enabling, health resumes probing the configured provider."""
        monkeypatch.setattr(
            main, "_check_gemini_reachable", AsyncMock(return_value=True)
        )
        # Save a provider while AI is still disabled (the enable-later
        # workflow), then flip the switch.
        client.put(
            "/config/llm",
            json={"provider": "gemini", "model": "gemini-2.0", "api_key": "k"},
        )
        client.put("/config/llm/enabled", json={"enabled": True})

        data = client.get("/llm-health").json()
        assert data["llm_enabled"] is True
        assert data["llm_configured"] is True
        assert data["llm_reachable"] is True