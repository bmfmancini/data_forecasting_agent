"""Tests for the admin LLM config endpoints (masked reads, one-way writes)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import core.config as settings
from core import secret_store
from core.database import init_database
from main import app


@pytest.fixture(autouse=True)
def _isolated_backend(tmp_path: Any, monkeypatch: Any) -> None:
    """Fresh DB + secrets dir per test; auth disabled for simplicity."""
    db_path = str(tmp_path / "backend.db")
    monkeypatch.setattr(settings, "BACKEND_DB_PATH", db_path)
    monkeypatch.setattr(settings, "API_KEY_ENABLED", False)
    monkeypatch.setattr(settings, "SECRETS_DIR", str(tmp_path / "secrets"))
    secret_store.reset_cache()
    init_database()
    yield
    secret_store.reset_cache()


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


class TestLLMConfigRead:
    """GET /config/llm never exposes the key."""

    def test_masked_read_after_write(self, client: TestClient) -> None:
        put = client.put(
            "/config/llm",
            json={
                "provider": "gemini",
                "model": "gemini-2.0",
                "api_key": "super-secret-plaintext",
                "temperature": 0.2,
            },
        )
        assert put.status_code == 200

        response = client.get("/config/llm")

        assert response.status_code == 200
        data = response.json()
        assert data == {
            "provider": "gemini",
            "model": "gemini-2.0",
            "base_url": None,
            "temperature": pytest.approx(0.2),
            "api_key_set": True,
            "configured": True,
        }
        assert "super-secret-plaintext" not in response.text

    def test_unconfigured_reports_false(self, client: TestClient) -> None:
        data = client.get("/config/llm").json()
        assert data["configured"] is False


class TestLLMConfigWrite:
    """PUT /config/llm one-way write semantics."""

    def test_unknown_provider_rejected(self, client: TestClient) -> None:
        response = client.put(
            "/config/llm",
            json={"provider": "bogus", "model": "m"},
        )
        assert response.status_code == 400

    def test_omitting_key_preserves_stored(self, client: TestClient) -> None:
        client.put(
            "/config/llm",
            json={"provider": "gemini", "model": "m1", "api_key": "keep-me"},
        )
        response = client.put(
            "/config/llm",
            json={"provider": "gemini", "model": "m2"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "m2"
        assert data["api_key_set"] is True

    def test_secret_str_repr_does_not_leak(self) -> None:
        from schemas import LLMConfigUpdateRequest

        request = LLMConfigUpdateRequest(
            provider="gemini", model="m", api_key="top-secret"
        )
        assert "top-secret" not in repr(request)
        assert "top-secret" not in str(request)
