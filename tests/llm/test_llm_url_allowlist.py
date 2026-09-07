"""Persistent LLM trust policy and admin-only API access."""

from __future__ import annotations

import httpx
import pytest
from auth.api_key_db import create_api_user, create_first_user
from core import config as settings
from core.database import init_database
from core.llm_config_store import put_llm_config
from core.llm_url_allowlist import get_allowed_origins, put_allowed_origins
from fastapi.testclient import TestClient
from main import app
from services import llm_validation_service as service


@pytest.fixture(autouse=True)
def isolated_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "BACKEND_DB_PATH", str(tmp_path / "backend.db"))
    init_database()


@pytest.fixture
def admin_headers():
    create_first_user(username="admin", api_key="admin-test-key")
    return {"X-API-Username": "admin", "X-API-Key": "admin-test-key"}


def test_defaults_seeded_and_edits_survive_reinitialization():
    assert set(get_allowed_origins()) == {
        "http://localhost:11434",
        "http://host.docker.internal:11434",
        "https://ollama.com",
        "https://api.ollama.com",
    }
    assert put_allowed_origins(
        [" https://custom.example/proxy/ ", "https://custom.example/proxy"]
    ) == ["https://custom.example/proxy"]
    init_database()
    assert get_allowed_origins() == ["https://custom.example/proxy"]
    put_allowed_origins([])
    init_database()
    assert get_allowed_origins() == []


@pytest.mark.parametrize(
    "url",
    [
        "",
        "*",
        "https://*.example",
        "ftp://example.com",
        "http://[broken",
        "https://user:secret@example.com",
        "https://@example.com",
        "https://example.com?redirect=evil",
        "https://example.com#fragment",
        "https://example.com\\@evil",
        "https://exam\nple.com",
        "https://example.com:99999",
        "https://example.com/" + "a" * 2048,
    ],
)
def test_invalid_policy_never_partially_saves(url):
    before = get_allowed_origins()
    with pytest.raises(ValueError):
        put_allowed_origins(["https://valid.example", url])
    assert get_allowed_origins() == before


@pytest.mark.parametrize("auth_enabled", [False, True])
@pytest.mark.parametrize("method", ["GET", "PUT"])
def test_only_verified_admins_can_access_policy(
    monkeypatch, admin_headers, auth_enabled, method
):
    monkeypatch.setattr(settings, "API_KEY_ENABLED", auth_enabled)
    regular_key = create_api_user(
        username="regular", description="Test user", is_admin=False
    )
    client = TestClient(app)
    kwargs = (
        {"json": {"origins": ["https://custom.example"]}} if method == "PUT" else {}
    )
    assert (
        client.request(method, "/config/llm/allowed-origins", **kwargs).status_code
        == 401
    )
    assert (
        client.request(
            method,
            "/config/llm/allowed-origins",
            headers={"X-API-Username": "admin", "X-API-Key": "wrong"},
            **kwargs,
        ).status_code
        == 401
    )
    assert (
        client.request(
            method,
            "/config/llm/allowed-origins",
            headers={"X-API-Username": "regular", "X-API-Key": regular_key},
            **kwargs,
        ).status_code
        == 403
    )
    response = client.request(
        method, "/config/llm/allowed-origins", headers=admin_headers, **kwargs
    )
    assert response.status_code == 200
    assert "origins" in response.json()


def test_invalid_api_input_leaves_policy_unchanged(admin_headers):
    before = get_allowed_origins()
    client = TestClient(app)
    response = client.put(
        "/config/llm/allowed-origins",
        headers=admin_headers,
        json={"origins": ["https://user:secret@example.com"]},
    )
    assert response.status_code == 400
    assert "secret" not in response.text
    assert (
        client.put(
            "/config/llm/allowed-origins",
            headers=admin_headers,
            json={"origins": ["https://example.com"] * 101},
        ).status_code
        == 422
    )
    assert get_allowed_origins() == before


@pytest.mark.asyncio
async def test_saved_policy_controls_subsequent_requests(monkeypatch, admin_headers):
    client = TestClient(app)
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, json={"message": {"content": "pong"}})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        service.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    args = {
        "provider": "ollama",
        "model": "test",
        "base_url": "https://custom.example",
        "api_key": None,
    }
    # Saving provider configuration cannot approve its own destination.
    put_llm_config(
        provider="ollama",
        model="test",
        base_url=args["base_url"],
        api_key=None,
        temperature=0.1,
    )
    assert not (await service.validate_llm_configuration(**args)).ok
    assert calls == []
    assert (
        client.put(
            "/config/llm/allowed-origins",
            headers=admin_headers,
            json={"origins": [args["base_url"]]},
        ).status_code
        == 200
    )
    assert (await service.validate_llm_configuration(**args)).ok
    assert len(calls) == 3
    calls.clear()
    assert (
        client.put(
            "/config/llm/allowed-origins", headers=admin_headers, json={"origins": []}
        ).status_code
        == 200
    )
    assert not (await service.validate_llm_configuration(**args)).ok
    assert calls == []
