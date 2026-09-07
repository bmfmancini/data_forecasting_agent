"""Tests for staged candidate LLM validation."""

from __future__ import annotations

from typing import Any, Self

import httpx
import pytest
from core import config as settings
from core.database import init_database
from core.llm_url_allowlist import put_allowed_origins
from services import llm_validation_service as service


@pytest.fixture(autouse=True)
def isolated_allowlist(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "BACKEND_DB_PATH", str(tmp_path / "backend.db"))
    init_database()


class _Response:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self.is_success = 200 <= status_code < 300
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    def __init__(self, responses: list[_Response | httpx.RequestError]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def _request(self, method: str, url: str, **kwargs: Any) -> _Response:
        self.calls.append((method, url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, httpx.RequestError):
            raise response
        return response

    async def get(self, url: str, **kwargs: Any) -> _Response:
        return await self._request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> _Response:
        return await self._request("POST", url, **kwargs)


def _install_client(monkeypatch: Any, client: _FakeClient) -> None:
    monkeypatch.setattr(service.httpx, "AsyncClient", lambda **_kwargs: client)


@pytest.mark.asyncio
async def test_ollama_validation_runs_all_three_stages(monkeypatch: Any) -> None:
    client = _FakeClient(
        [
            _Response(200, {}),
            _Response(200, {"models": []}),
            _Response(200, {"message": {"content": "pong"}}),
        ]
    )
    _install_client(monkeypatch, client)

    result = await service.validate_llm_configuration(
        provider="ollama_cloud",
        model="llama-test",
        base_url="https://ollama.com/",
        api_key="secret-key",
    )

    assert result.ok is True
    assert result.url_reachable is True
    assert result.credentials_valid is True
    assert result.llm_responded is True
    assert result.response == "pong"
    assert [call[1] for call in client.calls] == [
        "https://ollama.com",
        "https://ollama.com/api/tags",
        "https://ollama.com/api/chat",
    ]
    assert client.calls[1][2]["headers"]["Authorization"] == "Bearer secret-key"
    assert client.calls[2][2]["json"]["think"] is False
    assert client.calls[2][2]["json"]["options"]["num_predict"] == 64


@pytest.mark.asyncio
async def test_invalid_credentials_stop_before_ping(monkeypatch: Any) -> None:
    client = _FakeClient([_Response(200, {}), _Response(401, {})])
    _install_client(monkeypatch, client)

    result = await service.validate_llm_configuration(
        provider="ollama_cloud",
        model="llama-test",
        base_url="https://ollama.com",
        api_key="invalid-key",
    )

    assert result.ok is False
    assert result.url_reachable is True
    assert result.credentials_valid is False
    assert result.llm_responded is False
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_unreachable_url_stops_before_credentials(monkeypatch: Any) -> None:
    request = httpx.Request("GET", "http://localhost:11434")
    client = _FakeClient([httpx.ConnectError("offline", request=request)])
    _install_client(monkeypatch, client)

    result = await service.validate_llm_configuration(
        provider="ollama",
        model="llama-test",
        base_url="http://localhost:11434",
        api_key=None,
    )

    assert result.ok is False
    assert result.url_reachable is False
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_model_rejection_returns_safe_provider_diagnostic(
    monkeypatch: Any,
) -> None:
    client = _FakeClient(
        [
            _Response(200, {}),
            _Response(200, {"models": []}),
            _Response(404, {"error": "model 'missing-model' not found"}),
        ]
    )
    _install_client(monkeypatch, client)

    result = await service.validate_llm_configuration(
        provider="ollama",
        model="missing-model",
        base_url="https://ollama.com",
        api_key=None,
    )

    assert result.credentials_valid is True
    assert result.diagnostic == "HTTP 404: model 'missing-model' not found"


@pytest.mark.asyncio
async def test_ping_auth_rejection_marks_credentials_invalid(monkeypatch: Any) -> None:
    client = _FakeClient(
        [
            _Response(200, {}),
            _Response(200, {"models": []}),
            _Response(401, {"error": {"message": "invalid API key"}}),
        ]
    )
    _install_client(monkeypatch, client)

    result = await service.validate_llm_configuration(
        provider="ollama_cloud",
        model="llama-test",
        base_url="https://ollama.com",
        api_key="invalid-key",
    )

    assert result.credentials_valid is False
    assert result.diagnostic == "HTTP 401: invalid API key"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    [
        None,
        "",
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:8080",
        "https://attacker.example",
        "https://ollama.com.attacker.example",
        "https://ollama.com@attacker.example",
        "https://user:password@ollama.com",
        "https://ollama.com:444",
        "https://ollama.com/api/delete",
        "https://ollama.com?url=http://attacker.example",
        "https://ollama.com#fragment",
        "https://ollama.com\\@attacker.example",
        "https://olla\nma.com",
        "file:///etc/passwd",
        "http://[broken",
    ],
)
async def test_untrusted_urls_make_no_requests(monkeypatch, base_url):
    client = _FakeClient([])
    _install_client(monkeypatch, client)
    result = await service.validate_llm_configuration(
        provider="ollama", model="test", base_url=base_url, api_key="stored-secret"
    )
    assert not result.ok
    assert "allowlist" in result.message
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:11434",
        "http://host.docker.internal:11434",
        "https://ollama.com",
        "https://api.ollama.com",
    ],
)
async def test_supported_deployment_urls(monkeypatch, base_url):
    client = _FakeClient(
        [
            _Response(200, {}),
            _Response(200, {}),
            _Response(200, {"message": {"content": "pong"}}),
        ]
    )
    _install_client(monkeypatch, client)
    result = await service.validate_llm_configuration(
        provider="ollama", model="test", base_url=base_url + "/", api_key=None
    )
    assert result.ok
    assert client.calls[2][1] == base_url + "/api/chat"


@pytest.mark.asyncio
async def test_custom_endpoint_requires_saved_allowlist(monkeypatch):
    base_url = "http://ollama.internal:11434/proxy"
    client = _FakeClient(
        [
            _Response(200, {}),
            _Response(200, {}),
            _Response(200, {"message": {"content": "pong"}}),
        ]
    )
    _install_client(monkeypatch, client)
    args = {
        "provider": "ollama",
        "model": "test",
        "base_url": base_url,
        "api_key": None,
    }
    assert not (await service.validate_llm_configuration(**args)).ok
    assert client.calls == []
    put_allowed_origins([base_url + "/"])
    assert (await service.validate_llm_configuration(**args)).ok
    assert client.calls[2][1] == base_url + "/api/chat"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["gemini", "ollama_cloud"])
@pytest.mark.parametrize("redirect_stage", [0, 1, 2])
async def test_redirects_never_reach_another_destination(
    monkeypatch, provider, redirect_stage
):
    # Use real HTTPX redirect handling with an in-memory transport.
    requests = []
    origin = (
        "https://generativelanguage.googleapis.com"
        if provider == "gemini"
        else "https://ollama.com"
    )

    def handler(request):
        requests.append(request)
        if len(requests) - 1 == redirect_stage:
            return httpx.Response(
                307, headers={"Location": "http://169.254.169.254/latest/meta-data/"}
            )
        return httpx.Response(
            200,
            json={
                "message": {"content": "pong"},
                "candidates": [{"content": {"parts": [{"text": "pong"}]}}],
            },
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        service.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    result = await service.validate_llm_configuration(
        provider=provider, model="test", base_url=origin, api_key="stored-secret"
    )
    assert all(
        request.url.copy_with(path="", query=None, fragment=None) == httpx.URL(origin)
        for request in requests
    )
    if redirect_stage:
        assert not result.ok
        assert len(requests) == redirect_stage + 1
    else:
        assert result.ok  # A redirect still proves reachability; its target is ignored.
