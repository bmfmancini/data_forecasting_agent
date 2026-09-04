"""Tests for staged candidate LLM validation."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from services import llm_validation_service as service


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

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
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
        base_url="https://ollama.example/",
        api_key="secret-key",
    )

    assert result.ok is True
    assert result.url_reachable is True
    assert result.credentials_valid is True
    assert result.llm_responded is True
    assert result.response == "pong"
    assert [call[1] for call in client.calls] == [
        "https://ollama.example",
        "https://ollama.example/api/tags",
        "https://ollama.example/api/chat",
    ]
    assert client.calls[1][2]["headers"]["Authorization"] == "Bearer secret-key"


@pytest.mark.asyncio
async def test_invalid_credentials_stop_before_ping(monkeypatch: Any) -> None:
    client = _FakeClient([_Response(200, {}), _Response(401, {})])
    _install_client(monkeypatch, client)

    result = await service.validate_llm_configuration(
        provider="ollama_cloud",
        model="llama-test",
        base_url="https://ollama.example",
        api_key="invalid-key",
    )

    assert result.ok is False
    assert result.url_reachable is True
    assert result.credentials_valid is False
    assert result.llm_responded is False
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_unreachable_url_stops_before_credentials(monkeypatch: Any) -> None:
    request = httpx.Request("GET", "https://offline.example")
    client = _FakeClient([httpx.ConnectError("offline", request=request)])
    _install_client(monkeypatch, client)

    result = await service.validate_llm_configuration(
        provider="ollama",
        model="llama-test",
        base_url="https://offline.example",
        api_key=None,
    )

    assert result.ok is False
    assert result.url_reachable is False
    assert len(client.calls) == 1
