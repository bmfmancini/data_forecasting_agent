"""Tests for provider-specific LangChain client construction."""

from __future__ import annotations

from typing import Any

import pytest

import core.llm_factory as llm_factory
from core.llm_config_store import LLMConfig


@pytest.mark.parametrize("provider", ["ollama_cloud", "ollama"])
def test_ollama_clients_pass_bearer_token_via_client_kwargs(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    """ChatOllama only forwards custom HTTP headers from ``client_kwargs``."""
    captured: dict[str, Any] = {}

    class FakeChatOllama:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(llm_factory, "ChatOllama", FakeChatOllama)
    monkeypatch.setattr(
        llm_factory,
        "get_llm_config",
        lambda: LLMConfig(
            provider=provider,
            model="test-model",
            base_url="https://ollama.example",
            api_key="test-token",
            temperature=0.1,
            version=1,
        ),
    )

    llm_factory.get_llm()

    assert captured["client_kwargs"] == {
        "headers": {"Authorization": "Bearer test-token"}
    }
    assert "headers" not in captured
