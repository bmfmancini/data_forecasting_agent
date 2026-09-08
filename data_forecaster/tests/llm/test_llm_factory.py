"""Tests for provider-specific LangChain client construction."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

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


def test_get_llm_raises_llm_disabled_error_when_ai_features_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deployment-wide switch is the mid-run enforcement point.

    ``get_llm`` raises before building any client, so an LLM call not yet
    dispatched — including inside a forecast already running when an
    administrator disabled AI — fails at construction and the calling
    agent's deterministic fallback takes over.
    """
    from exceptions import LLMDisabledError

    monkeypatch.setattr(llm_factory, "is_llm_enabled", lambda: False)
    monkeypatch.setattr(
        llm_factory,
        "get_llm_config",
        lambda: LLMConfig(
            provider="gemini",
            model="m",
            base_url=None,
            api_key=None,
            temperature=0.1,
            version=1,
        ),
    )

    with pytest.raises(LLMDisabledError, match="AI features are disabled"):
        llm_factory.get_llm()


def test_get_llm_builds_client_when_ai_features_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must not block construction when the switch is on."""
    monkeypatch.setattr(llm_factory, "is_llm_enabled", lambda: True)
    monkeypatch.setattr(
        llm_factory,
        "get_llm_config",
        lambda: LLMConfig(
            provider="gemini",
            model="m",
            base_url=None,
            api_key="key",
            temperature=0.1,
            version=1,
        ),
    )
    monkeypatch.setattr(llm_factory, "ChatGoogleGenerativeAI", MagicMock())

    llm_factory.get_llm()

    llm_factory.ChatGoogleGenerativeAI.assert_called_once()
