"""Unit tests for the LLM factory in core.llm_factory.

These tests mock ``ChatOllama`` and ``ChatGoogleGenerativeAI`` so no
network calls are made. They verify that the correct provider, base URL,
and authentication headers are selected based on the configuration.
"""

from unittest.mock import patch, MagicMock

import pytest

import core.config as config
from core.llm_factory import get_llm
from exceptions import LLMConfigError


@pytest.fixture
def reset_config():
    """Snapshot config values and restore them after each test."""
    original = {
        "USE_OLLAMA": config.USE_OLLAMA,
        "USE_OLLAMA_CLOUD": config.USE_OLLAMA_CLOUD,
        "OLLAMA_BASE_URL": config.OLLAMA_BASE_URL,
        "OLLAMA_MODEL": config.OLLAMA_MODEL,
        "OLLAMA_API_KEY": config.OLLAMA_API_KEY,
        "GEMINI_MODEL": config.GEMINI_MODEL,
        "GOOGLE_API_KEY": config.GOOGLE_API_KEY,
    }
    yield
    for key, value in original.items():
        setattr(config, key, value)


class TestGetLLM:
    """Tests for core.llm_factory.get_llm."""

    @patch("core.llm_factory.ChatOllama")
    def test_ollama_cloud_uses_cloud_base_url_and_bearer(
        self, mock_chat_ollama: MagicMock, reset_config
    ) -> None:
        """Ollama Cloud should target ollama.com with a Bearer API key."""
        config.USE_OLLAMA = True
        config.USE_OLLAMA_CLOUD = True
        config.OLLAMA_API_KEY = "test-cloud-key"
        config.OLLAMA_MODEL = "gemma4:31b-cloud"
        config.OLLAMA_BASE_URL = "https://ollama.com"

        get_llm(temperature=0)

        _, kwargs = mock_chat_ollama.call_args
        assert kwargs["base_url"] == "https://ollama.com"
        assert kwargs["model"] == "gemma4:31b-cloud"
        assert kwargs["headers"] == {"Authorization": "Bearer test-cloud-key"}

    @patch("core.llm_factory.ChatOllama")
    def test_ollama_cloud_missing_api_key_raises(
        self, mock_chat_ollama: MagicMock, reset_config
    ) -> None:
        """An empty OLLAMA_API_KEY with cloud enabled should raise LLMConfigError."""
        config.USE_OLLAMA = True
        config.USE_OLLAMA_CLOUD = True
        config.OLLAMA_API_KEY = None

        with pytest.raises(LLMConfigError):
            get_llm()

        mock_chat_ollama.assert_not_called()

    @patch("core.llm_factory.ChatOllama")
    def test_local_ollama_uses_local_base_url(
        self, mock_chat_ollama: MagicMock, reset_config
    ) -> None:
        """Local Ollama should target OLLAMA_BASE_URL, not the cloud URL."""
        config.USE_OLLAMA = True
        config.USE_OLLAMA_CLOUD = False
        config.OLLAMA_BASE_URL = "http://localhost:11434"
        config.OLLAMA_MODEL = "llama3"
        config.OLLAMA_API_KEY = None

        get_llm(temperature=0)

        _, kwargs = mock_chat_ollama.call_args
        assert kwargs["base_url"] == "http://localhost:11434"
        assert kwargs["headers"] is None

    @patch("core.llm_factory.ChatOllama")
    def test_local_ollama_with_api_key_sends_bearer(
        self, mock_chat_ollama: MagicMock, reset_config
    ) -> None:
        """A local Ollama with an API key should still send a Bearer header."""
        config.USE_OLLAMA = True
        config.USE_OLLAMA_CLOUD = False
        config.OLLAMA_BASE_URL = "http://localhost:11434"
        config.OLLAMA_API_KEY = "local-key"

        get_llm()

        _, kwargs = mock_chat_ollama.call_args
        assert kwargs["headers"] == {"Authorization": "Bearer local-key"}

    @patch("core.llm_factory.ChatGoogleGenerativeAI")
    def test_gemini_used_when_ollama_disabled(
        self, mock_chat_gemini: MagicMock, reset_config
    ) -> None:
        """When USE_OLLAMA is false, the Gemini provider should be selected."""
        config.USE_OLLAMA = False
        config.GEMINI_MODEL = "gemini-1.5-flash"
        config.GOOGLE_API_KEY = "google-key"

        get_llm(temperature=0.1)

        _, kwargs = mock_chat_gemini.call_args
        assert kwargs["model"] == "gemini-1.5-flash"
        assert kwargs["google_api_key"] == "google-key"
        assert kwargs["temperature"] == 0.1

    @patch("core.llm_factory.ChatOllama")
    def test_temperature_is_passed_through(
        self, mock_chat_ollama: MagicMock, reset_config
    ) -> None:
        """The temperature argument should reach the underlying LLM client."""
        config.USE_OLLAMA = True
        config.USE_OLLAMA_CLOUD = False
        config.OLLAMA_API_KEY = None

        get_llm(temperature=0.7)

        _, kwargs = mock_chat_ollama.call_args
        assert kwargs["temperature"] == 0.7