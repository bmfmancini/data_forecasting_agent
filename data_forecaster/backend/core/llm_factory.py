"""Factory for creating the configured LangChain chat LLM instance.

This module centralises LLM instantiation so that every agent and the
orchestrator share a single, consistent configuration source. It supports
three providers selected via environment variables in :mod:`core.config`:

1. **Ollama Cloud** (``USE_OLLAMA=true`` and ``USE_OLLAMA_CLOUD=true``) —
   reaches directly to ``https://ollama.com`` (or ``OLLAMA_CLOUD_BASE_URL``)
   using a Bearer ``OLLAMA_API_KEY``.
2. **Local Ollama** (``USE_OLLAMA=true`` and ``USE_OLLAMA_CLOUD=false``) —
   reaches to the local/remote Ollama daemon at ``OLLAMA_BASE_URL``.
3. **Google Gemini** (``USE_OLLAMA=false``) — uses
   ``ChatGoogleGenerativeAI`` with ``GOOGLE_API_KEY``.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

import core.config as config
from core.logging_config import get_logger
from exceptions import LLMConfigError

logger = get_logger(__name__)


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """Build and return the configured LangChain chat LLM.

    The provider is chosen from the project configuration:

    - Ollama Cloud when ``USE_OLLAMA`` and ``USE_OLLAMA_CLOUD`` are both
      ``True``.
    - Local Ollama when ``USE_OLLAMA`` is ``True`` and
      ``USE_OLLAMA_CLOUD`` is ``False``.
    - Google Gemini otherwise.

    Args:
        temperature: Sampling temperature to pass to the LLM. Defaults to
            ``0.0`` for deterministic output.

    Returns:
        A configured :class:`BaseChatModel` instance.

    Raises:
        LLMConfigError: When Ollama Cloud is enabled but
            ``OLLAMA_API_KEY`` is not set.
    """
    if config.USE_OLLAMA and config.USE_OLLAMA_CLOUD:
        if not config.OLLAMA_API_KEY:
            raise LLMConfigError(
                "USE_OLLAMA_CLOUD is enabled but OLLAMA_API_KEY is not "
                "set. Create an API key at https://ollama.com/settings/keys "
                "and set the OLLAMA_API_KEY environment variable."
            )
        logger.info(
            "Using Ollama Cloud (model=%s, base_url=%s)",
            config.OLLAMA_MODEL,
            config.OLLAMA_BASE_URL,
        )
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
            headers={"Authorization": f"Bearer {config.OLLAMA_API_KEY}"},
        )

    if config.USE_OLLAMA:
        logger.info(
            "Using local Ollama (model=%s, base_url=%s)",
            config.OLLAMA_MODEL,
            config.OLLAMA_BASE_URL,
        )
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
            headers=(
                {"Authorization": f"Bearer {config.OLLAMA_API_KEY}"}
                if config.OLLAMA_API_KEY
                else None
            ),
        )

    logger.info("Using Google Gemini (model=%s)", config.GEMINI_MODEL)
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=temperature,
    )
