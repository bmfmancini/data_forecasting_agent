"""Custom domain-specific exceptions for the Data Forecasting Agent backend.

All validation and configuration errors raised by backend modules should
inherit from :class:`ForecastingAgentError` so callers can catch the family
with a single ``except`` clause when desired.
"""

from __future__ import annotations


class ForecastingAgentError(Exception):
    """Base class for all custom backend exceptions."""


class LLMConfigError(ForecastingAgentError):
    """Raised when the LLM provider configuration is invalid or incomplete.

    For example, when ``USE_OLLAMA_CLOUD`` is enabled but no
    ``OLLAMA_API_KEY`` has been supplied.
    """
