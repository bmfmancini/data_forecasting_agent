"""Custom domain-specific exceptions for the Data Forecasting Agent backend.

All validation and configuration errors raised by backend modules should
inherit from :class:`ForecastingAgentError` so callers can catch the family
with a single ``except`` clause when desired.
"""

from __future__ import annotations


class ForecastingAgentError(Exception):
    """Base class for all custom backend exceptions."""


class DataValidationError(ForecastingAgentError, ValueError):
    """Raised when uploaded or transformed data violates forecast requirements."""


class LLMConfigError(ForecastingAgentError):
    """Raised when the LLM provider configuration is invalid or incomplete.

    For example, when ``USE_OLLAMA_CLOUD`` is enabled but no
    ``OLLAMA_API_KEY`` has been supplied.
    """


class LLMDisabledError(ForecastingAgentError):
    """Raised when an LLM is constructed while AI features are disabled.

    The deployment-wide "Enable AI features" switch (``system_settings.
    llm_enabled``) is off.  :func:`core.llm_factory.get_llm` raises this
    before building a client so that any LLM call not yet dispatched —
    including calls inside a forecast that was already running when an
    administrator flipped the setting — fails fast at construction and the
    calling agent's deterministic fallback path takes over.
    """


class PipelineExecutionError(ForecastingAgentError):
    """Raised when an expected pipeline stage cannot complete successfully."""


class StorageAccessError(ForecastingAgentError):
    """Raised when persisted forecast artifacts cannot be read or written safely."""
