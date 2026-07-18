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


class PipelineExecutionError(ForecastingAgentError):
    """Raised when an expected pipeline stage cannot complete successfully."""


class StorageAccessError(ForecastingAgentError):
    """Raised when persisted forecast artifacts cannot be read or written safely."""


class JobCancelledError(ForecastingAgentError):
    """Raised when a pipeline stage detects a cooperative cancellation request.

    This is raised inside :func:`run_pipeline` when the ``cancel_check``
    callback returns ``True`` at a stage boundary.  The job worker catches it
    and transitions the job to the ``cancelled`` terminal status.
    """


class ForecastResourceError(ForecastingAgentError):
    """Raised when a forecast cannot fit within configured memory capacity."""
