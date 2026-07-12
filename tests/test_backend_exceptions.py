"""Tests for backend domain exception contracts."""

from __future__ import annotations

from exceptions import DataValidationError, ForecastingAgentError


def test_data_validation_error_is_domain_and_value_error() -> None:
    """Validation failures can be caught by domain or value-error handlers."""
    exc = DataValidationError("bad data")

    assert isinstance(exc, ForecastingAgentError)
    assert isinstance(exc, ValueError)
