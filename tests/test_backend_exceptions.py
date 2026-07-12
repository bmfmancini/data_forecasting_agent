"""Tests for backend domain exception contracts."""

from __future__ import annotations

import os
import sys

_backend_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data_forecaster", "backend")
)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from exceptions import DataValidationError, ForecastingAgentError  # noqa: E402


def test_data_validation_error_is_domain_and_value_error() -> None:
    """Validation failures can be caught by domain or value-error handlers."""
    exc = DataValidationError("bad data")

    assert isinstance(exc, ForecastingAgentError)
    assert isinstance(exc, ValueError)
