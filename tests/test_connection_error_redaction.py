"""Tests for safe backend connection error messages."""

from __future__ import annotations

from data_forecaster.frontend.services.connection_errors import (
    sanitize_connection_error,
)


def test_sanitize_connection_error_redacts_api_keys() -> None:
    """Raw API keys should never be reflected back into browser messages."""
    message = (
        "Unauthorized 401 for X-API-Key: "
        "4qxOABc41uEL-5rO-zoKU4dWp3Sna29kyD7Ux-JHy1c"
    )

    sanitized = sanitize_connection_error(message)

    assert "4qxOABc41uEL" not in sanitized
    assert sanitized == "Authentication failed. Check the configured credentials."


def test_sanitize_connection_error_redacts_url_credentials() -> None:
    """Credentials embedded in a URL should not appear in the safe message."""
    message = (
        "ConnectError for https://frontend:"
        "4qxOABc41uEL-5rO-zoKU4dWp3Sna29kyD7Ux-JHy1c@backend"
    )

    sanitized = sanitize_connection_error(message)

    assert "4qxOABc41uEL" not in sanitized
    assert sanitized.startswith("Could not connect to the backend.")


def test_sanitize_connection_error_redacts_authorization_scheme_and_token() -> None:
    """Authorization schemes and short tokens should be redacted together."""
    message = "Unauthorized 401 for Authorization: Bearer short-token"

    sanitized = sanitize_connection_error(message)

    assert "Bearer" not in sanitized
    assert "short-token" not in sanitized
    assert sanitized == "Authentication failed. Check the configured credentials."
