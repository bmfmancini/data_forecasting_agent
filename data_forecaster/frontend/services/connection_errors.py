"""Safe user-facing messages for backend connection failures."""

from __future__ import annotations

import re

_HEADER_SECRET_RE: re.Pattern[str] = re.compile(
    r"(authorization)(['\":=\s]+)(?:[A-Za-z]+\s+)?([^,\s'\"}]+)"
    r"|"
    r"(x-api-key|api[_-]?key)(['\":=\s]+)([^,\s'\"}]+)",
    re.IGNORECASE,
)
_URL_CREDENTIAL_RE: re.Pattern[str] = re.compile(r"//[^/@\s]+@")
_LONG_TOKEN_RE: re.Pattern[str] = re.compile(r"\b[A-Za-z0-9_-]{24,}\b")


def sanitize_connection_error(error_message: str) -> str:
    """Return a short connection error message safe for browser display.

    Args:
        error_message: Raw exception text from a backend request failure.

    Returns:
        A generic, user-actionable message that does not include credentials,
        tokens, stack traces, or internal request details.
    """

    def _redact_header(match: re.Match[str]) -> str:
        if match.group(1):
            return f"{match.group(1)}{match.group(2)}[redacted]"
        return f"{match.group(4)}{match.group(5)}[redacted]"

    redacted = _HEADER_SECRET_RE.sub(_redact_header, error_message)
    redacted = _URL_CREDENTIAL_RE.sub("//[redacted]@", redacted)
    redacted = _LONG_TOKEN_RE.sub("[redacted]", redacted)
    lowered = redacted.lower()

    if "connection refused" in lowered or "connecterror" in lowered:
        return (
            "Could not connect to the backend. Check the Base URL and ensure "
            "the service is running."
        )
    if "timeout" in lowered or "timed out" in lowered:
        return "Connection timed out. The backend may be overloaded or unreachable."
    if "401" in lowered or "unauthorized" in lowered:
        return "Authentication failed. Check the configured credentials."
    return "Connection failed. Verify the backend URL and network accessibility."
