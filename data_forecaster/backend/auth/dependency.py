"""FastAPI dependency for API key authentication.

Provides :func:`require_api_key` — a reusable dependency that extracts
``X-API-Username`` and ``X-API-Key`` headers from the request, validates
them against the SQLite database, and raises a generic 401 on any
failure.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request, status

import core.config as settings
from auth.api_key_db import verify_api_key


def require_api_key(request: Request) -> dict[str, Any]:
    """Validate API key credentials from request headers.

    Extracts ``X-API-Username`` and ``X-API-Key`` headers, looks up the
    username in the database, verifies the account is enabled, and
    checks the supplied key against the stored Argon2id hash.  On
    success, updates ``last_used`` and ``last_used_ip`` and returns the
    user dict.  On any failure, raises a generic 401 that does not
    reveal whether the username or key was invalid.

    When ``settings.API_KEY_ENABLED`` is ``False`` (e.g. local dev),
    the dependency is a no-op and returns an empty dict.

    Args:
        request: The incoming :class:`fastapi.Request`.

    Returns:
        A dict with the authenticated user's fields, or an empty dict
        when auth is disabled.

    Raises:
        HTTPException: 401 Unauthorized when credentials are missing,
            invalid, or the account is disabled.
    """
    if not settings.API_KEY_ENABLED:
        return {}

    username: str | None = request.headers.get("X-API-Username")
    api_key: str | None = request.headers.get("X-API-Key")

    if not username or not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )

    client_ip: str | None = None
    if request.client:
        client_ip = request.client.host

    user: dict[str, Any] | None = verify_api_key(
        username, api_key, client_ip=client_ip
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )

    return user