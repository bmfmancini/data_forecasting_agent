"""Decorators for Flask blueprint route protection.

Provides :func:`password_change_required` which redirects users to the
password-change page when their ``must_change_password`` flag is set, and
:func:`get_backend_setup_status` which powers the first-run setup gate.
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, TypeVar

import requests
from flask import flash, redirect, request, url_for
from flask_login import current_user

from services.api_client import BackendAPIClient, resolve_backend_connection

_F = TypeVar("_F", bound=Callable[..., ...])

logger = logging.getLogger(__name__)


def get_backend_setup_status() -> dict[str, Any]:
    """Probe the backend ``GET /setup/status`` endpoint.

    Builds a lightweight, unauthenticated client — the endpoint requires no
    auth, so the probe works before setup completes and before any
    credentials are stored.  Connection errors are tolerated and reported
    as "setup incomplete" so the wizard's backend-connection step can
    handle them.

    The connection settings come from the ``api_credentials`` DB row first
    (via :func:`resolve_backend_connection`) so every gunicorn worker
    agrees on the backend URL, with the in-process config as fallback.

    Returns:
        The parsed status payload, or ``{"setup_complete": False}`` when
        the backend URL is not configured, the backend is unreachable, or
        the response is unexpected.
    """
    base_url, verify_ssl = resolve_backend_connection()
    if not base_url:
        return {"setup_complete": False}

    client = BackendAPIClient(base_url=base_url, verify=verify_ssl)
    try:
        resp = client.get_setup_status()
        if resp.status_code == 200:
            data: dict[str, Any] = resp.json()
            return data
    except (requests.RequestException, ValueError):
        logger.debug("Setup status probe failed — treating as incomplete.")
    return {"setup_complete": False}


def password_change_required(f: _F) -> _F:
    """Redirect to the password-change page if the user must change their password.

    Args:
        f: The route function to wrap.

    Returns:
        The wrapped function that enforces the password-change precondition.
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for("auth.login", next=request.url))

        if not getattr(current_user, "must_change_password", False):
            return f(*args, **kwargs)

        if not request.endpoint or request.endpoint == "auth.change_password":
            return f(*args, **kwargs)

        flash("You must change your password before you can continue.", "warning")
        return redirect(url_for("auth.change_password"))

    return decorated_function
