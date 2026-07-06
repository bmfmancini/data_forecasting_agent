"""Decorators for Flask blueprint route protection.

Provides :func:`password_change_required` which redirects users to the
password-change page when their ``must_change_password`` flag is set.
"""

from __future__ import annotations

from functools import wraps
from typing import Callable, TypeVar

from flask import flash, redirect, request, url_for
from flask_login import current_user

_F = TypeVar("_F", bound=Callable[..., ...])


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
