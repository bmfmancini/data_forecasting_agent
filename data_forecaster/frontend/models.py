"""
User model for Flask-Login integration.

Defines the ``User`` class that wraps a database row and satisfies the
Flask-Login ``UserMixin`` interface.
"""

from __future__ import annotations

from flask_login import UserMixin


class User(UserMixin):  # type: ignore[misc]
    """Authenticated user object populated from the ``users`` database table.

    Args:
        user_id:   Primary key from the ``users`` table.
        username:  The user's login name.
        role_name: The name of the assigned role (e.g. ``'admin'``).
        active:    Whether the account is enabled.
    """

    def __init__(
        self,
        user_id: int,
        username: str,
        role_name: str,
        active: bool,
    ) -> None:
        self._id = user_id
        self.username = username
        self.role_name = role_name
        self._active = active

    @property
    def id(self) -> int:
        """Integer primary key — overrides the UserMixin string property."""
        return self._id

    @property
    def is_active(self) -> bool:
        """Return whether the account is enabled."""
        return self._active

    @property
    def is_admin(self) -> bool:
        """Return whether the user holds the admin role."""
        return self.role_name == "admin"

    def get_id(self) -> str:
        """Return the user identifier as a string (required by Flask-Login)."""
        return str(self._id)
