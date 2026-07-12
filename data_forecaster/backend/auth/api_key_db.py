"""SQLite database helpers for API key user management.

Provides CRUD operations for the ``api_users`` table and the first-run
bootstrap mechanism that creates an initial API credential.

Follows the same raw-``sqlite3`` pattern used by the Flask frontend's
``db/db.py`` — no ORM, parameterized queries, ``Row`` row factory.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from auth.argon2_helpers import (
    generate_api_key,
    hash_api_key,
    verify_api_key as verify_hash,
)
from core.database import get_connection
from core.logging_config import get_logger

logger = get_logger(__name__)

_BOOTSTRAP_DESCRIPTION: str = "Bootstrap API user (created via admin bootstrap)"

_SELECT_USER_BY_ID: str = "SELECT id FROM api_users WHERE id = ?"


def has_any_users() -> bool:
    """Check whether any API users exist in the database.

    Returns:
        ``True`` when at least one user exists.
    """
    conn: sqlite3.Connection = get_connection()
    try:
        row: sqlite3.Row | None = conn.execute(
            "SELECT COUNT(*) AS cnt FROM api_users"
        ).fetchone()
        return int(row["cnt"]) > 0 if row else False
    finally:
        conn.close()


def create_first_user(username: str, api_key: str) -> dict[str, Any]:
    """Create the first API user and return a user dict (no key hash).

    The bootstrap user is always created as an administrator so that the
    first API credential can manage subsequent API users.

    Args:
        username:    Username for the first API user.
        api_key:     Plaintext API key chosen by the admin.

    Returns:
        A dict with the new user's fields (``id``, ``username``, etc.).

    Raises:
        ValueError: When users already exist (bootstrap is one-time only)
            or the username is empty.
    """
    if not username or not username.strip():
        raise ValueError("Username is required.")
    if not api_key:
        raise ValueError("API key is required.")

    conn: sqlite3.Connection = get_connection()
    try:
        count_row: sqlite3.Row | None = conn.execute(
            "SELECT COUNT(*) AS cnt FROM api_users"
        ).fetchone()
        count: int = int(count_row["cnt"]) if count_row else 0
        if count > 0:
            raise ValueError(
                "API users already exist — bootstrap is no longer available."
            )

        key_hash: str = hash_api_key(api_key)
        conn.execute(
            """
            INSERT INTO api_users
                (username, api_key_hash, description, enabled, bootstrap, is_admin)
            VALUES (?, ?, ?, 1, 1, 1)
            """,
            (username.strip(), key_hash, _BOOTSTRAP_DESCRIPTION),
        )
        conn.commit()
        logger.info("First API user '%s' created via bootstrap.", username)

        row: sqlite3.Row | None = conn.execute(
            """
            SELECT id, username, description, enabled, bootstrap, is_admin,
                   created_at, last_used, last_used_ip
            FROM api_users WHERE username = ?
            """,
            (username.strip(),),
        ).fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


def reconcile_service_user(username: str, api_key: str) -> bool:
    """Ensure an existing service API user matches the configured key.

    The frontend and backend can be rebuilt while their SQLite volumes
    persist. In that case environment variables may contain a new
    ``FRONTEND_API_KEY`` while the backend database still stores the old
    Argon2 hash. This helper makes startup idempotent for the named service
    account without creating users when none exist.

    Args:
        username: Existing service account username.
        api_key: Plaintext API key from the deployment environment.

    Returns:
        ``True`` when the user existed and is now enabled with a matching key;
        ``False`` when no matching user exists.

    Raises:
        ValueError: When username or api_key is empty.
    """
    if not username or not username.strip():
        raise ValueError("Username is required.")
    if not api_key:
        raise ValueError("API key is required.")

    clean_username = username.strip()
    conn: sqlite3.Connection = get_connection()
    try:
        row: sqlite3.Row | None = conn.execute(
            """
            SELECT id, api_key_hash, enabled
            FROM api_users
            WHERE username = ?
            """,
            (clean_username,),
        ).fetchone()
        if row is None:
            return False

        key_matches = verify_hash(api_key, str(row["api_key_hash"]))
        if key_matches and int(row["enabled"]):
            return True

        conn.execute(
            """
            UPDATE api_users
            SET api_key_hash = ?,
                enabled = 1
            WHERE id = ?
            """,
            (hash_api_key(api_key), int(row["id"])),
        )
        conn.commit()
        logger.info("Service API user '%s' reconciled from env.", clean_username)
        return True
    finally:
        conn.close()


def verify_api_key(
    username: str, api_key: str, client_ip: str | None = None
) -> dict[str, Any] | None:
    """Verify an API key against the database.

    Looks up the username, checks the account is enabled, verifies the
    supplied key against the stored Argon2id hash, and updates the
    ``last_used`` and ``last_used_ip`` columns on success.

    Args:
        username:  Username from the ``X-API-Username`` header.
        api_key:   Plaintext key from the ``X-API-Key`` header.
        client_ip: Optional client IP address for audit logging.

    Returns:
        A dict with user fields (``id``, ``username``, ``description``,
        ``enabled``, ``bootstrap``, ``created_at``, ``last_used``,
        ``last_used_ip``) when authentication succeeds, or ``None``
        when the username does not exist, the account is disabled, or
        the key does not match.
    """
    conn: sqlite3.Connection = get_connection()
    try:
        row: sqlite3.Row | None = conn.execute(
            """
            SELECT id, username, api_key_hash, description, enabled, bootstrap,
                   is_admin, created_at, last_used, last_used_ip
            FROM api_users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()

        if row is None:
            return None

        if not int(row["enabled"]):
            return None

        if not verify_hash(api_key, str(row["api_key_hash"])):
            return None

        conn.execute(
            """
            UPDATE api_users
            SET last_used = datetime('now'),
                last_used_ip = ?
            WHERE id = ?
            """,
            (client_ip, int(row["id"])),
        )
        conn.commit()

        return {
            "id": int(row["id"]),
            "username": str(row["username"]),
            "description": str(row["description"]),
            "enabled": bool(row["enabled"]),
            "bootstrap": bool(row["bootstrap"]),
            "is_admin": bool(row["is_admin"]),
            "created_at": str(row["created_at"]),
            "last_used": str(row["last_used"]) if row["last_used"] else None,
            "last_used_ip": (str(row["last_used_ip"]) if row["last_used_ip"] else None),
        }
    finally:
        conn.close()


def list_api_users() -> list[dict[str, Any]]:
    """Return all API users without exposing key hashes.

    Returns:
        List of user dicts ordered by ``id``.
    """
    conn: sqlite3.Connection = get_connection()
    try:
        rows: list[sqlite3.Row] = conn.execute(
            """
            SELECT id, username, description, enabled, bootstrap, is_admin,
                   created_at, last_used, last_used_ip
            FROM api_users
            ORDER BY id
            """
        ).fetchall()
        return [
            {
                "id": int(r["id"]),
                "username": str(r["username"]),
                "description": str(r["description"]),
                "enabled": bool(r["enabled"]),
                "bootstrap": bool(r["bootstrap"]),
                "is_admin": bool(r["is_admin"]),
                "created_at": str(r["created_at"]),
                "last_used": str(r["last_used"]) if r["last_used"] else None,
                "last_used_ip": (str(r["last_used_ip"]) if r["last_used_ip"] else None),
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_api_user(user_id: int) -> dict[str, Any] | None:
    """Return a single API user by ID without the key hash.

    Args:
        user_id: Primary key of the API user.

    Returns:
        User dict or ``None`` when not found.
    """
    conn: sqlite3.Connection = get_connection()
    try:
        row: sqlite3.Row | None = conn.execute(
            """
            SELECT id, username, description, enabled, bootstrap, is_admin,
                   created_at, last_used, last_used_ip
            FROM api_users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "username": str(row["username"]),
            "description": str(row["description"]),
            "enabled": bool(row["enabled"]),
            "bootstrap": bool(row["bootstrap"]),
            "is_admin": bool(row["is_admin"]),
            "created_at": str(row["created_at"]),
            "last_used": str(row["last_used"]) if row["last_used"] else None,
            "last_used_ip": (str(row["last_used_ip"]) if row["last_used_ip"] else None),
        }
    finally:
        conn.close()


def create_api_user(username: str, description: str, is_admin: bool = False) -> str:
    """Create a new API user and return the plaintext key (once).

    Args:
        username:    Unique username for the new API user.
        description: Human-readable description / purpose.
        is_admin:    Whether the new user is an administrator. Defaults to
            ``False`` (regular user).

    Returns:
        The plaintext API key — displayed once to the caller.

    Raises:
        ValueError: When the username already exists.
    """
    plaintext_key: str = generate_api_key()
    key_hash: str = hash_api_key(plaintext_key)

    conn: sqlite3.Connection = get_connection()
    try:
        existing: sqlite3.Row | None = conn.execute(
            "SELECT id FROM api_users WHERE username = ?",
            (username,),
        ).fetchone()
        if existing:
            raise ValueError(f"Username '{username}' already exists.")

        conn.execute(
            """
            INSERT INTO api_users
                (username, api_key_hash, description, enabled, bootstrap, is_admin)
            VALUES (?, ?, ?, 1, 0, ?)
            """,
            (username, key_hash, description, int(is_admin)),
        )
        conn.commit()
        logger.info("API user '%s' created (admin=%s).", username, is_admin)
        return plaintext_key
    finally:
        conn.close()


def rotate_api_key(user_id: int) -> str:
    """Generate a new API key for an existing user.

    Replaces the stored hash immediately, invalidating the old key.

    Args:
        user_id: Primary key of the API user.

    Returns:
        The new plaintext API key — displayed once to the caller.

    Raises:
        ValueError: When the user ID does not exist.
    """
    plaintext_key: str = generate_api_key()
    key_hash: str = hash_api_key(plaintext_key)

    conn: sqlite3.Connection = get_connection()
    try:
        row: sqlite3.Row | None = conn.execute(
            _SELECT_USER_BY_ID,
            (user_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"API user with id {user_id} not found.")

        conn.execute(
            "UPDATE api_users SET api_key_hash = ? WHERE id = ?",
            (key_hash, user_id),
        )
        conn.commit()
        logger.info("API key rotated for user_id=%d.", user_id)
        return plaintext_key
    finally:
        conn.close()


def set_user_enabled(user_id: int, enabled: bool) -> None:
    """Enable or disable an API user.

    Disabled users cannot authenticate.

    Args:
        user_id:  Primary key of the API user.
        enabled:  ``True`` to enable, ``False`` to disable.

    Raises:
        ValueError: When the user ID does not exist.
    """
    conn: sqlite3.Connection = get_connection()
    try:
        row: sqlite3.Row | None = conn.execute(
            _SELECT_USER_BY_ID,
            (user_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"API user with id {user_id} not found.")

        conn.execute(
            "UPDATE api_users SET enabled = ? WHERE id = ?",
            (int(enabled), user_id),
        )
        conn.commit()
        logger.info("API user id=%d %s.", user_id, "enabled" if enabled else "disabled")
    finally:
        conn.close()


def set_user_admin(user_id: int, is_admin: bool) -> None:
    """Promote or demote an API user.

    Args:
        user_id:   Primary key of the API user.
        is_admin:  ``True`` to grant administrator privileges,
            ``False`` to revoke them.

    Raises:
        ValueError: When the user ID does not exist.
    """
    conn: sqlite3.Connection = get_connection()
    try:
        row: sqlite3.Row | None = conn.execute(
            _SELECT_USER_BY_ID,
            (user_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"API user with id {user_id} not found.")

        conn.execute(
            "UPDATE api_users SET is_admin = ? WHERE id = ?",
            (int(is_admin), user_id),
        )
        conn.commit()
        logger.info("API user id=%d admin status set to %s.", user_id, is_admin)
    finally:
        conn.close()


def delete_api_user(user_id: int) -> None:
    """Permanently delete an API user.

    Args:
        user_id: Primary key of the API user to delete.

    Raises:
        ValueError: When the user ID does not exist, or when the user owns
            uploaded files or forecast jobs that prevent deletion.
    """
    conn: sqlite3.Connection = get_connection()
    try:
        row: sqlite3.Row | None = conn.execute(
            _SELECT_USER_BY_ID,
            (user_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"API user with id {user_id} not found.")

        owned_file = conn.execute(
            "SELECT 1 FROM uploaded_files WHERE owner_id = ? LIMIT 1", (user_id,)
        ).fetchone()
        if owned_file is not None:
            raise ValueError(
                f"API user with id {user_id} owns uploaded files and cannot be deleted."
            )

        owned_job = conn.execute(
            "SELECT 1 FROM forecast_jobs WHERE backend_owner_id = ? LIMIT 1",
            (user_id,),
        ).fetchone()
        if owned_job is not None:
            raise ValueError(
                f"API user with id {user_id} owns forecast jobs and cannot be deleted."
            )

        conn.execute("DELETE FROM api_users WHERE id = ?", (user_id,))
        conn.commit()
        logger.info("API user id=%d deleted.", user_id)
    finally:
        conn.close()


def has_bootstrap_user() -> bool:
    """Check whether any bootstrap-flagged API user still exists.

    Returns:
        ``True`` when at least one user has ``bootstrap = 1``.
    """
    conn: sqlite3.Connection = get_connection()
    try:
        row: sqlite3.Row | None = conn.execute(
            "SELECT COUNT(*) AS cnt FROM api_users WHERE bootstrap = 1"
        ).fetchone()
        return int(row["cnt"]) > 0 if row else False
    finally:
        conn.close()
