"""SQLite database helpers for API key user management.

Provides CRUD operations for the ``api_users`` table and the first-run
bootstrap mechanism that creates an initial API credential.

Follows the same raw-``sqlite3`` pattern used by the Flask frontend's
``db/db.py`` — no ORM, parameterized queries, ``Row`` row factory.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any

import core.config as settings
from auth.argon2_helpers import generate_api_key, hash_api_key
from core.logging_config import get_logger

logger = get_logger(__name__)

_BOOTSTRAP_USERNAME: str = "frontend"
_BOOTSTRAP_DESCRIPTION: str = "Bootstrap API user (auto-created on first run)"

_SCHEMA: str = """
CREATE TABLE IF NOT EXISTS api_users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    api_key_hash  TEXT    NOT NULL,
    description   TEXT    NOT NULL DEFAULT '',
    enabled       INTEGER NOT NULL DEFAULT 1,
    bootstrap     INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    last_used     TEXT,
    last_used_ip  TEXT
);
"""


def _get_db_path() -> str:
    """Return the filesystem path to the API key SQLite database.

    Returns:
        Absolute path to ``api_keys.db`` inside the configured data
        directory.
    """
    db_dir: str = settings.API_KEY_DB_PATH
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "api_keys.db")


def _get_connection() -> sqlite3.Connection:
    """Open a new SQLite connection with ``Row`` row factory.

    Returns:
        A :class:`sqlite3.Connection` configured for dict-style row
        access.
    """
    conn: sqlite3.Connection = sqlite3.connect(
        _get_db_path(),
        detect_types=sqlite3.PARSE_DECLTYPES,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create the ``api_users`` table if needed and run bootstrap.

    Called once during FastAPI application startup.  Creates the schema
    and, when no API users exist, generates the initial bootstrap
    credential and prints it to stdout.
    """
    conn: sqlite3.Connection = _get_connection()
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()

    bootstrap_api_user()


def bootstrap_api_user() -> None:
    """Create an initial API user when the database is empty.

    If one or more API users already exist, this function does nothing
    (ensuring restarts do not regenerate credentials).  When the table
    is empty, a ``frontend`` user is created with a cryptographically
    secure random key hashed via Argon2id.  The plaintext key is printed
    once to stdout and never stored or logged again.
    """
    conn: sqlite3.Connection = _get_connection()
    try:
        count_row: sqlite3.Row | None = conn.execute(
            "SELECT COUNT(*) AS cnt FROM api_users"
        ).fetchone()
        count: int = int(count_row["cnt"]) if count_row else 0

        if count > 0:
            logger.info(
                "API key DB already has %d user(s) — skipping bootstrap.",
                count,
            )
            return

        plaintext_key: str = generate_api_key()
        key_hash: str = hash_api_key(plaintext_key)

        conn.execute(
            """
            INSERT INTO api_users
                (username, api_key_hash, description, enabled, bootstrap)
            VALUES (?, ?, ?, 1, 1)
            """,
            (_BOOTSTRAP_USERNAME, key_hash, _BOOTSTRAP_DESCRIPTION),
        )
        conn.commit()

        _print_bootstrap_banner(_BOOTSTRAP_USERNAME, plaintext_key)
        logger.info("Bootstrap API user '%s' created.", _BOOTSTRAP_USERNAME)
    finally:
        conn.close()


def _print_bootstrap_banner(username: str, api_key: str) -> None:
    """Print the one-time bootstrap credential banner to stdout.

    Args:
        username: The bootstrap username.
        api_key:  The plaintext API key (displayed once, never stored).
    """
    banner: str = f"""
========================================

Initial API Credentials Created

Username:
{username}

API Key:
{api_key}

This key will only be displayed once.

Log into the Admin panel and rotate
or replace this credential.

========================================
"""
    print(banner, flush=True)


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
    from auth.argon2_helpers import verify_api_key as _verify

    conn: sqlite3.Connection = _get_connection()
    try:
        row: sqlite3.Row | None = conn.execute(
            "SELECT * FROM api_users WHERE username = ?",
            (username,),
        ).fetchone()

        if row is None:
            return None

        if not int(row["enabled"]):
            return None

        if not _verify(api_key, str(row["api_key_hash"])):
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
            "created_at": str(row["created_at"]),
            "last_used": str(row["last_used"]) if row["last_used"] else None,
            "last_used_ip": (
                str(row["last_used_ip"]) if row["last_used_ip"] else None
            ),
        }
    finally:
        conn.close()


def list_api_users() -> list[dict[str, Any]]:
    """Return all API users without exposing key hashes.

    Returns:
        List of user dicts ordered by ``id``.
    """
    conn: sqlite3.Connection = _get_connection()
    try:
        rows: list[sqlite3.Row] = conn.execute(
            """
            SELECT id, username, description, enabled, bootstrap,
                   created_at, last_used, last_used_ip
            FROM api_users
            ORDER BY id
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_api_user(user_id: int) -> dict[str, Any] | None:
    """Return a single API user by ID without the key hash.

    Args:
        user_id: Primary key of the API user.

    Returns:
        User dict or ``None`` when not found.
    """
    conn: sqlite3.Connection = _get_connection()
    try:
        row: sqlite3.Row | None = conn.execute(
            """
            SELECT id, username, description, enabled, bootstrap,
                   created_at, last_used, last_used_ip
            FROM api_users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_api_user(username: str, description: str) -> str:
    """Create a new API user and return the plaintext key (once).

    Args:
        username:    Unique username for the new API user.
        description: Human-readable description / purpose.

    Returns:
        The plaintext API key — displayed once to the caller.

    Raises:
        ValueError: When the username already exists.
    """
    plaintext_key: str = generate_api_key()
    key_hash: str = hash_api_key(plaintext_key)

    conn: sqlite3.Connection = _get_connection()
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
                (username, api_key_hash, description, enabled, bootstrap)
            VALUES (?, ?, ?, 1, 0)
            """,
            (username, key_hash, description),
        )
        conn.commit()
        logger.info("API user '%s' created.", username)
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

    conn: sqlite3.Connection = _get_connection()
    try:
        row: sqlite3.Row | None = conn.execute(
            "SELECT id FROM api_users WHERE id = ?",
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
    conn: sqlite3.Connection = _get_connection()
    try:
        row: sqlite3.Row | None = conn.execute(
            "SELECT id FROM api_users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"API user with id {user_id} not found.")

        conn.execute(
            "UPDATE api_users SET enabled = ? WHERE id = ?",
            (int(enabled), user_id),
        )
        conn.commit()
        logger.info(
            "API user id=%d %s.", user_id, "enabled" if enabled else "disabled"
        )
    finally:
        conn.close()


def delete_api_user(user_id: int) -> None:
    """Permanently delete an API user.

    Args:
        user_id: Primary key of the API user to delete.

    Raises:
        ValueError: When the user ID does not exist.
    """
    conn: sqlite3.Connection = _get_connection()
    try:
        row: sqlite3.Row | None = conn.execute(
            "SELECT id FROM api_users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"API user with id {user_id} not found.")

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
    conn: sqlite3.Connection = _get_connection()
    try:
        row: sqlite3.Row | None = conn.execute(
            "SELECT COUNT(*) AS cnt FROM api_users WHERE bootstrap = 1"
        ).fetchone()
        return int(row["cnt"]) > 0 if row else False
    finally:
        conn.close()