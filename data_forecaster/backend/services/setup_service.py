"""First-run setup service for the Data Forecaster backend.

Owns the atomic bootstrap that creates the first admin API user, generates
the backend encryption key, and flips ``setup_complete``. Also owns the
startup migration that seeds DB-backed configuration for deployments that
pre-date the setup wizard.

The bootstrap is race-safe: the guard is a conditional ``INSERT ... WHERE
NOT EXISTS`` inside a ``BEGIN IMMEDIATE`` transaction (the same pattern as
``services.job_service._claim_job``), so two simultaneous first-run
requests cannot both succeed.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from auth.argon2_helpers import hash_api_key
from core import secret_store
from core.database import get_connection
from core.llm_config_store import is_configured
from core.logging_config import get_logger

logger = get_logger(__name__)

_BOOTSTRAP_DESCRIPTION: str = "Admin API user (created via setup wizard)"


class SetupAlreadyCompleteError(Exception):
    """Raised when bootstrap is attempted after setup has completed."""


def is_setup_complete(db_path: str | None = None) -> bool:
    """Return whether first-run setup has completed.

    Args:
        db_path: Optional database path override (testing).

    Returns:
        ``True`` when the ``system_settings.setup_complete`` flag is set.
    """
    with get_connection(db_path) as connection:
        row: sqlite3.Row | None = connection.execute(
            "SELECT setup_complete FROM system_settings WHERE singleton = 1"
        ).fetchone()
        return bool(row["setup_complete"]) if row else False


def mark_setup_complete(db_path: str | None = None) -> None:
    """Set the ``setup_complete`` flag (idempotent)."""
    with get_connection(db_path) as connection:
        connection.execute(
            "UPDATE system_settings SET setup_complete = 1,"
            " updated_at = datetime('now') WHERE singleton = 1"
        )
        connection.commit()


def get_setup_status(db_path: str | None = None) -> dict[str, Any]:
    """Return setup completion state without exposing any secrets.

    Args:
        db_path: Optional database path override (testing).

    Returns:
        A dict with ``setup_complete``, ``admin_exists``,
        ``llm_configured``, and ``models_enabled`` booleans/counts.
    """
    with get_connection(db_path) as connection:
        user_row: sqlite3.Row | None = connection.execute(
            "SELECT COUNT(*) AS cnt FROM api_users WHERE is_admin = 1"
        ).fetchone()
        model_row: sqlite3.Row | None = connection.execute(
            "SELECT COUNT(*) AS cnt FROM model_config WHERE enabled = 1"
        ).fetchone()
    admin_exists = bool(user_row and int(user_row["cnt"]) > 0)
    enabled_models = int(model_row["cnt"]) if model_row else 0
    return {
        "setup_complete": is_setup_complete(db_path),
        "admin_exists": admin_exists,
        "llm_configured": is_configured(db_path),
        "models_enabled": enabled_models,
    }


def run_bootstrap(
    username: str,
    api_key: str,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Atomically create the first admin API user and complete setup.

    Generates and persists the backend encryption key (skipped when already
    present), inserts the admin user only when no API users exist, and sets
    ``setup_complete`` — all inside a single ``BEGIN IMMEDIATE``
    transaction so concurrent first-run requests cannot both succeed.

    Args:
        username: Username for the first admin API user.
        api_key:  Plaintext API key chosen by the admin.
        db_path:  Optional database path override (testing).

    Returns:
        A dict with the new user's fields (no key material).

    Raises:
        ValueError: When the username or key is empty.
        SetupAlreadyCompleteError: When API users already exist.
    """
    if not username or not username.strip():
        raise ValueError("Username is required.")
    if not api_key:
        raise ValueError("API key is required.")

    connection: sqlite3.Connection = get_connection(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        cursor: sqlite3.Cursor = connection.execute(
            """
            INSERT INTO api_users
                (username, api_key_hash, description, enabled, bootstrap,
                 is_admin)
            SELECT ?, ?, ?, 1, 1, 1
            WHERE NOT EXISTS (SELECT 1 FROM api_users)
            """,
            (username.strip(), hash_api_key(api_key), _BOOTSTRAP_DESCRIPTION),
        )
        if cursor.rowcount == 0:
            connection.rollback()
            raise SetupAlreadyCompleteError(
                "API users already exist — setup bootstrap is no longer "
                "available."
            )
        connection.execute(
            "UPDATE system_settings SET setup_complete = 1,"
            " updated_at = datetime('now') WHERE singleton = 1"
        )
        connection.commit()
    except SetupAlreadyCompleteError:
        raise
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    secret_store.generate_and_persist_key()
    logger.info("Setup bootstrap complete. Admin user '%s' created.", username)

    with get_connection(db_path) as read_conn:
        row: sqlite3.Row | None = read_conn.execute(
            """
            SELECT id, username, description, enabled, bootstrap, is_admin,
                   created_at, last_used, last_used_ip
            FROM api_users WHERE username = ?
            """,
            (username.strip(),),
        ).fetchone()
    return dict(row) if row else {}


def migrate_legacy_deployment(db_path: str | None = None) -> None:
    """Seed DB-backed configuration for pre-wizard deployments.

    When API users already exist but ``setup_complete`` is false, the
    deployment predates the setup wizard: mark setup complete so the
    env-based service-user reconciliation is skipped from now on. The
    ``model_config`` seed rows are created by ``init_database`` itself.

    Args:
        db_path: Optional database path override (testing).
    """
    with get_connection(db_path) as connection:
        row: sqlite3.Row | None = connection.execute(
            "SELECT COUNT(*) AS cnt FROM api_users"
        ).fetchone()
    if row and int(row["cnt"]) > 0 and not is_setup_complete(db_path):
        mark_setup_complete(db_path)
        logger.info(
            "Existing API users found — marking setup complete (legacy "
            "deployment migration)."
        )
