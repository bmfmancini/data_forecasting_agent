"""Shared SQLite infrastructure for backend persistent state.

The backend keeps its relational state in one SQLite database. Domain modules
own their queries, while this module owns connection policy and schema setup.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

import core.config as settings

_BUSY_TIMEOUT_MS = 5_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    api_key_hash  TEXT    NOT NULL,
    description   TEXT    NOT NULL DEFAULT '',
    enabled       INTEGER NOT NULL DEFAULT 1,
    bootstrap     INTEGER NOT NULL DEFAULT 0,
    is_admin      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    last_used     TEXT,
    last_used_ip  TEXT
);

CREATE TABLE IF NOT EXISTS uploaded_files (
    file_id    TEXT PRIMARY KEY,
    owner_id   INTEGER,
    date_col   TEXT NOT NULL,
    value_col  TEXT NOT NULL,
    freq       TEXT NOT NULL,
    filename   TEXT,
    active_jobs INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (owner_id) REFERENCES api_users(id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS uploaded_files_owner_created_idx
ON uploaded_files(owner_id, created_at);

CREATE TABLE IF NOT EXISTS forecast_jobs (
    job_id                    TEXT PRIMARY KEY,
    backend_owner_id          INTEGER,
    application_user_id       INTEGER,
    application_username      TEXT NOT NULL DEFAULT '',
    application_user_is_admin INTEGER NOT NULL DEFAULT 0,
    file_id                   TEXT NOT NULL,
    date_col                  TEXT NOT NULL,
    value_col                 TEXT NOT NULL,
    forecast_horizon          INTEGER NOT NULL,
    forced_model              TEXT,
    user_prompt               TEXT,
    preflight_options         TEXT NOT NULL DEFAULT '{}',
    status                    TEXT NOT NULL,
    progress                  INTEGER NOT NULL DEFAULT 0,
    step                      TEXT NOT NULL,
    error                     TEXT,
    queued_at                 TEXT NOT NULL DEFAULT (datetime('now')),
    started_at                TEXT,
    completed_at              TEXT,
    FOREIGN KEY (backend_owner_id) REFERENCES api_users(id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS forecast_jobs_status_queued_idx
ON forecast_jobs(status, queued_at);

CREATE INDEX IF NOT EXISTS forecast_jobs_application_status_idx
ON forecast_jobs(application_user_id, status);

CREATE TABLE IF NOT EXISTS forecast_job_settings (
    singleton                 INTEGER PRIMARY KEY CHECK (singleton = 1),
    max_running_jobs_per_user INTEGER NOT NULL DEFAULT 1,
    retention_days            INTEGER,
    cleanup_enabled           INTEGER NOT NULL DEFAULT 1,
    updated_at                TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO forecast_job_settings
    (singleton, max_running_jobs_per_user, retention_days, cleanup_enabled)
VALUES (1, 1, 30, 1);
"""


def _database_path(db_path: str | None = None) -> str:
    """Return a database path and create its parent directory when needed."""
    path = db_path or settings.BACKEND_DB_PATH
    if path != ":memory:" and not path.startswith("file:"):
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
    return path


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Open a configured SQLite connection for backend persistence.

    Args:
        db_path: Optional SQLite file path. Defaults to ``BACKEND_DB_PATH``.

    Returns:
        A connection with row access, foreign keys, WAL, and busy timeout
        configured.
    """
    path = _database_path(db_path)
    connection = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    return connection


@contextmanager
def transaction(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a database transaction and commit or roll it back safely.

    Args:
        db_path: Optional SQLite file path. Defaults to ``BACKEND_DB_PATH``.

    Yields:
        A configured SQLite connection with an active transaction.
    """
    connection = get_connection(db_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_database() -> None:
    """Create all backend persistence tables and indexes."""
    with transaction() as connection:
        connection.executescript(_SCHEMA)
