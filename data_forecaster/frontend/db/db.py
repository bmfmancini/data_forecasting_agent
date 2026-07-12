"""
Database access helpers for the Flask forecaster frontend.

Provides a per-request SQLite connection managed through Flask's application
context, plus lightweight query helpers and schema initialisation.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any, cast

import click
from flask import Flask, current_app, g
from werkzeug.security import generate_password_hash

from db.crypto import encrypt


def get_db() -> sqlite3.Connection:
    """Return the SQLite connection for the current application context.

    A new connection is opened on the first call within each request and
    reused for the lifetime of that request.
    """
    if "db" not in g:
        g.db = sqlite3.connect(
            current_app.config["DATABASE"],
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return cast(sqlite3.Connection, g.db)


def close_db(e: BaseException | None = None) -> None:
    """Close the per-request database connection if one was opened."""
    db: sqlite3.Connection | None = g.pop("db", None)
    if db is not None:
        db.close()


def query_db(
    sql: str,
    args: tuple[Any, ...] = (),
    one: bool = False,
) -> list[dict[str, Any]] | dict[str, Any] | None:
    """Execute a SELECT query and return the results as plain dicts.

    Args:
        sql:  SQL query string with ``?`` placeholders.
        args: Positional parameters bound to the placeholders.
        one:  When *True*, return only the first row or ``None``.

    Returns:
        A list of row dicts, a single row dict, or ``None`` when *one* is
        ``True`` and no rows match.
    """
    cur = get_db().execute(sql, args)
    rows = [dict(row) for row in cur.fetchall()]
    if one:
        return rows[0] if rows else None
    return rows


def execute_db(sql: str, args: tuple[Any, ...] = ()) -> int:
    """Execute a write statement and return the last inserted row ID.

    The change is committed immediately.

    Args:
        sql:  SQL statement with ``?`` placeholders.
        args: Positional parameters bound to the placeholders.

    Returns:
        The ``lastrowid`` of the executed statement.
    """
    db = get_db()
    cur = db.execute(sql, args)
    db.commit()
    return cur.lastrowid or 0


def init_db() -> None:
    """Initialise the database schema and seed default data.

    Reads ``db/schema.sql`` relative to the application root, executes the
    DDL, then inserts seed rows (roles, default admin user, default API
    credential entry) only when they do not already exist.

    The default admin username and password are read from
    ``FRONTEND_ADMIN_USERNAME`` and ``FRONTEND_ADMIN_PASSWORD`` env vars
    (documented in ``.env.example``).  The seeded admin is created with
    ``must_change_password = 1`` so the operator is forced to rotate the
    password on first login.
    """
    db = get_db()

    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, encoding="utf-8") as f:
        db.executescript(f.read())

    # ``CREATE TABLE IF NOT EXISTS`` does not add columns to installations
    # created by earlier releases, so apply this additive migration here.
    report_columns = {
        row["name"] for row in db.execute("PRAGMA table_info(forecast_reports)")
    }
    if "custom_settings_json" not in report_columns:
        db.execute("ALTER TABLE forecast_reports ADD COLUMN custom_settings_json TEXT")

    user_columns = {row["name"] for row in db.execute("PRAGMA table_info(users)")}
    if "session_version" not in user_columns:
        db.execute("ALTER TABLE users ADD COLUMN session_version INTEGER NOT NULL DEFAULT 0")

    db.execute("INSERT OR IGNORE INTO roles (id, name) VALUES (1, 'admin')")
    db.execute("INSERT OR IGNORE INTO roles (id, name) VALUES (2, 'user')")

    admin_username = current_app.config["FRONTEND_ADMIN_USERNAME"]
    admin_password = current_app.config["FRONTEND_ADMIN_PASSWORD"]
    # Only seed the default admin when credentials are explicitly provided
    # (development/testing).  In production, the admin must set credentials
    # via environment variables or the admin panel.
    if admin_username and admin_password:
        admin_hash = generate_password_hash(admin_password)
        db.execute(
            """
            INSERT OR IGNORE INTO users
                (username, password_hash, role_id, active, must_change_password)
            VALUES (?, ?, 1, 1, 1)
            """,
            (admin_username, admin_hash),
        )

    backend_url = current_app.config.get("BACKEND_URL", "http://localhost:8000")
    verify_ssl = 1 if current_app.config.get("API_VERIFY_SSL", True) else 0

    # Auto-seed pre-shared backend credentials from env vars so the
    # admin does not have to enter them manually on first boot.  Only
    # seeds when both username and key are present and no credentials
    # are already stored for the 'default' label.
    seed_user: str = current_app.config.get("FRONTEND_API_USERNAME", "")
    seed_key: str = current_app.config.get("FRONTEND_API_KEY", "")
    enc_user: str | None = None
    enc_pass: str | None = None
    if seed_user and seed_key:
        try:
            enc_user = encrypt(seed_user)
            enc_pass = encrypt(seed_key)
        except RuntimeError:
            enc_user = None
            enc_pass = None

    if enc_user and enc_pass:
        db.execute(
            """
            INSERT INTO api_credentials
                (label, base_url, encrypted_username, encrypted_password,
                 timeout, verify_ssl)
            VALUES ('default', ?, ?, ?, 30, ?)
            ON CONFLICT(label) DO UPDATE SET
                encrypted_username = excluded.encrypted_username,
                encrypted_password = excluded.encrypted_password
            WHERE excluded.encrypted_username IS NOT NULL
            """,
            (backend_url, enc_user, enc_pass, verify_ssl),
        )
    else:
        # Only update base_url and verify_ssl if they are not already set
        db.execute(
            """
            INSERT INTO api_credentials (label, base_url, timeout, verify_ssl)
            VALUES ('default', ?, 30, ?)
            ON CONFLICT(label) DO NOTHING
            """,
            (backend_url, verify_ssl),
        )

    db.execute("""
        INSERT OR IGNORE INTO app_config (key, value)
        VALUES ('app_name', 'Time Series Data Forecaster Agent')
        """)
    db.execute("""
        INSERT OR IGNORE INTO app_config (key, value)
        VALUES ('max_reports_per_user', '10')
        """)

    db.commit()


def init_app(app: Flask) -> None:
    """Register database teardown and CLI commands with *app*.

    Args:
        app: The Flask application instance.
    """
    app.teardown_appcontext(close_db)

    @app.cli.command("init-db")
    def init_db_command() -> None:
        """Initialise the database schema and seed default data."""
        with app.app_context():
            init_db()
        click.echo("Database initialised.")
