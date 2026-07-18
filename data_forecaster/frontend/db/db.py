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
    DDL and static seed data (roles, api_credentials, app_config defaults),
    then applies additive column migrations for older installations and
    seeds the bootstrap admin user when no users exist.

    On first initialisation, the database seeds a single ``admin`` user with
    the password from ``FRONTEND_DEFAULT_ADMIN_PASSWORD`` (default ``admin``)
    and ``must_change_password = 1`` so the operator is forced to rotate the
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
    if "job_id" not in report_columns:
        db.execute("ALTER TABLE forecast_reports ADD COLUMN job_id TEXT")
    db.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS forecast_reports_job_id_uq "
        "ON forecast_reports(job_id) WHERE job_id IS NOT NULL"
    )

    user_columns = {row["name"] for row in db.execute("PRAGMA table_info(users)")}
    if "session_version" not in user_columns:
        db.execute(
            "ALTER TABLE users ADD COLUMN session_version INTEGER NOT NULL DEFAULT 0"
        )

    # Seed the bootstrap admin user only when no users exist.  Roles,
    # api_credentials, and app_config defaults are seeded in schema.sql.
    user_count_row = db.execute("SELECT COUNT(*) AS count FROM users").fetchone()
    user_count = int(user_count_row["count"]) if user_count_row else 0
    if user_count == 0:
        # Bootstrap-only: used for the initial admin login and invalidated
        # immediately by ``must_change_password = 1``. Override via the
        # ``FRONTEND_DEFAULT_ADMIN_PASSWORD`` env var in production.
        default_admin_password = current_app.config["DEFAULT_ADMIN_PASSWORD"]
        admin_hash = generate_password_hash(default_admin_password)
        db.execute(
            """
            INSERT INTO users
                (username, password_hash, role_id, active, must_change_password)
            VALUES (?, ?, 1, 1, 1)
            """,
            ("admin", admin_hash),
        )

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
