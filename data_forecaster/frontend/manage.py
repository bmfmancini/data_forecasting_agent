"""
CLI management commands for the Time Series Data Forecaster frontend.

Register these commands with the Flask application by calling
:func:`register_commands`.  They are then available via the ``flask`` CLI:

    flask user-create --username admin --password secret --role admin
    flask user-list
    flask credentials-set --label default --base-url http://backend:8000
    flask credentials-list
    flask generate-key
"""

from __future__ import annotations

import re

import click
from flask import Flask
from werkzeug.security import generate_password_hash

from blueprints.auth.forms import PASSWORD_COMPLEXITY_MESSAGE, PASSWORD_COMPLEXITY_RE


def register_commands(app: Flask) -> None:
    """Attach all CLI commands to *app*.

    Args:
        app: The Flask application instance to receive the commands.
    """

    @app.cli.command("user-create")
    @click.option("--username", required=True, help="Login username.")
    @click.option("--password", required=True, help="Initial password.")
    @click.option(
        "--role",
        default="user",
        type=click.Choice(["admin", "user"]),
        help="Role to assign.",
    )
    def user_create(username: str, password: str, role: str) -> None:
        """Create a new application user."""
        from db.db import execute_db, query_db

        with app.app_context():
            role_row = query_db(
                "SELECT id FROM roles WHERE name = ?", (role,), one=True
            )
            if not role_row or not isinstance(role_row, dict):
                click.echo(f"Role '{role}' not found.", err=True)
                raise SystemExit(1)

            role_id = int(role_row["id"])
            if len(password) < 8 or not re.match(PASSWORD_COMPLEXITY_RE, password):
                click.echo(
                    f"Password must be at least 8 characters. {PASSWORD_COMPLEXITY_MESSAGE}",
                    err=True,
                )
                raise SystemExit(1)

            pw_hash = generate_password_hash(password)

            try:
                execute_db(
                    """
                    INSERT INTO users
                        (username, password_hash, role_id, must_change_password)
                    VALUES (?, ?, ?, 1)
                    """,
                    (username, pw_hash, role_id),
                )
                click.echo(f"User '{username}' created with role '{role}'.")
            except Exception as exc:
                click.echo(f"Failed to create user: {exc}", err=True)
                raise SystemExit(1)

    @app.cli.command("user-list")
    def user_list() -> None:
        """List all application users."""
        from db.db import query_db

        with app.app_context():
            rows = query_db("""
                SELECT u.id, u.username, r.name AS role, u.active, u.created_at
                FROM users u
                JOIN roles r ON r.id = u.role_id
                ORDER BY u.id
                """)
            if not rows or not isinstance(rows, list):
                click.echo("No users found.")
                return
            click.echo(
                f"{'ID':<5} {'Username':<20} {'Role':<10} {'Active':<8} {'Created'}"
            )
            click.echo("-" * 65)
            for row in rows:
                assert isinstance(row, dict)
                active_label = "yes" if row["active"] else "no"
                click.echo(
                    f"{row['id']:<5} {row['username']:<20} {row['role']:<10} "
                    f"{active_label:<8} {row['created_at']}"
                )

    @app.cli.command("credentials-set")
    @click.option("--label", default="default", help="Credential label.")
    @click.option("--base-url", required=True, help="Backend API base URL.")
    @click.option("--username", default=None, help="API username (optional).")
    @click.option("--password", default=None, help="API password (optional).")
    @click.option("--timeout", default=30, type=int, help="Request timeout in seconds.")
    def credentials_set(
        label: str,
        base_url: str,
        username: str | None,
        password: str | None,
        timeout: int,
    ) -> None:
        """Create or update an API credential entry."""
        from db.db import execute_db

        with app.app_context():
            enc_user: str | None = None
            enc_pass: str | None = None

            if username or password:
                from db.crypto import encrypt

                if username:
                    enc_user = encrypt(username)
                if password:
                    enc_pass = encrypt(password)

            execute_db(
                """
                INSERT INTO api_credentials
                    (label, base_url, encrypted_username, encrypted_password, timeout)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(label) DO UPDATE SET
                    base_url           = excluded.base_url,
                    encrypted_username = excluded.encrypted_username,
                    encrypted_password = excluded.encrypted_password,
                    timeout            = excluded.timeout
                """,
                (label, base_url, enc_user, enc_pass, timeout),
            )
            click.echo(f"Credential '{label}' saved.")

    @app.cli.command("credentials-list")
    def credentials_list() -> None:
        """List stored API credentials (passwords redacted)."""
        from db.db import query_db

        with app.app_context():
            rows = query_db(
                "SELECT id, label, base_url, timeout, created_at FROM api_credentials"
            )
            if not rows or not isinstance(rows, list):
                click.echo("No credentials found.")
                return
            click.echo(f"{'ID':<5} {'Label':<15} {'Base URL':<40} {'Timeout'}")
            click.echo("-" * 70)
            for row in rows:
                assert isinstance(row, dict)
                click.echo(
                    f"{row['id']:<5} {row['label']:<15} {row['base_url']:<40} {row['timeout']}s"
                )

    @app.cli.command("credentials-delete")
    @click.option("--label", required=True, help="Label of the credential to delete.")
    def credentials_delete(label: str) -> None:
        """Delete an API credential entry by label."""
        from db.db import execute_db

        with app.app_context():
            execute_db("DELETE FROM api_credentials WHERE label = ?", (label,))
            click.echo(f"Credential '{label}' deleted.")

    @app.cli.command("generate-key")
    def generate_key() -> None:
        """Generate a new Fernet encryption key for FLASK_ENCRYPTION_KEY."""
        from cryptography.fernet import Fernet

        key = Fernet.generate_key().decode()
        click.echo(f"Generated key (set as FLASK_ENCRYPTION_KEY):\n{key}")
