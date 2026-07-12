"""Tests for frontend database initialization behavior."""

from __future__ import annotations

from pathlib import Path
import sys

from cryptography.fernet import Fernet
from flask import Flask
from werkzeug.security import check_password_hash

FRONTEND_ROOT = Path(__file__).resolve().parents[1] / "data_forecaster" / "frontend"
if str(FRONTEND_ROOT) in sys.path:
    sys.path.remove(str(FRONTEND_ROOT))
sys.path.insert(0, str(FRONTEND_ROOT))
sys.modules.pop("services", None)

from app import _sync_app_config_from_db
from db.crypto import decrypt, encrypt
from db.db import get_db, init_app, init_db


def _app(tmp_path: Path) -> Flask:
    app = Flask(__name__)
    app.config.update(
        DATABASE=str(tmp_path / "frontend.db"),
    )
    init_app(app)
    return app


def test_init_db_seeds_forced_reset_admin_and_blank_api_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """First boot uses DB-owned admin/API config setup."""
    monkeypatch.setenv("FLASK_ENCRYPTION_KEY", Fernet.generate_key().decode())
    app = _app(tmp_path)

    with app.app_context():
        init_db()
        db = get_db()
        user = db.execute(
            """
            SELECT username, password_hash, must_change_password
            FROM users
            WHERE username = 'admin'
            """
        ).fetchone()
        api_config = db.execute(
            """
            SELECT base_url, timeout, verify_ssl, encrypted_username,
                   encrypted_password
            FROM api_credentials
            WHERE label = 'default'
            """
        ).fetchone()
        upload_config = db.execute(
            "SELECT value FROM app_config WHERE key = 'max_upload_mb'"
        ).fetchone()

    assert user is not None
    assert check_password_hash(user["password_hash"], "admin")
    assert int(user["must_change_password"]) == 1
    assert api_config["base_url"] == ""
    assert api_config["timeout"] == 30
    assert int(api_config["verify_ssl"]) == 0
    assert api_config["encrypted_username"] is None
    assert api_config["encrypted_password"] is None
    assert upload_config["value"] == "100"


def test_sync_app_config_applies_upload_limit(tmp_path: Path, monkeypatch) -> None:
    """The DB-owned upload limit should drive Flask MAX_CONTENT_LENGTH."""
    monkeypatch.setenv("FLASK_ENCRYPTION_KEY", Fernet.generate_key().decode())
    app = _app(tmp_path)

    with app.app_context():
        init_db()
        db = get_db()
        db.execute(
            """
            UPDATE app_config
            SET value = '42'
            WHERE key = 'max_upload_mb'
            """
        )
        db.commit()

        _sync_app_config_from_db(app)

    assert app.config["MAX_CONTENT_LENGTH"] == 42 * 1024 * 1024


def test_init_db_preserves_existing_api_credentials(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Repeated initialization should not overwrite API Config."""
    monkeypatch.setenv("FLASK_ENCRYPTION_KEY", Fernet.generate_key().decode())
    app = _app(tmp_path)

    with app.app_context():
        init_db()
        db = get_db()
        db.execute(
            """
            UPDATE api_credentials
            SET encrypted_username = ?,
                encrypted_password = ?
            WHERE label = 'default'
            """,
            (encrypt("admin-user"), encrypt("admin-key")),
        )
        db.commit()

        init_db()

        row = db.execute(
            """
            SELECT encrypted_username, encrypted_password
            FROM api_credentials
            WHERE label = 'default'
            """
        ).fetchone()

    assert decrypt(row["encrypted_username"]) == "admin-user"
    assert decrypt(row["encrypted_password"]) == "admin-key"
