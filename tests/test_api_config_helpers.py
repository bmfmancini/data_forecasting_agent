"""Tests for API Config credential handling."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from cryptography.fernet import Fernet
from flask import Flask

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "data_forecaster" / "backend"
FRONTEND_ROOT = REPO_ROOT / "data_forecaster" / "frontend"
if str(FRONTEND_ROOT) in sys.path:
    sys.path.remove(str(FRONTEND_ROOT))
sys.path.insert(0, str(FRONTEND_ROOT))
sys.modules.pop("services", None)

from blueprints.admin import routes
from db.crypto import decrypt, encrypt
from db.db import get_db, init_app, init_db

sys.modules.pop("services", None)
if str(FRONTEND_ROOT) in sys.path:
    sys.path.remove(str(FRONTEND_ROOT))
if str(BACKEND_ROOT) in sys.path:
    sys.path.remove(str(BACKEND_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(1, str(FRONTEND_ROOT))


def _app(tmp_path: Path) -> Flask:
    app = Flask(__name__)
    app.config.update(
        DATABASE=str(tmp_path / "frontend.db"),
    )
    init_app(app)
    return app


def test_load_current_api_config_never_decrypts_key(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The summary may decrypt the username but must not decrypt the API key."""
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
            ("enc-user", "enc-key"),
        )
        db.commit()

        def fake_decrypt(ciphertext: str) -> str:
            assert ciphertext == "enc-user"
            return "frontend"

        monkeypatch.setattr(routes, "decrypt", fake_decrypt)

        config = routes._load_current_api_config()

    assert config is not None
    assert config["username"] == "frontend"
    assert config["has_key"] is True


def test_save_api_credentials_preserves_existing_key(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Changing username with a blank key should keep the encrypted key."""
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
            (encrypt("old-user"), encrypt("old-key")),
        )
        db.commit()

        routes._save_api_credentials(
            base_url="https://new-backend",
            timeout=45,
            verify_ssl=1,
            enc_user=encrypt("new-user"),
            enc_pass=None,
            preserve_existing_key=True,
        )

        row = db.execute("""
            SELECT base_url, timeout, verify_ssl, encrypted_username,
                   encrypted_password
            FROM api_credentials
            WHERE label = 'default'
            """).fetchone()

    assert row["base_url"] == "https://new-backend"
    assert row["timeout"] == 45
    assert row["verify_ssl"] == 1
    assert decrypt(row["encrypted_username"]) == "new-user"
    assert decrypt(row["encrypted_password"]) == "old-key"


def test_client_from_api_config_form_uses_stored_key_server_side(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Testing edited config can preserve the blank key field."""
    monkeypatch.setenv("FLASK_ENCRYPTION_KEY", Fernet.generate_key().decode())
    app = _app(tmp_path)

    with app.app_context():
        init_db()
        db = get_db()
        db.execute(
            """
            UPDATE api_credentials
            SET encrypted_password = ?
            WHERE label = 'default'
            """,
            (encrypt("stored-key"),),
        )
        db.commit()

    with app.test_request_context(
        "/admin/api-config/test",
        method="POST",
        data={
            "base_url": "https://edited-backend",
            "api_username": "edited-user",
            "api_password": "",
            "verify_ssl": "on",
        },
    ):
        client = routes._client_from_api_config_form()

    assert client is not None
    assert client._base_url == "https://edited-backend"
    assert client._api_username == "edited-user"
    assert client._api_key == "stored-key"
    assert client._verify is True
