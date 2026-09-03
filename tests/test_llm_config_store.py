"""Unit tests for the DB-backed LLM configuration store."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = str(Path(__file__).resolve().parent.parent / "data_forecaster" / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core import llm_config_store, secret_store  # noqa: E402
from core import config as settings  # noqa: E402
from core.database import init_database  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Temp backend DB + secrets dir; patch the module's own references."""
    secrets = tmp_path / "secrets"
    monkeypatch.setattr(secret_store.settings, "SECRETS_DIR", str(secrets))
    secret_store.reset_cache()
    secret_store.generate_and_persist_key()
    llm_config_store.reset_cache()

    db_path = str(tmp_path / "backend.db")
    monkeypatch.setattr(settings, "BACKEND_DB_PATH", db_path)
    init_database()
    yield db_path
    llm_config_store.reset_cache()


class TestEnvFallback:
    """Pre-setup behaviour when no llm_config row exists."""

    def test_gemini_fallback(self, db, monkeypatch):
        monkeypatch.setattr(settings, "USE_OLLAMA", False)
        monkeypatch.setattr(settings, "GOOGLE_API_KEY", "env-key")
        monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-test")

        config = llm_config_store.get_llm_config(db)

        assert config.provider == "gemini"
        assert config.model == "gemini-test"
        assert config.api_key == "env-key"
        assert config.version == 0

    def test_ollama_fallback(self, db, monkeypatch):
        monkeypatch.setattr(settings, "USE_OLLAMA", True)
        monkeypatch.setattr(settings, "USE_OLLAMA_CLOUD", False)
        monkeypatch.setattr(settings, "OLLAMA_MODEL", "llama3")
        monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://ollama:11434")

        config = llm_config_store.get_llm_config(db)

        assert config.provider == "ollama"
        assert config.base_url == "http://ollama:11434"


class TestDbBackedConfig:
    """DB row is authoritative once present."""

    def test_put_then_get_round_trip(self, db):
        llm_config_store.put_llm_config(
            provider="ollama_cloud",
            model="gpt-oss",
            base_url="https://ollama.com",
            api_key="secret-key-123",
            temperature=0.2,
            db_path=db,
        )

        config = llm_config_store.get_llm_config(db)

        assert config.provider == "ollama_cloud"
        assert config.api_key == "secret-key-123"
        assert config.temperature == pytest.approx(0.2)
        assert config.version == 1

    def test_db_overrides_env(self, db, monkeypatch):
        monkeypatch.setattr(settings, "USE_OLLAMA", False)
        monkeypatch.setattr(settings, "GOOGLE_API_KEY", "env-key")
        llm_config_store.put_llm_config(
            provider="ollama",
            model="llama3",
            base_url="http://ollama:11434",
            api_key=None,
            temperature=0.1,
            db_path=db,
        )

        config = llm_config_store.get_llm_config(db)

        assert config.provider == "ollama"

    def test_write_invalidates_cache_without_restart(self, db):
        llm_config_store.put_llm_config(
            "gemini", "gemini-1.5-flash", None, "key-a", 0.1, db_path=db
        )
        assert llm_config_store.get_llm_config(db).api_key == "key-a"

        llm_config_store.put_llm_config(
            "gemini", "gemini-2.0", None, "key-b", 0.1, db_path=db
        )

        config = llm_config_store.get_llm_config(db)
        assert config.model == "gemini-2.0"
        assert config.api_key == "key-b"
        assert config.version == 2

    def test_key_is_encrypted_at_rest(self, db):
        import sqlite3

        llm_config_store.put_llm_config(
            "gemini", "gemini-1.5-flash", None, "plaintext-key", 0.1, db_path=db
        )

        with sqlite3.connect(db) as connection:
            row = connection.execute(
                "SELECT encrypted_api_key FROM llm_config WHERE singleton = 1"
            ).fetchone()
        assert row[0] != "plaintext-key"
        assert "plaintext-key" not in row[0]

    def test_put_without_key_preserves_existing(self, db):
        llm_config_store.put_llm_config(
            "gemini", "gemini-1.5-flash", None, "keep-me", 0.1, db_path=db
        )
        llm_config_store.put_llm_config(
            "gemini", "gemini-2.0", None, None, 0.3, db_path=db
        )

        config = llm_config_store.get_llm_config(db)
        assert config.model == "gemini-2.0"
        assert config.api_key == "keep-me"

    def test_unknown_provider_rejected(self, db):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            llm_config_store.put_llm_config(
                "bogus", "m", None, None, 0.1, db_path=db
            )

    def test_is_configured(self, db):
        assert llm_config_store.is_configured(db) is False
        llm_config_store.put_llm_config(
            "gemini", "gemini-1.5-flash", None, "k", 0.1, db_path=db
        )
        assert llm_config_store.is_configured(db) is True
