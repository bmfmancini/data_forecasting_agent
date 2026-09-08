"""Unit tests for the backend secret store (Fernet key lifecycle)."""

from __future__ import annotations

import os
import stat

import pytest

from data_forecaster.backend.core import secret_store


@pytest.fixture
def secrets_dir(tmp_path, monkeypatch):
    """Point SECRETS_DIR at a temp dir and reset the cached Fernet.

    Note: ``secret_store`` imports its settings as ``core.config`` (backend
    import style), which is a distinct module object from
    ``data_forecaster.backend.core.config``.  Patch the object the module
    under test actually references.
    """
    directory = tmp_path / "secrets"
    monkeypatch.setattr(secret_store.settings, "SECRETS_DIR", str(directory))
    secret_store.reset_cache()
    yield directory
    secret_store.reset_cache()


class TestKeyLifecycle:
    """Key generation, persistence, and loading."""

    def test_generate_persists_key_with_0600(self, secrets_dir):
        secret_store.generate_and_persist_key()

        key_path = secrets_dir / ".encryption_key"
        assert key_path.is_file()
        mode = stat.S_IMODE(os.stat(key_path).st_mode)
        assert mode == 0o600

    def test_generate_is_idempotent(self, secrets_dir):
        secret_store.generate_and_persist_key()
        key_path = secrets_dir / ".encryption_key"
        first = key_path.read_bytes()

        secret_store.generate_and_persist_key()

        assert key_path.read_bytes() == first

    def test_get_fernet_missing_key_raises(self, secrets_dir):
        with pytest.raises(RuntimeError, match="encryption key not found"):
            secret_store.get_fernet()

    def test_round_trip_encrypt_decrypt(self, secrets_dir):
        secret_store.generate_and_persist_key()

        ciphertext = secret_store.encrypt("super-secret-api-key")

        assert ciphertext != "super-secret-api-key"
        assert secret_store.decrypt(ciphertext) == "super-secret-api-key"

    def test_key_survives_cache_reset(self, secrets_dir):
        secret_store.generate_and_persist_key()
        ciphertext = secret_store.encrypt("value")

        secret_store.reset_cache()

        assert secret_store.decrypt(ciphertext) == "value"
