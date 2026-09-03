"""Unit tests for the first-run setup service (atomic bootstrap)."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

_BACKEND = str(Path(__file__).resolve().parent.parent / "data_forecaster" / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core import secret_store  # noqa: E402
from core import config as settings  # noqa: E402
from core.database import get_connection, init_database  # noqa: E402
from services import setup_service  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Temp backend DB + secrets dir; patch the modules' own references."""
    secrets = tmp_path / "secrets"
    monkeypatch.setattr(secret_store.settings, "SECRETS_DIR", str(secrets))
    secret_store.reset_cache()

    db_path = str(tmp_path / "backend.db")
    monkeypatch.setattr(settings, "BACKEND_DB_PATH", db_path)
    init_database()
    yield db_path
    secret_store.reset_cache()


class TestBootstrap:
    """Atomic first-run bootstrap."""

    def test_first_bootstrap_succeeds(self, db):
        user = setup_service.run_bootstrap("admin", "secret-key", db_path=db)

        assert user["username"] == "admin"
        assert user["is_admin"] == 1
        assert "api_key_hash" not in user
        assert setup_service.is_setup_complete(db) is True
        assert secret_store.key_file_exists()

    def test_second_bootstrap_rejected(self, db):
        setup_service.run_bootstrap("admin", "secret-key", db_path=db)

        with pytest.raises(setup_service.SetupAlreadyCompleteError):
            setup_service.run_bootstrap("other", "other-key", db_path=db)

    def test_concurrent_bootstrap_exactly_one_wins(self, db):
        results: list[str] = []
        errors: list[Exception] = []

        def attempt(name: str) -> None:
            try:
                setup_service.run_bootstrap(name, f"key-{name}", db_path=db)
                results.append(name)
            except setup_service.SetupAlreadyCompleteError as exc:
                errors.append(exc)

        threads = [threading.Thread(target=attempt, args=(f"user{i}",)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(results) == 1
        assert len(errors) == 3
        with get_connection(db) as connection:
            count = connection.execute(
                "SELECT COUNT(*) AS cnt FROM api_users"
            ).fetchone()["cnt"]
        assert count == 1

    def test_empty_username_rejected(self, db):
        with pytest.raises(ValueError, match="Username is required"):
            setup_service.run_bootstrap("  ", "key", db_path=db)


class TestSetupStatus:
    """Setup status reporting (no secrets)."""

    def test_fresh_install_status(self, db):
        status = setup_service.get_setup_status(db)

        assert status == {
            "setup_complete": False,
            "admin_exists": False,
            "llm_configured": False,
            "models_enabled": 5,
        }

    def test_status_after_bootstrap(self, db):
        setup_service.run_bootstrap("admin", "secret-key", db_path=db)

        status = setup_service.get_setup_status(db)

        assert status["setup_complete"] is True
        assert status["admin_exists"] is True


class TestLegacyMigration:
    """Existing deployments migrate to setup_complete automatically."""

    def test_marks_complete_when_users_exist(self, db):
        setup_service.run_bootstrap("admin", "secret-key", db_path=db)
        # Simulate a pre-wizard deployment: users exist, flag cleared.
        with get_connection(db) as connection:
            connection.execute(
                "UPDATE system_settings SET setup_complete = 0 WHERE singleton = 1"
            )
            connection.commit()

        setup_service.migrate_legacy_deployment(db)

        assert setup_service.is_setup_complete(db) is True

    def test_noop_on_fresh_install(self, db):
        setup_service.migrate_legacy_deployment(db)

        assert setup_service.is_setup_complete(db) is False
