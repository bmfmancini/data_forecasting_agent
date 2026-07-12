"""Tests for reconciling the frontend service API user."""

from __future__ import annotations

from typing import Any

import pytest

import core.config as settings
from auth.api_key_db import (
    create_first_user,
    reconcile_service_user,
    set_user_enabled,
    verify_api_key,
)
from core.database import init_database


@pytest.fixture(autouse=True)
def _backend_db(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use an isolated backend database for each reconciliation test."""
    db_path = str(tmp_path / "backend.db")
    monkeypatch.setattr(settings, "BACKEND_DB_PATH", db_path)
    init_database()


def test_reconcile_service_user_updates_stale_key() -> None:
    """Existing service account hashes are updated from env key material."""
    create_first_user("frontend", "old-key")

    assert verify_api_key("frontend", "old-key") is not None
    assert verify_api_key("frontend", "new-key") is None

    assert reconcile_service_user("frontend", "new-key") is True

    assert verify_api_key("frontend", "old-key") is None
    assert verify_api_key("frontend", "new-key") is not None


def test_reconcile_service_user_reenables_disabled_user() -> None:
    """The configured service account is re-enabled during reconciliation."""
    user = create_first_user("frontend", "service-key")
    set_user_enabled(int(user["id"]), False)

    assert verify_api_key("frontend", "service-key") is None

    assert reconcile_service_user("frontend", "service-key") is True

    assert verify_api_key("frontend", "service-key") is not None


def test_reconcile_service_user_returns_false_for_missing_user() -> None:
    """Startup reconciliation does not create accounts once users exist."""
    assert reconcile_service_user("frontend", "service-key") is False
