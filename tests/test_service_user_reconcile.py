"""Tests for reconciling the frontend service API user."""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

_backend_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data_forecaster", "backend")
)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import core.config as settings  # noqa: E402
from auth.api_key_db import (  # noqa: E402
    create_first_user,
    reconcile_service_user,
    set_user_enabled,
    verify_api_key,
)
from core.database import init_database  # noqa: E402


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
