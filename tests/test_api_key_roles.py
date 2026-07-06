"""Tests for role-based access control on backend API key endpoints."""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest
from fastapi.testclient import TestClient

# Ensure backend modules are importable from the tests directory.
_backend_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data_forecaster", "backend"
)
_backend_dir = os.path.abspath(_backend_dir)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from auth.api_key_db import (  # noqa: E402
    create_api_user,
    create_first_user,
    delete_api_user,
    init_db,
    list_api_users,
    set_user_admin,
)
from auth.dependency import require_admin_api_key, require_api_key  # noqa: E402
from main import app  # noqa: E402

# Test API key reused from the ADMIN_API_KEY env var set in _reset_api_key_db.
_ADMIN_KEY = "test-admin-key"


@pytest.fixture(autouse=True)
def _reset_api_key_db(tmp_path: Any, monkeypatch: Any) -> None:
    """Use a fresh temporary SQLite database for every test."""
    db_dir = tmp_path / "api_keys"
    db_dir.mkdir()
    monkeypatch.setenv("API_KEY_DB_PATH", str(db_dir))
    monkeypatch.setenv("API_KEY_ENABLED", "true")
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("FILE_STORAGE_DIR", str(tmp_path / "files"))
    init_db()


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def admin_user() -> tuple[str, str]:
    """Create an admin API user and return (username, plaintext_key)."""
    create_first_user(username="admin", api_key=_ADMIN_KEY)
    return "admin", _ADMIN_KEY


@pytest.fixture
def regular_user(admin_user: tuple[str, str]) -> tuple[str, str]:
    """Create a regular API user and return (username, plaintext_key)."""
    plaintext = create_api_user(
        username="regular", description="Regular user", is_admin=False
    )
    return "regular", plaintext


def _auth_headers(username: str, api_key: str) -> dict[str, str]:
    """Return headers for API key authentication."""
    return {"X-API-Username": username, "X-API-Key": api_key}


class TestRequireAdminApiKey:
    """Unit tests for the admin API key dependency."""

    def test_missing_credentials_returns_401(self, client: TestClient) -> None:
        """A request with no credentials receives 401."""
        response = client.get("/api-users")
        assert response.status_code == 401

    def test_regular_user_returns_403(
        self, client: TestClient, regular_user: tuple[str, str]
    ) -> None:
        """A regular API user receives 403 on admin endpoints."""
        username, key = regular_user
        response = client.get("/api-users", headers=_auth_headers(username, key))
        assert response.status_code == 403

    def test_admin_user_succeeds(
        self, client: TestClient, admin_user: tuple[str, str]
    ) -> None:
        """An admin API user can list API users."""
        username, key = admin_user
        response = client.get("/api-users", headers=_auth_headers(username, key))
        assert response.status_code == 200


class TestAdminEndpoints:
    """Tests that admin endpoints reject regular users."""

    @pytest.mark.parametrize(
        "method,path,payload",
        [
            ("GET", "/api-users", None),
            ("POST", "/api-users", {"username": "new", "description": ""}),
            ("POST", "/api-users/1/rotate", None),
            ("POST", "/api-users/1/toggle", {"enabled": False}),
            ("POST", "/api-users/1/admin", {"is_admin": True}),
            ("DELETE", "/api-users/1", None),
            ("GET", "/api-users/bootstrap-status", None),
        ],
    )
    def test_regular_user_forbidden(
        self,
        client: TestClient,
        admin_user: tuple[str, str],
        regular_user: tuple[str, str],
        method: str,
        path: str,
        payload: dict[str, Any] | None,
    ) -> None:
        """Regular users are forbidden from all admin endpoints."""
        username, key = regular_user
        response = client.request(
            method,
            path,
            headers=_auth_headers(username, key),
            json=payload,
        )
        assert response.status_code == 403

    def test_admin_can_create_user(
        self, client: TestClient, admin_user: tuple[str, str]
    ) -> None:
        """An admin can create a new regular API user."""
        username, key = admin_user
        response = client.post(
            "/api-users",
            headers=_auth_headers(username, key),
            json={"username": "created", "description": "", "is_admin": False},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["user"]["username"] == "created"
        assert data["user"]["is_admin"] is False
        assert "api_key" in data

    def test_admin_can_promote_user(
        self, client: TestClient, admin_user: tuple[str, str]
    ) -> None:
        """An admin can promote a regular user to admin."""
        admin_name, admin_key = admin_user
        create_api_user(username="to-promote", description="", is_admin=False)
        response = client.post(
            "/api-users/2/admin",
            headers=_auth_headers(admin_name, admin_key),
            json={"is_admin": True},
        )
        assert response.status_code == 200
        assert response.json()["is_admin"] is True


class TestBootstrap:
    """Tests for the bootstrap endpoint."""

    def test_bootstrap_creates_admin_user(self, client: TestClient) -> None:
        """The bootstrap endpoint creates the first user as an admin."""
        response = client.post(
            "/api-users/bootstrap",
            headers={"X-Admin-Key": "test-admin-key"},
            json={"username": "bootstrap-admin", "api_key": "secret"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user"]["is_admin"] is True
        assert data["auth_enabled"] is True

    def test_bootstrap_requires_admin_key(self, client: TestClient) -> None:
        """The bootstrap endpoint requires the deployment admin key."""
        response = client.post(
            "/api-users/bootstrap",
            json={"username": "bootstrap-admin", "api_key": "secret"},
        )
        assert response.status_code == 403


class TestSetUserAdmin:
    """Tests for the set_user_admin helper."""

    def test_set_user_admin_updates_flag(self) -> None:
        """set_user_admin updates the is_admin flag."""
        create_first_user(username="admin", api_key="secret")
        _ = create_api_user(username="regular", description="", is_admin=False)
        # create_api_user returns the plaintext key; find the actual id.
        users = list_api_users()
        regular = next(u for u in users if u["username"] == "regular")
        set_user_admin(regular["id"], True)
        updated = next(u for u in list_api_users() if u["id"] == regular["id"])
        assert updated["is_admin"] is True

    def test_set_user_admin_missing_user_raises(self) -> None:
        """set_user_admin raises ValueError for a missing user id."""
        with pytest.raises(ValueError, match="API user with id 999 not found"):
            set_user_admin(999, True)


class TestCreateApiUser:
    """Tests for the create_api_user helper."""

    def test_create_api_user_defaults_to_regular(self) -> None:
        """create_api_user defaults to a non-admin user."""
        create_first_user(username="admin", api_key="secret")
        create_api_user(username="regular", description="")
        users = list_api_users()
        regular = next(u for u in users if u["username"] == "regular")
        assert regular["is_admin"] is False

    def test_create_api_user_can_be_admin(self) -> None:
        """create_api_user can create an admin user."""
        create_first_user(username="admin", api_key="secret")
        create_api_user(username="another-admin", description="", is_admin=True)
        users = list_api_users()
        admin = next(u for u in users if u["username"] == "another-admin")
        assert admin["is_admin"] is True
