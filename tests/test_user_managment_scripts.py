"""Tests for backend and frontend user management scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sqlite3
from typing import Any

import pytest
from werkzeug.security import check_password_hash

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "data_forecaster" / "backend"
FRONTEND_ROOT = REPO_ROOT / "data_forecaster" / "frontend"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_backend_script() -> Any:
    return _load_module(
        "backend_user_managment_script",
        BACKEND_ROOT / "scripts" / "user_managment.py",
    )


def _load_frontend_script() -> Any:
    return _load_module(
        "frontend_user_managment_script",
        FRONTEND_ROOT / "scripts" / "user_managment.py",
    )


def _frontend_db(path: Path) -> None:
    schema = FRONTEND_ROOT / "db" / "schema.sql"
    conn = sqlite3.connect(path)
    try:
        conn.executescript(schema.read_text(encoding="utf-8"))
        conn.execute("INSERT INTO roles (id, name) VALUES (1, 'admin')")
        conn.execute("INSERT INTO roles (id, name) VALUES (2, 'user')")
        conn.commit()
    finally:
        conn.close()


def test_backend_script_add_disable_reset_and_delete(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """Backend script manages API users by username and id."""
    db_path = tmp_path / "backend.db"
    monkeypatch.setenv("BACKEND_DB_PATH", str(db_path))
    module = _load_backend_script()

    assert (
        module.main(["--db-path", str(db_path), "--add--user", "alice", "--admin"]) == 0
    )
    output = capsys.readouterr().out
    assert "Created API user 'alice'." in output
    assert "API key:" in output

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT id, enabled, is_admin FROM api_users WHERE username = 'alice'"
        ).fetchone()
        assert row is not None
        assert int(row["enabled"]) == 1
        assert int(row["is_admin"]) == 1
        user_id = int(row["id"])
    finally:
        conn.close()

    assert (
        module.main(
            ["--db-path", str(db_path), "--disable--user", "--id", str(user_id)]
        )
        == 0
    )
    conn = sqlite3.connect(db_path)
    try:
        disabled = conn.execute(
            "SELECT enabled FROM api_users WHERE id = ?",
            (user_id,),
        ).fetchone()
        assert disabled is not None
        assert int(disabled[0]) == 0
    finally:
        conn.close()

    assert module.main(["--db-path", str(db_path), "--reset--apikey", "alice"]) == 0
    assert "Reset API key for 'alice'" in capsys.readouterr().out

    assert module.main(["--db-path", str(db_path), "--delete--user", "alice"]) == 0
    conn = sqlite3.connect(db_path)
    try:
        deleted = conn.execute(
            "SELECT id FROM api_users WHERE id = ?",
            (user_id,),
        ).fetchone()
        assert deleted is None
    finally:
        conn.close()


def test_backend_script_rejects_ambiguous_identifier(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """Backend script requires exactly one user identifier."""
    db_path = tmp_path / "backend.db"
    monkeypatch.setenv("BACKEND_DB_PATH", str(db_path))
    module = _load_backend_script()

    assert (
        module.main(
            ["--db-path", str(db_path), "--disable--user", "alice", "--id", "1"]
        )
        == 1
    )
    assert "Provide exactly one" in capsys.readouterr().err


def test_backend_script_resolves_relative_db_path_from_backend_root(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Default-style relative DB paths should not depend on cwd."""
    module = _load_backend_script()
    monkeypatch.chdir(BACKEND_ROOT / "scripts")
    monkeypatch.setattr(module, "BACKEND_ROOT", tmp_path)

    assert module.main(["--db-path", "data/backend.db", "--add--user", "svc"]) == 0

    assert (tmp_path / "data" / "backend.db").exists()
    assert not (BACKEND_ROOT / "scripts" / "data" / "backend.db").exists()


def test_frontend_script_add_disable_reset_and_delete(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """Frontend script manages app users and session invalidation fields."""
    db_path = tmp_path / "frontend.db"
    _frontend_db(db_path)
    module = _load_frontend_script()

    assert (
        module.main(
            [
                "--db-path",
                str(db_path),
                "--add--user",
                "alice",
                "--admin",
                "--generate-temp-password",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Created frontend user 'alice' (admin)." in output
    assert "Temporary password:" in output
    generated_password = output.split("Temporary password: ", 1)[1].splitlines()[0]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("""
            SELECT u.id, u.password_hash, u.active, u.must_change_password,
                   u.session_version, r.name AS role
            FROM users u
            JOIN roles r ON r.id = u.role_id
            WHERE u.username = 'alice'
            """).fetchone()
        assert row is not None
        assert row["role"] == "admin"
        assert int(row["active"]) == 1
        assert int(row["must_change_password"]) == 1
        assert int(row["session_version"]) == 0
        assert check_password_hash(str(row["password_hash"]), generated_password)
        user_id = int(row["id"])
    finally:
        conn.close()

    assert (
        module.main(
            ["--db-path", str(db_path), "--disable--user", "--id", str(user_id)]
        )
        == 0
    )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT active, session_version FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        assert row is not None
        assert int(row["active"]) == 0
        assert int(row["session_version"]) == 1
    finally:
        conn.close()

    monkeypatch.setenv("FRONTEND_USER_PASSWORD", "Password2!")
    reset_args = [
        "--db-path",
        str(db_path),
        "--reset--password",
        "alice",
        "--no-must-change-password",
    ]
    assert module.main(reset_args) == 0
    assert "Temporary password:" not in capsys.readouterr().out
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT password_hash, must_change_password, session_version
            FROM users WHERE id = ?
            """,
            (user_id,),
        ).fetchone()
        assert row is not None
        assert check_password_hash(str(row["password_hash"]), "Password2!")
        assert int(row["must_change_password"]) == 0
        assert int(row["session_version"]) == 2
    finally:
        conn.close()

    with pytest.raises(SystemExit):
        module.main(["--db-path", str(db_path), "--add--user", "bob", "--password"])

    assert (
        module.main(["--db-path", str(db_path), "--delete--user", "alice", "--yes"])
        == 0
    )
    conn = sqlite3.connect(db_path)
    try:
        deleted = conn.execute(
            "SELECT id FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        assert deleted is None
    finally:
        conn.close()


def test_frontend_script_requires_delete_confirmation(
    tmp_path: Path,
    capsys: Any,
) -> None:
    """Frontend deletion requires explicit confirmation."""
    db_path = tmp_path / "frontend.db"
    _frontend_db(db_path)
    module = _load_frontend_script()
    assert (
        module.main(
            [
                "--db-path",
                str(db_path),
                "--add--user",
                "alice",
                "--generate-temp-password",
            ]
        )
        == 0
    )

    assert module.main(["--db-path", str(db_path), "--delete--user", "alice"]) == 1
    assert "Re-run with --yes" in capsys.readouterr().err
