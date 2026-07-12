"""Admin CLI for frontend user management.

Run from the frontend scripts directory:

    python user_managment.py --add--user alice
    python user_managment.py --delete--user alice --yes
    python user_managment.py --disable--user --id 3
    python user_managment.py --reset--password alice
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import getpass
import os
from pathlib import Path
import sqlite3
import sys

from werkzeug.security import generate_password_hash

FRONTEND_ROOT = Path(__file__).resolve().parents[1]
if str(FRONTEND_ROOT) not in sys.path:
    sys.path.insert(0, str(FRONTEND_ROOT))

from config import BaseConfig  # noqa: E402

Action = Callable[[argparse.Namespace], int]


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _ensure_database(db_path: str) -> None:
    if db_path == ":memory:":
        return
    if not os.path.exists(db_path):
        raise ValueError(f"database not found at {db_path}")


def _resolve_user(
    conn: sqlite3.Connection,
    username: str | None,
    user_id: int | None,
) -> tuple[int, str]:
    """Resolve exactly one CLI identifier to a frontend user id and username."""
    if (username is not None) == (user_id is not None):
        raise ValueError("Provide exactly one of username or --id.")

    if user_id is not None:
        row = conn.execute(
            "SELECT id, username FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id, username FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if row is None:
        target = f"id={user_id}" if user_id is not None else f"username='{username}'"
        raise ValueError(f"frontend user with {target} not found.")

    return int(row["id"]), str(row["username"])


def _role_id(conn: sqlite3.Connection, role_name: str) -> int:
    row = conn.execute("SELECT id FROM roles WHERE name = ?", (role_name,)).fetchone()
    if row is None:
        raise ValueError(f"role '{role_name}' not found; initialize the frontend DB.")
    return int(row["id"])


def _read_password(args: argparse.Namespace) -> str:
    password = args.password or ""
    if password:
        return password
    if not sys.stdin.isatty():
        raise ValueError("No password provided. Use --password or run interactively.")
    first = getpass.getpass("Password: ")
    second = getpass.getpass("Confirm password: ")
    if first != second:
        raise ValueError("Passwords do not match.")
    if not first:
        raise ValueError("Password is required.")
    return first


def _add(args: argparse.Namespace) -> int:
    password = _read_password(args)
    role_name = "admin" if args.admin else "user"
    with _connect(args.db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?",
            (args.username.strip(),),
        ).fetchone()
        if existing is not None:
            raise ValueError(f"Username '{args.username.strip()}' already exists.")

        conn.execute(
            """
            INSERT INTO users
                (username, password_hash, role_id, active, must_change_password)
            VALUES (?, ?, ?, 1, ?)
            """,
            (
                args.username.strip(),
                generate_password_hash(password),
                _role_id(conn, role_name),
                int(args.must_change_password),
            ),
        )
        conn.commit()
    print(f"Created frontend user '{args.username.strip()}' ({role_name}).")
    return 0


def _delete(args: argparse.Namespace) -> int:
    with _connect(args.db_path) as conn:
        user_id, username = _resolve_user(conn, args.username, args.user_id)
        if not args.yes:
            raise ValueError(
                "Deleting a user also deletes their reports. Re-run with --yes."
            )
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    print(f"Deleted frontend user '{username}' (id={user_id}).")
    return 0


def _disable(args: argparse.Namespace) -> int:
    with _connect(args.db_path) as conn:
        user_id, username = _resolve_user(conn, args.username, args.user_id)
        conn.execute(
            """
            UPDATE users
            SET active = 0,
                session_version = session_version + 1
            WHERE id = ?
            """,
            (user_id,),
        )
        conn.commit()
    print(f"Disabled frontend user '{username}' (id={user_id}).")
    return 0


def _reset_password(args: argparse.Namespace) -> int:
    password = _read_password(args)
    with _connect(args.db_path) as conn:
        user_id, username = _resolve_user(conn, args.username, args.user_id)
        conn.execute(
            """
            UPDATE users
            SET password_hash = ?,
                must_change_password = ?,
                session_version = session_version + 1
            WHERE id = ?
            """,
            (
                generate_password_hash(password),
                int(args.must_change_password),
                user_id,
            ),
        )
        conn.commit()
    print(f"Reset password for frontend user '{username}' (id={user_id}).")
    return 0


def _add_identifier_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--id", dest="user_id", type=int, help="Frontend user ID.")


def _add_password_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--password",
        default=None,
        help="Plaintext password. If omitted, prompted interactively.",
    )
    parser.add_argument(
        "--must-change-password",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require password change on next login.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage frontend users.")
    parser.add_argument(
        "--db-path",
        default=BaseConfig.DATABASE,
        help=f"SQLite database path. Defaults to {BaseConfig.DATABASE}.",
    )
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--add--user",
        dest="add_user",
        metavar="USERNAME",
        help="Add a frontend user.",
    )
    action_group.add_argument(
        "--delete--user",
        dest="delete_user",
        nargs="?",
        const="",
        metavar="USERNAME",
        help="Delete a frontend user by username, or use with --id.",
    )
    action_group.add_argument(
        "--disable--user",
        dest="disable_user",
        nargs="?",
        const="",
        metavar="USERNAME",
        help="Disable a frontend user by username, or use with --id.",
    )
    action_group.add_argument(
        "--reset--password",
        dest="reset_password_user",
        nargs="?",
        const="",
        metavar="USERNAME",
        help="Reset a frontend user's password by username, or use with --id.",
    )
    parser.add_argument(
        "--admin",
        action="store_true",
        help="Create the frontend user with admin privileges.",
    )
    _add_password_args(parser)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion and cascading report deletion.",
    )
    _add_identifier_args(parser)

    return parser


def _select_action(args: argparse.Namespace) -> tuple[Action, str | None]:
    if args.add_user is not None:
        if args.user_id is not None:
            raise ValueError("--add--user does not support --id.")
        return _add, args.add_user
    if args.delete_user is not None:
        return _delete, args.delete_user or None
    if args.disable_user is not None:
        return _disable, args.disable_user or None
    if args.reset_password_user is not None:
        return _reset_password, args.reset_password_user or None
    raise ValueError("No action selected.")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _ensure_database(args.db_path)
        func, username = _select_action(args)
        args.username = username
        return int(func(args))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
