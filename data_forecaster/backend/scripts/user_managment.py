"""Admin CLI for backend API user management.

Run from the backend scripts directory:

    python user_managment.py --add--user alice --description "Analyst API access"
    python user_managment.py --delete--user alice
    python user_managment.py --disable--user --id 3
    python user_managment.py --reset--apikey alice
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import sqlite3
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from auth.api_key_db import (  # noqa: E402
    create_api_user,
    delete_api_user,
    rotate_api_key,
    set_user_enabled,
)
import core.config as settings  # noqa: E402
from core.database import get_connection, init_database  # noqa: E402

Action = Callable[[argparse.Namespace], int]


def _resolve_db_path(db_path: str) -> str:
    """Resolve relative backend DB paths from the backend project root."""
    if db_path == ":memory:":
        return db_path
    path = Path(db_path)
    if path.is_absolute():
        return str(path)
    return str(BACKEND_ROOT / path)


def _resolve_user(
    username: str | None,
    user_id: int | None,
) -> tuple[int, str]:
    """Resolve exactly one CLI identifier to an API user id and username."""
    if (username is not None) == (user_id is not None):
        raise ValueError("Provide exactly one of username or --id.")

    conn: sqlite3.Connection = get_connection()
    try:
        if user_id is not None:
            row = conn.execute(
                "SELECT id, username FROM api_users WHERE id = ?",
                (user_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id, username FROM api_users WHERE username = ?",
                (username,),
            ).fetchone()
    finally:
        conn.close()

    if row is None:
        target = f"id={user_id}" if user_id is not None else f"username='{username}'"
        raise ValueError(f"API user with {target} not found.")

    return int(row["id"]), str(row["username"])


def _add(args: argparse.Namespace) -> int:
    api_key = create_api_user(
        username=args.username.strip(),
        description=(args.description or "").strip(),
        is_admin=bool(args.admin),
    )
    print(f"Created API user '{args.username.strip()}'.")
    print(f"API key: {api_key}")
    print("Store this key now; it cannot be recovered later.")
    return 0


def _delete(args: argparse.Namespace) -> int:
    user_id, username = _resolve_user(args.username, args.user_id)
    delete_api_user(user_id)
    print(f"Deleted API user '{username}' (id={user_id}).")
    return 0


def _disable(args: argparse.Namespace) -> int:
    user_id, username = _resolve_user(args.username, args.user_id)
    set_user_enabled(user_id, False)
    print(f"Disabled API user '{username}' (id={user_id}).")
    return 0


def _reset_api_key(args: argparse.Namespace) -> int:
    user_id, username = _resolve_user(args.username, args.user_id)
    api_key = rotate_api_key(user_id)
    print(f"Reset API key for '{username}' (id={user_id}).")
    print(f"API key: {api_key}")
    print("Store this key now; it cannot be recovered later.")
    return 0


def _add_identifier_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--id", dest="user_id", type=int, help="API user ID.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage backend API users.")
    parser.add_argument(
        "--db-path",
        default=settings.BACKEND_DB_PATH,
        help=f"SQLite database path. Defaults to {settings.BACKEND_DB_PATH}.",
    )
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--add--user",
        dest="add_user",
        metavar="USERNAME",
        help="Add an API user.",
    )
    action_group.add_argument(
        "--delete--user",
        dest="delete_user",
        nargs="?",
        const="",
        metavar="USERNAME",
        help="Delete an API user by username, or use with --id.",
    )
    action_group.add_argument(
        "--disable--user",
        dest="disable_user",
        nargs="?",
        const="",
        metavar="USERNAME",
        help="Disable an API user by username, or use with --id.",
    )
    action_group.add_argument(
        "--reset--apikey",
        dest="reset_apikey_user",
        nargs="?",
        const="",
        metavar="USERNAME",
        help="Reset an API user's key by username, or use with --id.",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Human-readable description for the API user.",
    )
    parser.add_argument(
        "--admin",
        action="store_true",
        help="Create the API user with admin privileges.",
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
    if args.reset_apikey_user is not None:
        return _reset_api_key, args.reset_apikey_user or None
    raise ValueError("No action selected.")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings.BACKEND_DB_PATH = _resolve_db_path(str(args.db_path))
    init_database()

    try:
        func, username = _select_action(args)
        args.username = username
        return int(func(args))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
