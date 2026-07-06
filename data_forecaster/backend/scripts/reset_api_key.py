"""Admin CLI script to reset a backend API user's key.

Run inside the backend container (or any environment with access to the
backend's data directory) when an API key is lost or the stored Argon2
hash no longer matches the configured plaintext key.

Examples:
    Reset the pre-shared ``frontend`` service account to the value in
    ``FRONTEND_API_KEY``:

        python -m backend.scripts.reset_api_key frontend

    Reset to an explicit key:

        python -m backend.scripts.reset_api_key frontend --key new-secret-key

    Reset by user ID instead of username:

        python -m backend.scripts.reset_api_key --id 1 --key new-secret-key
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure local modules are importable when run as ``python -m`` or directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth.api_key_db import _get_connection, hash_api_key
from core.config import API_KEY_DB_PATH


def _resolve_user(
    conn: any,
    username: str | None,
    user_id: int | None,
) -> tuple[int, str]:
    """Resolve a username or user ID into the primary key and username.

    Args:
        conn:    SQLite connection object.
        username: API username to look up (optional).
        user_id:  API user ID to look up (optional).

    Returns:
        Tuple of ``(user_id, username)``.

    Raises:
        SystemExit: When no user matches or both/neither identifiers are given.
    """
    if bool(username) == bool(user_id):
        print(
            "Error: provide exactly one of --username USERNAME or --id ID.",
            file=sys.stderr,
        )
        sys.exit(1)

    if user_id is not None:
        row = conn.execute(
            "SELECT id, username FROM api_users WHERE id = ?", (user_id,)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id, username FROM api_users WHERE username = ?", (username,)
        ).fetchone()

    if row is None:
        target = f"id={user_id}" if user_id is not None else f"username='{username}'"
        print(f"Error: API user with {target} not found.", file=sys.stderr)
        sys.exit(1)

    return int(row["id"]), str(row["username"])


def main() -> int:
    """Parse CLI arguments and reset the selected API user's key.

    Returns:
        ``0`` on success, ``1`` on error.
    """
    parser = argparse.ArgumentParser(
        description="Reset a backend API user's key hash.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m backend.scripts.reset_api_key frontend
  python -m backend.scripts.reset_api_key frontend --key my-new-key
  python -m backend.scripts.reset_api_key --id 1 --key my-new-key
""",
    )
    parser.add_argument(
        "username",
        nargs="?",
        help="Username of the API user to reset.",
    )
    parser.add_argument(
        "--id",
        dest="user_id",
        type=int,
        help="API user ID to reset (alternative to username).",
    )
    parser.add_argument(
        "--key",
        dest="api_key",
        default=os.environ.get("FRONTEND_API_KEY", ""),
        help="New plaintext API key. Defaults to FRONTEND_API_KEY env var.",
    )
    parser.add_argument(
        "--db-path",
        dest="db_path",
        default=API_KEY_DB_PATH,
        help=f"Directory containing api_keys.db. Defaults to {API_KEY_DB_PATH}.",
    )

    args = parser.parse_args()

    if not args.api_key:
        print(
            "Error: no API key provided. Set --key or FRONTEND_API_KEY.",
            file=sys.stderr,
        )
        return 1

    db_file = os.path.join(args.db_path, "api_keys.db")
    if not os.path.exists(db_file):
        print(f"Error: database not found at {db_file}", file=sys.stderr)
        return 1

    conn = _get_connection()
    try:
        user_id, username = _resolve_user(conn, args.username, args.user_id)
        key_hash = hash_api_key(args.api_key)
        conn.execute(
            "UPDATE api_users SET api_key_hash = ? WHERE id = ?",
            (key_hash, user_id),
        )
        conn.commit()
    finally:
        conn.close()

    print(f"API key reset for user '{username}' (id={user_id}).")
    print(
        "Update the frontend's stored credentials (admin panel or "
        "FRONTEND_API_KEY env var) to match the new key."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
