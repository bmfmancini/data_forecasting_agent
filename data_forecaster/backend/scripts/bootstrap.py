"""Interactive first-run bootstrap for Data Forecaster deployments.

This script prepares the frontend and backend environment files without
requiring project dependencies on the host machine.  It generates stable
runtime secrets, synchronises the pre-shared frontend/backend service
credentials, and can optionally collect LLM provider credentials.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import os
import secrets
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Callable

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]
RUNNING_IN_BACKEND_CONTAINER = BACKEND_ROOT == Path("/app")
sys.path.insert(0, str(BACKEND_ROOT))

if RUNNING_IN_BACKEND_CONTAINER:
    BACKEND_ENV = BACKEND_ROOT / ".env"
    BACKEND_ENV_EXAMPLE = BACKEND_ROOT / ".env.example"
    FRONTEND_ENV = None
    FRONTEND_ENV_EXAMPLE = None
    COMPOSE_FILE = None
else:
    BACKEND_ENV = BACKEND_ROOT / ".env"
    BACKEND_ENV_EXAMPLE = BACKEND_ROOT / ".env.example"
    FRONTEND_ENV = PROJECT_ROOT / "frontend" / ".env"
    FRONTEND_ENV_EXAMPLE = PROJECT_ROOT / "frontend" / ".env.example"
    COMPOSE_FILE = PROJECT_ROOT / "docker" / "docker-compose.yml"

PLACEHOLDERS = {
    "",
    "admin",
    "change-me-in-production",
    "change-me-to-a-strong-random-value",
    "frontend",
    "generate-a-strong-random-key-here",
    "generate-a-fernet-key-and-paste-here",
    "your_google_api_key_here",
}


def generate_secret_urlsafe(length: int = 32) -> str:
    """Return a URL-safe random secret for API keys and Flask sessions."""
    return secrets.token_urlsafe(length)


def generate_fernet_key() -> str:
    """Return a Fernet-compatible key without importing cryptography."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii")


def parse_env(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE lines from an env file."""
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def ensure_env_file(path: Path, example_path: Path) -> None:
    """Create an env file from its example when missing."""
    if path.exists():
        return
    if not example_path.exists():
        path.touch()
        return
    shutil.copyfile(example_path, path)
    print(f"Created {display_path(path)} from example.")


def display_path(path: Path) -> str:
    """Return a readable path for status messages."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_env_value(path: Path, key: str, value: str) -> None:
    """Update or append a KEY=VALUE line while preserving comments."""
    lines = path.read_text(encoding="utf-8").splitlines()
    updated = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        line_key = stripped.split("=", 1)[0].strip()
        if line_key == key:
            lines[index] = f"{key}={value}"
            updated = True
            break
    if not updated:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def is_placeholder(value: str | None) -> bool:
    """Return True when a value is missing or still looks like a template."""
    if value is None:
        return True
    return value.strip() in PLACEHOLDERS


def confirm(prompt: str, default: bool = False, assume_yes: bool = False) -> bool:
    """Prompt for yes/no confirmation."""
    if assume_yes:
        return default
    suffix = " [Y/n]: " if default else " [y/N]: "
    answer = input(prompt + suffix).strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def maybe_set_secret(
    path: Path,
    values: dict[str, str],
    key: str,
    value_factory: Callable[[], str],
    *,
    assume_yes: bool,
    force: bool,
) -> str:
    """Generate and persist a secret when missing, placeholder, or forced."""
    current = values.get(key)
    if force or is_placeholder(current):
        value = value_factory()
        write_env_value(path, key, value)
        values[key] = value
        print(f"Set {key} in {display_path(path)}.")
        return value

    if confirm(
        f"{key} already exists in {path.name}. Replace it?", assume_yes=assume_yes
    ):
        value = value_factory()
        write_env_value(path, key, value)
        values[key] = value
        print(f"Replaced {key} in {display_path(path)}.")
        return value

    return current or ""


def prompt_secret(label: str, current: str | None, *, force: bool) -> str | None:
    """Prompt for a sensitive optional value."""
    if current and not is_placeholder(current) and not force:
        replace = confirm(f"{label} is already set. Replace it?")
        if not replace:
            return None

    value = getpass.getpass(f"{label} (leave blank to skip): ").strip()
    return value or None


def sync_value(
    source: str, target_path: Path, target_values: dict[str, str], key: str
) -> None:
    """Write a value to an env file and its in-memory mapping."""
    write_env_value(target_path, key, source)
    target_values[key] = source


def reset_backend_service_user_if_present(username: str, api_key: str) -> None:
    """Update an existing backend API user's stored hash to match api_key."""
    try:
        from auth.argon2_helpers import hash_api_key
        from core.config import BACKEND_DB_PATH
        from core.database import get_connection
    except Exception as exc:
        print(f"Skipped backend API user reset; backend modules unavailable: {exc}")
        return

    if not os.path.exists(BACKEND_DB_PATH):
        return

    conn: sqlite3.Connection = get_connection(db_path=BACKEND_DB_PATH)
    try:
        row = conn.execute(
            "SELECT id FROM api_users WHERE username = ?",
            (username,),
        ).fetchone()
        if row is None:
            return

        conn.execute(
            "UPDATE api_users SET api_key_hash = ? WHERE id = ?",
            (hash_api_key(api_key), int(row["id"])),
        )
        conn.commit()
        print(
            f"Reset existing backend API user '{username}' to match FRONTEND_API_KEY."
        )
    finally:
        conn.close()


def collect_llm_credentials(
    backend_values: dict[str, str],
    *,
    force: bool,
    assume_yes: bool,
) -> None:
    """Optionally collect provider credentials for the backend."""
    if not confirm(
        "Configure LLM provider credentials now?", default=False, assume_yes=assume_yes
    ):
        return

    use_ollama = confirm("Use Ollama instead of Google Gemini?", default=True)
    write_env_value(BACKEND_ENV, "USE_OLLAMA", str(use_ollama).lower())
    backend_values["USE_OLLAMA"] = str(use_ollama).lower()

    if use_ollama:
        use_cloud = confirm("Use Ollama Cloud?", default=False)
        write_env_value(BACKEND_ENV, "USE_OLLAMA_CLOUD", str(use_cloud).lower())
        backend_values["USE_OLLAMA_CLOUD"] = str(use_cloud).lower()
        if use_cloud:
            key = prompt_secret(
                "OLLAMA_API_KEY", backend_values.get("OLLAMA_API_KEY"), force=force
            )
            if key:
                write_env_value(BACKEND_ENV, "OLLAMA_API_KEY", key)
        model = input(
            f"OLLAMA_MODEL [{backend_values.get('OLLAMA_MODEL', 'llama3')}]: "
        ).strip()
        if model:
            write_env_value(BACKEND_ENV, "OLLAMA_MODEL", model)
        base_url = input(
            f"OLLAMA_BASE_URL [{backend_values.get('OLLAMA_BASE_URL', 'http://localhost:11434')}]: "
        ).strip()
        if base_url:
            write_env_value(BACKEND_ENV, "OLLAMA_BASE_URL", base_url)
        return

    key = prompt_secret(
        "GOOGLE_API_KEY", backend_values.get("GOOGLE_API_KEY"), force=force
    )
    if key:
        write_env_value(BACKEND_ENV, "GOOGLE_API_KEY", key)
    model = input(
        f"GEMINI_MODEL [{backend_values.get('GEMINI_MODEL', 'gemini-1.5-flash')}]: "
    ).strip()
    if model:
        write_env_value(BACKEND_ENV, "GEMINI_MODEL", model)


def restart_services() -> int:
    """Restart the Docker Compose stack if Docker is available."""
    if COMPOSE_FILE is None:
        print("Docker Compose restart is only available when run from the host repo.")
        return 1
    command = ["docker", "compose", "-f", str(COMPOSE_FILE), "restart"]
    print("Restarting services with Docker Compose...")
    return subprocess.call(command, cwd=PROJECT_ROOT)


def bootstrap(args: argparse.Namespace) -> int:
    """Run the interactive bootstrap workflow."""
    ensure_env_file(BACKEND_ENV, BACKEND_ENV_EXAMPLE)
    if FRONTEND_ENV is not None and FRONTEND_ENV_EXAMPLE is not None:
        ensure_env_file(FRONTEND_ENV, FRONTEND_ENV_EXAMPLE)
    elif not args.backend_only:
        print("Running inside the backend container; frontend .env is not available.")
        print("Continuing in backend-only mode.")

    backend_values = parse_env(BACKEND_ENV)
    frontend_values = parse_env(FRONTEND_ENV) if FRONTEND_ENV is not None else {}

    if FRONTEND_ENV is not None and not args.backend_only:
        maybe_set_secret(
            FRONTEND_ENV,
            frontend_values,
            "FLASK_ENCRYPTION_KEY",
            generate_fernet_key,
            assume_yes=args.yes,
            force=args.force,
        )
        maybe_set_secret(
            FRONTEND_ENV,
            frontend_values,
            "SECRET_KEY",
            generate_secret_urlsafe,
            assume_yes=args.yes,
            force=args.force,
        )
    maybe_set_secret(
        BACKEND_ENV,
        backend_values,
        "ADMIN_API_KEY",
        generate_secret_urlsafe,
        assume_yes=args.yes,
        force=args.force,
    )

    service_username = backend_values.get("FRONTEND_API_USERNAME") or "frontend"
    if args.force or is_placeholder(backend_values.get("FRONTEND_API_KEY")):
        service_key = generate_secret_urlsafe()
        sync_value(
            service_username, BACKEND_ENV, backend_values, "FRONTEND_API_USERNAME"
        )
        sync_value(service_key, BACKEND_ENV, backend_values, "FRONTEND_API_KEY")
        reset_backend_service_user_if_present(service_username, service_key)
        if FRONTEND_ENV is not None and not args.backend_only:
            sync_value(
                service_username, FRONTEND_ENV, frontend_values, "FRONTEND_API_USERNAME"
            )
            sync_value(service_key, FRONTEND_ENV, frontend_values, "FRONTEND_API_KEY")
            print("Generated matching frontend/backend service credentials.")
        else:
            print("Generated backend service credentials.")
    else:
        service_key = backend_values["FRONTEND_API_KEY"]
        reset_backend_service_user_if_present(service_username, service_key)
        if FRONTEND_ENV is not None and not args.backend_only:
            sync_value(
                service_username, FRONTEND_ENV, frontend_values, "FRONTEND_API_USERNAME"
            )
            sync_value(service_key, FRONTEND_ENV, frontend_values, "FRONTEND_API_KEY")
            print("Synced existing backend service credentials into frontend env.")

    if FRONTEND_ENV is not None and not args.backend_only:
        admin_user = frontend_values.get("FRONTEND_ADMIN_USERNAME") or "admin"
        sync_value(admin_user, FRONTEND_ENV, frontend_values, "FRONTEND_ADMIN_USERNAME")
        if args.force or is_placeholder(frontend_values.get("FRONTEND_ADMIN_PASSWORD")):
            if args.yes:
                admin_password = generate_secret_urlsafe()
                print("Generated a frontend admin password in frontend/.env.")
            else:
                admin_password = getpass.getpass("Frontend admin password: ").strip()
            if admin_password:
                sync_value(
                    admin_password,
                    FRONTEND_ENV,
                    frontend_values,
                    "FRONTEND_ADMIN_PASSWORD",
                )
            else:
                print(
                    "Skipped frontend admin password; current value was left unchanged."
                )

    if not args.skip_llm:
        collect_llm_credentials(backend_values, force=args.force, assume_yes=args.yes)

    if args.restart:
        return restart_services()

    print("")
    print("Bootstrap complete.")
    if COMPOSE_FILE is not None:
        print("Restart services so env changes take effect:")
        print(f"  docker compose -f {COMPOSE_FILE.relative_to(PROJECT_ROOT)} restart")
    else:
        print("Restart the backend container so env changes take effect.")
        print(
            "Note: /app/.env changes are container-local unless .env is bind-mounted."
        )
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes", action="store_true", help="accept default yes/no prompts"
    )
    parser.add_argument(
        "--force", action="store_true", help="replace existing non-placeholder secrets"
    )
    parser.add_argument(
        "--backend-only", action="store_true", help="only update the backend .env"
    )
    parser.add_argument(
        "--skip-llm", action="store_true", help="do not prompt for LLM credentials"
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="restart Docker Compose services when done",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(bootstrap(parse_args(sys.argv[1:])))
