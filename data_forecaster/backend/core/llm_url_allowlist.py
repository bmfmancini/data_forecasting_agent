"""Persistent, administrator-managed destinations for LLM connection tests."""

from __future__ import annotations

import json
from urllib.parse import urlsplit

from core.database import transaction


def normalize_origins(origins: list[str]) -> list[str]:
    """Validate exact HTTP(S) base URLs without expanding their authority."""
    if len(origins) > 100:
        raise ValueError("Enter at most 100 allowed URLs.")
    normalized = []
    for origin in origins:
        url = origin.strip().rstrip("/")
        try:
            parsed = urlsplit(url)
            valid = (
                0 < len(url) <= 2048
                and parsed.scheme in {"http", "https"}
                and parsed.hostname
                and parsed.username is None
                and parsed.password is None
                and not any(char in url for char in "\\?#*")
                and not any(char.isspace() or ord(char) < 32 for char in url)
            )
            _ = parsed.port
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(
                "Enter exact HTTP(S) base URLs without credentials, wildcards, "
                "queries, or fragments."
            )
        if url not in normalized:
            normalized.append(url)
    return normalized


def get_allowed_origins() -> list[str]:
    """Read current policy on every call so changes apply across workers."""
    with transaction() as connection:
        row = connection.execute(
            "SELECT origins_json FROM llm_url_allowlist WHERE singleton = 1"
        ).fetchone()
    # A missing policy fails closed, never silently restoring removed defaults.
    return json.loads(row["origins_json"]) if row else []


def put_allowed_origins(origins: list[str]) -> list[str]:
    """Replace the policy atomically after validating every entry."""
    normalized = normalize_origins(origins)
    with transaction() as connection:
        connection.execute(
            "INSERT INTO llm_url_allowlist (singleton, origins_json) VALUES (1, ?) "
            "ON CONFLICT(singleton) DO UPDATE SET "
            "origins_json = excluded.origins_json, updated_at = datetime('now')",
            (json.dumps(normalized),),
        )
    return normalized
