"""Shared persistence helpers for backend API credentials.

Extracted from the admin blueprint so both the admin panel and the
first-run setup wizard can upsert the ``api_credentials`` row without
importing private helpers across blueprints.
"""

from __future__ import annotations

from db.db import execute_db


def save_api_credentials(
    base_url: str,
    timeout: int,
    verify_ssl: int,
    enc_user: str | None,
    enc_pass: str | None,
    preserve_existing_key: bool = False,
) -> None:
    """Upsert the default API credential row.

    When both encrypted values are supplied the row is fully updated;
    otherwise only ``base_url``, ``timeout``, and ``verify_ssl`` are
    touched, preserving any existing encrypted credentials.

    Args:
        base_url:              Root URL of the FastAPI backend.
        timeout:               Request timeout in seconds.
        verify_ssl:            ``1`` to verify the backend TLS certificate,
                               ``0`` to skip verification.
        enc_user:              Fernet-encrypted API username, or ``None``.
        enc_pass:              Fernet-encrypted API key, or ``None``.
        preserve_existing_key: When ``True`` and only ``enc_user`` is
                               supplied, keep the stored encrypted key.
    """
    if enc_user and enc_pass:
        execute_db(
            """
            INSERT INTO api_credentials
                (label, base_url, encrypted_username, encrypted_password,
                 timeout, verify_ssl)
            VALUES ('default', ?, ?, ?, ?, ?)
            ON CONFLICT(label) DO UPDATE SET
                base_url           = excluded.base_url,
                encrypted_username = excluded.encrypted_username,
                encrypted_password = excluded.encrypted_password,
                timeout            = excluded.timeout,
                verify_ssl         = excluded.verify_ssl
            """,
            (base_url, enc_user, enc_pass, timeout, verify_ssl),
        )
    elif enc_user and preserve_existing_key:
        execute_db(
            """
            INSERT INTO api_credentials
                (label, base_url, encrypted_username, encrypted_password,
                 timeout, verify_ssl)
            VALUES ('default', ?, ?, NULL, ?, ?)
            ON CONFLICT(label) DO UPDATE SET
                base_url           = excluded.base_url,
                encrypted_username = excluded.encrypted_username,
                timeout            = excluded.timeout,
                verify_ssl         = excluded.verify_ssl
            """,
            (base_url, enc_user, timeout, verify_ssl),
        )
    else:
        execute_db(
            """
            INSERT INTO api_credentials
                (label, base_url, encrypted_username, encrypted_password,
                 timeout, verify_ssl)
            VALUES ('default', ?, NULL, NULL, ?, ?)
            ON CONFLICT(label) DO UPDATE SET
                base_url   = excluded.base_url,
                timeout    = excluded.timeout,
                verify_ssl = excluded.verify_ssl
            """,
            (base_url, timeout, verify_ssl),
        )
