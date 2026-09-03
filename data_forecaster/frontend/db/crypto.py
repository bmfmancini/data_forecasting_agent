"""
Symmetric encryption helpers for storing sensitive configuration at rest.

Uses Fernet (AES-128-CBC + HMAC-SHA256) from the ``cryptography`` library.
The encryption key is generated on first run and stored in the persistent
Flask instance directory with mode 0600. It is never stored in the database.
"""

from __future__ import annotations

import os
from pathlib import Path

from cryptography.fernet import Fernet


_INSTANCE_DIR = Path(__file__).resolve().parents[1] / "instance"
_KEY_FILE = _INSTANCE_DIR / ".encryption_key"
_LEGACY_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def _legacy_env_key() -> bytes | None:
    """Read the legacy key once so existing installs can migrate safely."""
    if not _LEGACY_ENV_FILE.is_file():
        return None
    for line in _LEGACY_ENV_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            name, value = stripped.split("=", 1)
            if name.strip() == "FLASK_ENCRYPTION_KEY" and value.strip():
                return value.strip().strip("\"'").encode()
    return None


def _read_or_create_key() -> bytes:
    """Load the persistent key, migrating legacy dotenv data if necessary."""
    _INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    if _KEY_FILE.is_file():
        return _KEY_FILE.read_bytes().strip()

    key = _legacy_env_key() or Fernet.generate_key()
    try:
        descriptor = os.open(_KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return _KEY_FILE.read_bytes().strip()
    try:
        os.write(descriptor, key)
    finally:
        os.close(descriptor)
    return key


def get_fernet() -> Fernet:
    """Return a Fernet instance initialised from the persistent key.

    Raises:
        ValueError:   When the key is not a valid 32-byte URL-safe
                      base-64-encoded value.
    """
    return Fernet(_read_or_create_key())


def encrypt(plaintext: str) -> str:
    """Encrypt *plaintext* and return a URL-safe base-64-encoded ciphertext.

    Args:
        plaintext: The string value to encrypt.

    Returns:
        The encrypted token as a decoded string (safe to store in the DB).
    """
    return get_fernet().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    """Decrypt *ciphertext* and return the original plaintext string.

    Args:
        ciphertext: An encrypted token previously produced by :func:`encrypt`.

    Returns:
        The decrypted plaintext string.

    Raises:
        cryptography.fernet.InvalidToken: When the token is invalid or
            the key does not match.
    """
    return get_fernet().decrypt(ciphertext.encode()).decode()
