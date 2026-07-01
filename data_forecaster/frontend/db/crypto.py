"""
Symmetric encryption helpers for storing sensitive configuration at rest.

Uses Fernet (AES-128-CBC + HMAC-SHA256) from the ``cryptography`` library.
The encryption key must be provided via the ``FLASK_ENCRYPTION_KEY``
environment variable and must never be stored in the database.
"""

from __future__ import annotations

import os

from cryptography.fernet import Fernet


def get_fernet() -> Fernet:
    """Return a Fernet instance initialised from the environment key.

    Raises:
        RuntimeError: When ``FLASK_ENCRYPTION_KEY`` is not set.
        ValueError:   When the key is not a valid 32-byte URL-safe
                      base-64-encoded value.
    """
    raw_key = os.environ.get("FLASK_ENCRYPTION_KEY", "")
    if not raw_key:
        raise RuntimeError(
            "FLASK_ENCRYPTION_KEY environment variable is not set. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    return Fernet(raw_key.encode())


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
