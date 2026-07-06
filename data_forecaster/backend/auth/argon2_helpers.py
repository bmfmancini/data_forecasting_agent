"""Argon2id hashing and verification helpers for API keys.

Provides functions to hash plaintext API keys using Argon2id (via
``argon2-cffi``), verify a plaintext key against a stored hash, and
generate cryptographically secure random API keys.
"""

from __future__ import annotations

import secrets

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from core.logging_config import get_logger

logger = get_logger(__name__)

_hasher: PasswordHasher = PasswordHasher()


def generate_api_key() -> str:
    """Generate a cryptographically secure random API key.

    Uses :func:`secrets.token_urlsafe` with 32 bytes of entropy,
    producing a ~43-character URL-safe base64 string.

    Returns:
        A random API key string.
    """
    return secrets.token_urlsafe(32)


def hash_api_key(plaintext: str) -> str:
    """Hash an API key using Argon2id.

    Args:
        plaintext: The plaintext API key to hash.

    Returns:
        The Argon2id hash string suitable for database storage.
    """
    return _hasher.hash(plaintext)


def verify_api_key(plaintext: str, hash_str: str) -> bool:
    """Verify a plaintext API key against a stored Argon2id hash.

    Args:
        plaintext: The plaintext API key supplied by the client.
        hash_str:  The stored Argon2id hash from the database.

    Returns:
        ``True`` when the key matches the hash, ``False`` otherwise.
        Never raises on mismatch — returns ``False`` so callers can
        return a generic 401 without revealing which field was wrong.
    """
    try:
        return _hasher.verify(hash_str, plaintext)
    except VerifyMismatchError:
        return False
    except Exception as exc:
        logger.warning("Argon2 verification error: %s", exc)
        return False
