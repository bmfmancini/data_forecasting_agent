"""Backend symmetric encryption helpers for secrets stored at rest.

Uses Fernet (AES-128-CBC + HMAC-SHA256) from the ``cryptography`` library,
mirroring ``frontend/db/crypto.py``. Unlike the frontend — which reads its
key from an environment variable — the backend GENERATES its key at setup
time and persists it to a dedicated secrets volume (``SECRETS_DIR``) with
mode 0600 so that no secret ever needs to live in a ``.env`` file.

Losing the key file renders all stored ciphertext unrecoverable; see
``docs/`` for the backup and rotation runbook.
"""

from __future__ import annotations

import os
import threading

from cryptography.fernet import Fernet

import core.config as settings
from core.logging_config import get_logger

logger = get_logger(__name__)

_KEY_FILENAME = ".encryption_key"

_fernet: Fernet | None = None
_fernet_lock = threading.Lock()


def _key_path() -> str:
    """Return the absolute path of the persisted encryption key file."""
    return os.path.join(settings.SECRETS_DIR, _KEY_FILENAME)


def key_file_exists() -> bool:
    """Return ``True`` when the persisted encryption key file is present."""
    return os.path.isfile(_key_path())


def generate_and_persist_key() -> None:
    """Generate a new Fernet key and persist it to the secrets volume.

    The key file is written with mode 0600. The write is skipped (and the
    existing key kept) when the file already exists, so this function is
    safe to call on every setup bootstrap.

    Raises:
        OSError: When the key file cannot be written.
    """
    path = _key_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path):
        logger.info("Encryption key already present at %s; keeping it.", path)
        return
    key = Fernet.generate_key()
    # Open with O_EXCL so a concurrent first-run never truncates an
    # existing key, and set 0600 before any content is written.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)
    logger.info("Generated new encryption key at %s (mode 0600).", path)


def get_fernet() -> Fernet:
    """Return the process-wide Fernet instance, loading the key from disk.

    Returns:
        A :class:`~cryptography.fernet.Fernet` initialised from the
        persisted key.

    Raises:
        RuntimeError: When the key file is missing. Callers that have
            stored ciphertext must treat this as fatal — the data is
            unrecoverable without the key.
    """
    global _fernet
    if _fernet is not None:
        return _fernet
    with _fernet_lock:
        if _fernet is not None:
            return _fernet
        path = _key_path()
        if not os.path.isfile(path):
            raise RuntimeError(
                f"Backend encryption key not found at {path}. Stored "
                "credentials cannot be decrypted without it. Restore the "
                "key from backup, or wipe the stored configuration and run "
                "setup again."
            )
        with open(path, "rb") as handle:
            _fernet = Fernet(handle.read().strip())
        return _fernet


def encrypt(plaintext: str) -> str:
    """Encrypt *plaintext* and return a URL-safe base64 ciphertext string.

    Args:
        plaintext: The secret value to encrypt.

    Returns:
        The encrypted token, safe to store in the database.
    """
    return get_fernet().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    """Decrypt *ciphertext* and return the original plaintext.

    Args:
        ciphertext: A token previously produced by :func:`encrypt`.

    Returns:
        The decrypted plaintext string.

    Raises:
        cryptography.fernet.InvalidToken: When the token is invalid or the
            key does not match.
    """
    return get_fernet().decrypt(ciphertext.encode()).decode()


def reset_cache() -> None:
    """Drop the cached Fernet instance (test support only)."""
    global _fernet
    with _fernet_lock:
        _fernet = None
