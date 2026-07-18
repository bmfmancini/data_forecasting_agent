"""Signing helpers for delegated frontend application-user identities."""

from __future__ import annotations

import hashlib
import hmac


def application_user_signature(application_user_id: int, secret: str) -> str:
    """Return the HMAC signature for a delegated application user ID.

    Args:
        application_user_id: Frontend application user ID being delegated.
        secret: Pre-shared identity-signing secret.

    Returns:
        A lowercase hexadecimal SHA-256 HMAC digest.
    """
    return hmac.new(
        secret.encode("utf-8"),
        str(application_user_id).encode("ascii"),
        hashlib.sha256,
    ).hexdigest()


def verify_application_user_signature(
    application_user_id: int,
    signature: str,
    secret: str,
) -> bool:
    """Return whether a delegated application-user signature is valid.

    Args:
        application_user_id: Frontend application user ID from the request.
        signature: Signature supplied by the trusted frontend service.
        secret: Pre-shared identity-signing secret.

    Returns:
        ``True`` when the supplied signature matches the expected digest.
    """
    expected = application_user_signature(application_user_id, secret)
    return hmac.compare_digest(signature, expected)
