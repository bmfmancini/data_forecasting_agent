"""Shared environment-variable parsing helpers.

Centralises the small ``env_int`` / ``env_float`` helpers that were
previously duplicated across :mod:`report.rules` and
:mod:`prompts.prompt_utils`.  Both modules now import from here so the
parsing logic lives in one place.
"""

from __future__ import annotations

import os


def env_int(key: str, default: int) -> int:
    """Read an integer environment variable with a fallback.

    Args:
        key:     Environment variable name.
        default: Fallback value if unset or unparseable.

    Returns:
        The integer value from the environment or the default.
    """
    try:
        return int(os.getenv(key, default))
    except (ValueError, TypeError):
        return default


def env_float(key: str, default: float) -> float:
    """Read a float environment variable with a fallback.

    Args:
        key:     Environment variable name.
        default: Fallback value if unset or unparseable.

    Returns:
        The float value from the environment or the default.
    """
    try:
        return float(os.getenv(key, default))
    except (ValueError, TypeError):
        return default