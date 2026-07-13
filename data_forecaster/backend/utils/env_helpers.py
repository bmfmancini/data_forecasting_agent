"""Shared environment-variable parsing helpers.

Centralises the small ``env_int`` / ``env_float`` helpers that were
previously duplicated across :mod:`report.rules` and
:mod:`prompts.prompt_utils`.  Both modules now import from here so the
parsing logic lives in one place.
"""

from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)


def env_int(key: str, default: int) -> int:
    """Read an integer environment variable with a fallback.

    Args:
        key:     Environment variable name.
        default: Fallback value if unset or unparseable.

    Returns:
        The integer value from the environment or the default.
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        logger.warning(
            "Invalid value for env var '%s': '%s'. Using default: %s",
            key,
            value,
            default,
        )
        return default


def env_float(key: str, default: float) -> float:
    """Read a float environment variable with a fallback.

    Args:
        key:     Environment variable name.
        default: Fallback value if unset or unparseable.

    Returns:
        The float value from the environment or the default.
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning(
            "Invalid value for env var '%s': '%s'. Using default: %s",
            key,
            value,
            default,
        )
        return default
