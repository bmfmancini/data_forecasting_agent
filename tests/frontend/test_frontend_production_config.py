"""Security regression tests for frontend production configuration."""

from __future__ import annotations

from config import get_config


def test_production_config_uses_persistent_generated_secret() -> None:
    """Production startup uses a stable non-empty instance signing secret."""
    first = get_config("production").SECRET_KEY
    second = get_config("production").SECRET_KEY

    assert first
    assert first == second
