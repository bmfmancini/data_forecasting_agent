"""Security regression tests for frontend production configuration."""

from __future__ import annotations

import importlib
import os
import sys

import pytest

_frontend_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data_forecaster", "frontend")
)
if _frontend_dir not in sys.path:
    sys.path.insert(0, _frontend_dir)


def _reload_config(monkeypatch: pytest.MonkeyPatch, secret_key: str | None):
    """Reload frontend config after changing SECRET_KEY."""
    if secret_key is None:
        monkeypatch.delenv("SECRET_KEY", raising=False)
    else:
        monkeypatch.setenv("SECRET_KEY", secret_key)

    import config  # pylint: disable=import-outside-toplevel

    return importlib.reload(config)


def test_production_config_requires_secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production startup must fail when no signing secret is configured."""
    config = _reload_config(monkeypatch, None)
    monkeypatch.setattr(config.ProductionConfig, "SECRET_KEY", "")

    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        config.get_config("production")


def test_production_config_uses_configured_secret_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production config accepts a non-empty externally supplied secret."""
    config = _reload_config(monkeypatch, "test-production-secret")

    assert config.get_config("production").SECRET_KEY == "test-production-secret"
