"""
Configuration classes for the Flask forecaster frontend.

Three environments are provided:

* ``DevelopmentConfig`` — local development, debug enabled.
* ``ProductionConfig``  — production deployment, strict security settings.
* ``TestingConfig``     — automated test runs, in-memory DB.

Select the active configuration by setting the ``FLASK_ENV`` environment
variable or by passing *config_name* to the application factory.
"""

from __future__ import annotations

import os
import secrets
from datetime import timedelta
from pathlib import Path

_INSTANCE_DIR = Path(__file__).resolve().parent / "instance"
_SESSION_KEY_FILE = _INSTANCE_DIR / ".session_key"
_LEGACY_ENV_FILE = Path(__file__).resolve().parent / ".env"


def _legacy_env_value(key: str) -> str | None:
    """Return a value from the pre-wizard ``.env`` file, if it exists.

    This is a one-time migration path only. Runtime configuration no longer
    loads dotenv files.
    """
    if not _LEGACY_ENV_FILE.is_file():
        return None
    for line in _LEGACY_ENV_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            name, value = stripped.split("=", 1)
            if name.strip() == key:
                return value.strip().strip("\"'") or None
    return None


def _read_or_create_secret(path: Path, legacy_key: str) -> str:
    """Load a stable instance secret or generate it securely on first run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()

    value = _legacy_env_value(legacy_key) or secrets.token_urlsafe(48)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path.read_text(encoding="utf-8").strip()
    try:
        os.write(descriptor, value.encode("utf-8"))
    finally:
        os.close(descriptor)
    return value


class BaseConfig:
    """Shared settings inherited by all environment configurations."""

    SECRET_KEY: str = _read_or_create_secret(_SESSION_KEY_FILE, "SECRET_KEY")
    DATABASE: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "instance", "forecaster.db"
    )
    BACKEND_URL: str = ""
    API_VERIFY_SSL: bool = False
    DEMO_DATA_PATH: str = os.environ.get(
        "DEMO_DATA_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data.csv"),
    )

    SESSION_TYPE: str = "filesystem"
    SESSION_FILE_DIR: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "instance", "sessions"
    )
    SESSION_FILE_THRESHOLD: int = 500
    SESSION_PERMANENT: bool = True
    PERMANENT_SESSION_LIFETIME: timedelta = timedelta(hours=24)

    WTF_CSRF_ENABLED: bool = True
    WTF_CSRF_TIME_LIMIT: int = 3600

    MAX_CONTENT_LENGTH: int = 100 * 1024 * 1024


class DevelopmentConfig(BaseConfig):
    """Configuration for local development."""

    DEBUG: bool = True
    TESTING: bool = False


class ProductionConfig(BaseConfig):
    """Configuration for production deployment.

    Uses the generated, persistent instance secret and disables debug output.
    """

    DEBUG: bool = False
    TESTING: bool = False

    SESSION_COOKIE_SECURE: bool = True
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = "Lax"

class TestingConfig(BaseConfig):
    """Configuration for automated testing."""

    TESTING: bool = True
    DEBUG: bool = True
    WTF_CSRF_ENABLED: bool = False
    DATABASE: str = ":memory:"
    SESSION_TYPE: str = "filesystem"


_CONFIG_MAP: dict[str, type[BaseConfig]] = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config(name: str = "development") -> BaseConfig:
    """Return a configuration instance for the given environment name.

    Args:
        name: One of ``'development'``, ``'production'``, or ``'testing'``.

    Returns:
        An instantiated configuration object.

    Raises:
        KeyError: When *name* is not a recognised environment.
    """
    config_cls = _CONFIG_MAP[name]
    return config_cls()
