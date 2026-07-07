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
from datetime import timedelta


class BaseConfig:
    """Shared settings inherited by all environment configurations."""

    SECRET_KEY: str = os.environ.get("SECRET_KEY", "change-me-in-production")
    DATABASE: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "instance", "forecaster.db"
    )
    BACKEND_URL: str = os.environ.get("BACKEND_URL", "http://localhost:8000")
    API_VERIFY_SSL: bool = os.environ.get("API_VERIFY_SSL", "false").lower() == "true"
    # Pre-shared service-account credentials for the FastAPI backend.
    # Defaults to ``frontend``/``frontend`` so the stack works out-of-the-box.
    # The admin MUST rotate the key for production via the admin panel.
    FRONTEND_API_USERNAME: str = os.environ.get("FRONTEND_API_USERNAME", "frontend")
    FRONTEND_API_KEY: str = os.environ.get("FRONTEND_API_KEY", "frontend")
    # Default admin login credentials for the Flask frontend.
    # Defaults to ``admin``/``admin`` so the stack works out-of-the-box.
    # The admin MUST set a strong password in .env at setup time and
    # will be forced to change it on first login.
    FRONTEND_ADMIN_USERNAME: str = os.environ.get(
        "FRONTEND_ADMIN_USERNAME", "admin"
    )
    FRONTEND_ADMIN_PASSWORD: str = os.environ.get(
        "FRONTEND_ADMIN_PASSWORD", "admin"
    )
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

    MAX_CONTENT_LENGTH: int = int(os.environ.get("MAX_UPLOAD_MB", "200")) * 1024 * 1024


class DevelopmentConfig(BaseConfig):
    """Configuration for local development."""

    DEBUG: bool = True
    TESTING: bool = False


class ProductionConfig(BaseConfig):
    """Configuration for production deployment.

    Enforces a strong ``SECRET_KEY`` from the environment and disables
    debug output.
    """

    DEBUG: bool = False
    TESTING: bool = False

    def __init__(self) -> None:
        super().__init__()
        if (
            self.FRONTEND_API_USERNAME == "frontend"
            and self.FRONTEND_API_KEY == "frontend"
        ):
            raise ValueError(
                "Default FRONTEND_API_USERNAME and FRONTEND_API_KEY values are not allowed in production. "
                "Please set these values in the environment."
            )

    SESSION_COOKIE_SECURE: bool = True
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = "Lax"

    SECRET_KEY: str = os.environ.get("SECRET_KEY", "")

    def __init__(self) -> None:
        """Raise when ``SECRET_KEY`` is absent in production."""
        if not self.SECRET_KEY:
            raise RuntimeError(
                "SECRET_KEY environment variable must be set in production."
            )


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
