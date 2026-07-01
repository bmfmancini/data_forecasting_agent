"""
Flask application factory for the Time Series Data Forecaster frontend.

Call :func:`create_app` to obtain a configured Flask application instance.
The factory pattern enables multiple application instances (useful for
testing) and defers extension initialisation until an app is available.
"""

from __future__ import annotations

import os
from typing import Any

from flask import Flask, session
from flask_session import Session  # type: ignore[import-untyped]

from config import get_config
from db.db import init_app as db_init_app, init_db, query_db
from extensions import csrf, login_manager
from models import User


def create_app(config_name: str | None = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        config_name: Environment name — ``'development'``, ``'production'``,
            or ``'testing'``.  Defaults to the value of the ``FLASK_ENV``
            environment variable, falling back to ``'development'``.

    Returns:
        A fully configured :class:`flask.Flask` instance with all blueprints
        registered, extensions initialised, and the database seeded.
    """
    if config_name is None:
        config_name = os.environ.get("FLASK_ENV", "development")

    app = Flask(__name__, instance_relative_config=True)

    cfg = get_config(config_name)
    app.config.from_object(cfg)

    _ensure_instance_dirs(app)

    login_manager.init_app(app)
    login_manager.login_view = "auth.login"  # type: ignore[assignment]
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "warning"

    csrf.init_app(app)

    Session(app)

    db_init_app(app)

    with app.app_context():
        init_db()
        _sync_backend_url_from_db(app)

    _register_blueprints(app)
    _register_context_processors(app)
    _register_user_loader()

    from manage import register_commands
    register_commands(app)

    return app


def _ensure_instance_dirs(app: Flask) -> None:
    """Create instance sub-directories that must exist before startup.

    Args:
        app: The Flask application instance whose ``instance_path`` is used.
    """
    for sub in ("", "sessions"):
        path = os.path.join(app.instance_path, sub) if sub else app.instance_path
        os.makedirs(path, exist_ok=True)

    session_dir = app.config.get("SESSION_FILE_DIR", "")
    if session_dir:
        os.makedirs(session_dir, exist_ok=True)


def _sync_backend_url_from_db(app: Flask) -> None:
    """Override ``BACKEND_URL`` in app config with the value stored in the DB.

    This allows the administrator to update the backend URL via the admin
    panel without restarting the application.

    Args:
        app: The Flask application instance to update.
    """
    row = query_db(
        "SELECT base_url FROM api_credentials WHERE label = 'default' LIMIT 1",
        one=True,
    )
    if row and isinstance(row, dict):
        url = row.get("base_url", "")
        if url:
            app.config["BACKEND_URL"] = url


def _register_blueprints(app: Flask) -> None:
    """Attach all application blueprints to *app*.

    Args:
        app: The Flask application instance.
    """
    from blueprints.auth import auth_bp
    from blueprints.main import main_bp
    from blueprints.admin import admin_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(admin_bp)


def _register_context_processors(app: Flask) -> None:
    """Register template context processors that inject session state.

    The sidebar in ``base.html`` needs access to upload state, job state,
    and analysis state on every page without each route having to pass
    them explicitly.

    Args:
        app: The Flask application instance.
    """

    @app.context_processor
    def inject_sidebar_state() -> dict[str, Any]:
        """Expose session state to all Jinja templates."""
        return {
            "upload_info": session.get("upload_info"),
            "date_col": session.get("date_col"),
            "value_col": session.get("value_col"),
            "forecast_horizon": session.get("forecast_horizon", 12),
            "model_choice": session.get("model_choice", "Auto (AI selects)"),
            "user_prompt": session.get("user_prompt", ""),
            "job_running": session.get("job_running", False),
            "job_progress": session.get("job_progress", 0),
            "job_step": session.get("job_step", ""),
            "analysis_complete": session.get("analysis_result") is not None,
            "preflight_result": session.get("preflight_result"),
            "preflight_options": session.get("preflight_options", {}),
            "analysis_error": session.get("analysis_error"),
        }


def _register_user_loader() -> None:
    """Register the Flask-Login user loader callback."""

    @login_manager.user_loader
    def load_user(user_id: str) -> User | None:
        """Load a user from the database by their string identifier.

        Args:
            user_id: String representation of the user's integer primary key.

        Returns:
            A :class:`~models.User` instance or ``None`` when not found.
        """
        from db.db import query_db as _query

        row = _query(
            """
            SELECT u.id, u.username, r.name AS role_name, u.active
            FROM users u
            JOIN roles r ON r.id = u.role_id
            WHERE u.id = ?
            """,
            (int(user_id),),
            one=True,
        )
        if row and isinstance(row, dict):
            return User(
                user_id=int(row["id"]),
                username=str(row["username"]),
                role_name=str(row["role_name"]),
                active=bool(row["active"]),
            )
        return None
