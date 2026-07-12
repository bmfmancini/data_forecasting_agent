"""
Flask application factory for the Time Series Data Forecaster frontend.

Call :func:`create_app` to obtain a configured Flask application instance.
The factory pattern enables multiple application instances (useful for
testing) and defers extension initialisation until an app is available.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from flask import Flask, jsonify, redirect, request, session, url_for
from flask_login import current_user
from flask_session import Session  # type: ignore[import-untyped]
from werkzeug.wrappers import Response

from blueprints.admin import admin_bp
from blueprints.auth import auth_bp
from blueprints.main import main_bp
from config import get_config
from db.db import init_app as db_init_app, init_db, query_db
from extensions import csrf, login_manager
from manage import register_commands
from models import User
from services.markdown_service import markdown_to_safe_html

logger = logging.getLogger(__name__)


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
        _sync_app_config_from_db(app)
        _sync_backend_url_from_db(app)

    _register_template_filters(app)
    _register_blueprints(app)
    _register_context_processors(app)
    _register_user_loader()
    _register_password_change(app)

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


def _sync_app_config_from_db(app: Flask) -> None:
    """Load DB-owned frontend settings into Flask runtime config."""
    row = query_db(
        "SELECT value FROM app_config WHERE key = 'max_upload_mb'",
        one=True,
    )
    if row and isinstance(row, dict):
        try:
            max_upload_mb = max(int(row.get("value", 100)), 1)
        except (TypeError, ValueError):
            max_upload_mb = 100
        app.config["MAX_CONTENT_LENGTH"] = max_upload_mb * 1024 * 1024


def _register_template_filters(app: Flask) -> None:
    """Register custom Jinja2 template filters.

    Provides a ``md`` filter that converts markdown-formatted text (typically
    produced by the LLM agents) to sanitised HTML using ``bleach``.

    Args:
        app: The Flask application instance.
    """

    @app.template_filter("md")
    def _markdown_to_html(text: str) -> str:
        """Convert a markdown string to sanitised HTML.

        Args:
            text: Markdown-formatted string.

        Returns:
            Safe HTML string with unsafe tags stripped.
        """
        return markdown_to_safe_html(text)


def _register_blueprints(app: Flask) -> None:
    """Attach all application blueprints to *app*.

    Args:
        app: The Flask application instance.
    """
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
        try:
            user_pk = int(user_id)
        except (TypeError, ValueError):
            return None

        row = query_db(
            """
            SELECT u.id, u.username, r.name AS role_name, u.active,
                   u.must_change_password, u.session_version
            FROM users u
            JOIN roles r ON r.id = u.role_id
            WHERE u.id = ?
            """,
            (user_pk,),
            one=True,
        )
        if row and isinstance(row, dict):
            if not bool(row.get("active")):
                return None
            session_version = int(row.get("session_version", 0))
            if session.get("user_session_version") != session_version:
                return None
            return User(
                user_id=int(row["id"]),
                username=str(row["username"]),
                role_name=str(row["role_name"]),
                active=bool(row["active"]),
                must_change_password=bool(row.get("must_change_password", 0)),
                session_version=session_version,
            )
        return None


def _register_password_change(app: Flask) -> None:
    """Enforce a forced password change across all blueprints.

    A user whose ``must_change_password`` flag is set may only access the
    ``auth.change_password`` and ``auth.logout`` endpoints.  Every other
    request — page or AJAX — is redirected (or, for JSON requests, rejected
    with a 403) to the change-password page.

    Args:
        app: The Flask application instance.
    """

    @app.before_request
    def _enforce_password_change() -> Response | tuple[str, int] | None:
        """Redirect or reject requests when a password change is required."""
        if not current_user.is_authenticated:
            return None

        if not getattr(current_user, "must_change_password", False):
            return None

        endpoint = request.endpoint or ""
        # Allow the user to reach the change-password page and log out.
        if endpoint in ("auth.change_password", "auth.logout", "static"):
            return None

        # AJAX/JSON callers get a structured error instead of a redirect.
        if request.path.startswith("/api/") or _wants_json():
            return jsonify({"error": "Password change required."}), 403

        return redirect(url_for("auth.change_password"))


def _wants_json() -> bool:
    """Return True when the client prefers a JSON response."""
    accept = request.accept_mimetypes
    return accept.best_match(["application/json", "text/html"]) == "application/json"
