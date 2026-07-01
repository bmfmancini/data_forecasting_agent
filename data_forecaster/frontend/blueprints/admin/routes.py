"""
Route handlers for the administration blueprint.

All routes require the user to be authenticated AND to hold the ``admin``
role.  The ``admin_required`` decorator enforces this.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar

from flask import (
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required
from werkzeug.security import generate_password_hash
from werkzeug.wrappers import Response

from blueprints.admin import admin_bp
from blueprints.admin.forms import APIConfigForm, UserCreateForm, UserEditForm
from db.db import execute_db, query_db

_F = TypeVar("_F", bound=Callable[..., Any])


def admin_required(f: _F) -> _F:
    """Decorator that restricts access to users with the admin role.

    Applies ``login_required`` first, then checks the current user's role.

    Args:
        f: The route function to protect.

    Returns:
        The wrapped function.
    """

    @wraps(f)
    @login_required  # type: ignore[misc]
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not current_user.is_authenticated or not current_user.is_admin:  # type: ignore[union-attr]
            flash("Administrator access required.", "danger")
            return redirect(url_for("main.index"))
        return f(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


@admin_bp.route("/")
@admin_required
def dashboard() -> str:
    """Render the admin dashboard.

    Shows summary statistics (user count, backend status) and quick links.

    Returns:
        Rendered HTML for the admin dashboard.
    """
    user_count: int = 0
    rows = query_db("SELECT COUNT(*) AS cnt FROM users")
    if rows and isinstance(rows, list):
        user_count = int(rows[0].get("cnt", 0))

    backend_ok = _check_backend_health()

    return render_template(
        "admin/dashboard.html",
        user_count=user_count,
        backend_ok=backend_ok,
    )


@admin_bp.route("/users")
@admin_required
def users() -> str:
    """List all application users.

    Returns:
        Rendered HTML for the user management list page.
    """
    rows = query_db(
        """
        SELECT u.id, u.username, r.name AS role, u.active, u.created_at
        FROM users u
        JOIN roles r ON r.id = u.role_id
        ORDER BY u.id
        """
    )
    user_list: list[dict[str, Any]] = rows if isinstance(rows, list) else []
    return render_template("admin/users.html", users=user_list)


@admin_bp.route("/users/new", methods=["GET", "POST"])
@admin_required
def user_new() -> str | Response:
    """Render and process the create-user form.

    Returns:
        Rendered form template on GET or validation error; redirect on success.
    """
    form = UserCreateForm()
    if form.validate_on_submit():
        username: str = str(form.username.data or "").strip()
        password: str = str(form.password.data or "")
        confirm: str = str(form.confirm_password.data or "")
        role_name: str = str(form.role.data or "user")

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("admin/user_form.html", form=form, edit=False)

        role_row = query_db(
            "SELECT id FROM roles WHERE name = ?", (role_name,), one=True
        )
        if not role_row or not isinstance(role_row, dict):
            flash("Invalid role.", "danger")
            return render_template("admin/user_form.html", form=form, edit=False)

        existing = query_db(
            "SELECT id FROM users WHERE username = ?", (username,), one=True
        )
        if existing:
            flash("Username already exists.", "danger")
            return render_template("admin/user_form.html", form=form, edit=False)

        pw_hash = generate_password_hash(password)
        execute_db(
            "INSERT INTO users (username, password_hash, role_id) VALUES (?, ?, ?)",
            (username, pw_hash, int(role_row["id"])),
        )
        flash(f"User '{username}' created successfully.", "success")
        return redirect(url_for("admin.users"))

    return render_template("admin/user_form.html", form=form, edit=False)


@admin_bp.route("/users/<int:user_id>/edit", methods=["GET", "POST"])
@admin_required
def user_edit(user_id: int) -> str | Response:
    """Render and process the edit-user form for *user_id*.

    Args:
        user_id: Primary key of the user to edit.

    Returns:
        Rendered form template on GET or validation error; redirect on success.
    """
    row = query_db(
        """
        SELECT u.id, u.username, r.name AS role_name, u.active
        FROM users u
        JOIN roles r ON r.id = u.role_id
        WHERE u.id = ?
        """,
        (user_id,),
        one=True,
    )
    if not row or not isinstance(row, dict):
        flash("User not found.", "danger")
        return redirect(url_for("admin.users"))

    form = UserEditForm()
    if request.method == "GET":
        form.role.data = str(row["role_name"])
        form.active.data = bool(row["active"])

    if form.validate_on_submit():
        new_password: str = str(form.password.data or "").strip()
        confirm_password: str = str(form.confirm_password.data or "").strip()
        new_role: str = str(form.role.data or "user")
        new_active: bool = bool(form.active.data)

        if new_password and new_password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("admin/user_form.html", form=form, edit=True, target_user=row)

        role_row = query_db(
            "SELECT id FROM roles WHERE name = ?", (new_role,), one=True
        )
        if not role_row or not isinstance(role_row, dict):
            flash("Invalid role.", "danger")
            return render_template("admin/user_form.html", form=form, edit=True, target_user=row)

        if new_password:
            pw_hash = generate_password_hash(new_password)
            execute_db(
                "UPDATE users SET password_hash = ?, role_id = ?, active = ? WHERE id = ?",
                (pw_hash, int(role_row["id"]), int(new_active), user_id),
            )
        else:
            execute_db(
                "UPDATE users SET role_id = ?, active = ? WHERE id = ?",
                (int(role_row["id"]), int(new_active), user_id),
            )

        flash("User updated successfully.", "success")
        return redirect(url_for("admin.users"))

    return render_template(
        "admin/user_form.html", form=form, edit=True, target_user=row
    )


@admin_bp.route("/settings", methods=["GET", "POST"])
@admin_required
def settings() -> str | Response:
    """Render and process the application settings form.

    Settings are stored as key-value pairs in the ``app_config`` table.

    Returns:
        Rendered settings template on GET; redirect on successful POST.
    """
    known_keys: list[str] = ["app_name"]

    if request.method == "POST":
        for key in known_keys:
            value: str = str(request.form.get(key, "")).strip()
            if value:
                execute_db(
                    """
                    INSERT INTO app_config (key, value, updated_at)
                    VALUES (?, ?, datetime('now'))
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                                   updated_at = excluded.updated_at
                    """,
                    (key, value),
                )
        flash("Settings saved.", "success")
        return redirect(url_for("admin.settings"))

    config_rows = query_db("SELECT key, value FROM app_config WHERE key IN (?)", (known_keys[0],))
    config_map: dict[str, str] = {}
    if isinstance(config_rows, list):
        for r in config_rows:
            if isinstance(r, dict):
                config_map[str(r["key"])] = str(r["value"])

    return render_template("admin/settings.html", config=config_map)


@admin_bp.route("/api-config", methods=["GET", "POST"])
@admin_required
def api_config() -> str | Response:
    """Render and process the backend API connection configuration form.

    Credentials are encrypted at rest using Fernet before being stored in
    the ``api_credentials`` table.

    Returns:
        Rendered config template on GET or validation error; redirect on success.
    """
    form = APIConfigForm()

    current_row = query_db(
        "SELECT base_url, timeout FROM api_credentials WHERE label = 'default' LIMIT 1",
        one=True,
    )

    if request.method == "GET" and current_row and isinstance(current_row, dict):
        form.base_url.data = str(current_row.get("base_url", ""))
        form.timeout.data = int(current_row.get("timeout", 30))

    if form.validate_on_submit():
        base_url: str = str(form.base_url.data or "").rstrip("/")
        api_username: str = str(form.api_username.data or "").strip()
        api_password: str = str(form.api_password.data or "").strip()
        timeout: int = int(form.timeout.data or 30)

        enc_user: str | None = None
        enc_pass: str | None = None
        if api_username and api_password:
            try:
                from db.crypto import encrypt

                enc_user = encrypt(api_username)
                enc_pass = encrypt(api_password)
            except RuntimeError as exc:
                flash(str(exc), "danger")
                return render_template("admin/api_config.html", form=form)

        execute_db(
            """
            INSERT INTO api_credentials
                (label, base_url, encrypted_username, encrypted_password, timeout)
            VALUES ('default', ?, ?, ?, ?)
            ON CONFLICT(label) DO UPDATE SET
                base_url           = excluded.base_url,
                encrypted_username = excluded.encrypted_username,
                encrypted_password = excluded.encrypted_password,
                timeout            = excluded.timeout
            """,
            (base_url, enc_user, enc_pass, timeout),
        )
        current_app.config["BACKEND_URL"] = base_url
        flash("API configuration saved.", "success")
        return redirect(url_for("admin.api_config"))

    return render_template("admin/api_config.html", form=form)


@admin_bp.route("/api-config/test", methods=["POST"])
@admin_required
def api_config_test() -> Response:
    """Test connectivity to the currently configured backend API.

    Calls ``GET /health`` on the backend using the stored credentials.
    Returns a sanitised result without exposing raw credential or
    connection details.

    Returns:
        JSON with ``ok`` (bool) and ``message`` (str) keys.
    """
    try:
        from services.api_client import get_api_client

        client = get_api_client()
        resp = client.health_check()
        if resp.status_code == 200:
            return jsonify({"ok": True, "message": "Connection successful."})
        return jsonify(
            {
                "ok": False,
                "message": f"Backend returned HTTP {resp.status_code}.",
            }
        )
    except Exception as exc:
        safe_message = _sanitise_connection_error(str(exc))
        return jsonify({"ok": False, "message": safe_message})


def _check_backend_health() -> bool:
    """Return whether the backend responds successfully to a health check.

    Returns:
        ``True`` when ``GET /health`` returns HTTP 200, ``False`` otherwise.
    """
    try:
        from services.api_client import get_api_client

        client = get_api_client()
        resp = client.health_check()
        return resp.status_code == 200
    except Exception:
        return False


def _sanitise_connection_error(error_message: str) -> str:
    """Produce a safe, user-facing description of a connection error.

    Strips internal hostnames, credentials, and stack trace details from the
    raw exception string before returning it to the browser.

    Args:
        error_message: Raw exception message.

    Returns:
        A short description safe for display.
    """
    if "Connection refused" in error_message or "ConnectError" in error_message:
        return "Could not connect to the backend. Check the Base URL and ensure the service is running."
    if "Timeout" in error_message or "timed out" in error_message.lower():
        return "Connection timed out. The backend may be overloaded or unreachable."
    if "401" in error_message or "Unauthorized" in error_message:
        return "Authentication failed. Check the configured credentials."
    return "Connection failed. Verify the backend URL and network accessibility."
