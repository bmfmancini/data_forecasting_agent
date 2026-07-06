"""
Route handlers for the administration blueprint.

All routes require the user to be authenticated AND to hold the ``admin``
role.  The ``admin_required`` decorator enforces this.
"""

from __future__ import annotations

import logging
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
from blueprints.admin.forms import (
    APIConfigForm,
    APIKeyCreateForm,
    UserCreateForm,
    UserEditForm,
)
from db.db import execute_db, query_db

_F = TypeVar("_F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)

_ADMIN_USERS_ENDPOINT: str = "admin.users"
_ADMIN_USER_FORM_TEMPLATE: str = "admin/user_form.html"
_ADMIN_API_CONFIG_ENDPOINT: str = "admin.api_config"
_ADMIN_API_KEYS_ENDPOINT: str = "admin.api_keys"


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

    api_key_count: int = 0
    has_bootstrap: bool = False
    try:
        from services.api_client import get_api_client

        client = get_api_client()
        resp = client.list_api_users()
        if resp.status_code == 200:
            api_users_list: list[dict[str, Any]] = resp.json()
            api_key_count = len(api_users_list)
            has_bootstrap = any(u.get("bootstrap") for u in api_users_list)
    except Exception:
        logger.exception("Failed to fetch API users for admin dashboard")

    return render_template(
        "admin/dashboard.html",
        user_count=user_count,
        backend_ok=backend_ok,
        api_key_count=api_key_count,
        has_bootstrap=has_bootstrap,
    )


@admin_bp.route("/users")
@admin_required
def users() -> str:
    """List all application users.

    Returns:
        Rendered HTML for the user management list page.
    """
    rows = query_db("""
        SELECT u.id, u.username, r.name AS role, u.active, u.created_at,
               u.must_change_password
        FROM users u
        JOIN roles r ON r.id = u.role_id
        ORDER BY u.id
        """)
    user_list: list[dict[str, Any]] = rows if isinstance(rows, list) else []
    return render_template("admin/users.html", users=user_list)


@admin_bp.route("/users/<int:user_id>/force-reset", methods=["POST"])
@admin_required
def user_force_reset(user_id: int) -> Response:
    """Force a user to change their password on next login.

    Sets the ``must_change_password`` flag for *user_id* and redirects back
    to the user list.

    Args:
        user_id: Primary key of the user to flag.

    Returns:
        A redirect response to the user management page.
    """
    row = query_db("SELECT id, username FROM users WHERE id = ?", (user_id,), one=True)
    if not row or not isinstance(row, dict):
        flash("User not found.", "danger")
        return redirect(url_for(_ADMIN_USERS_ENDPOINT))

    if int(row["id"]) == current_user.id:  # type: ignore[union-attr]
        flash("You cannot force a password reset on yourself.", "warning")
        return redirect(url_for(_ADMIN_USERS_ENDPOINT))

    execute_db(
        "UPDATE users SET must_change_password = 1 WHERE id = ?",
        (user_id,),
    )
    flash(
        f"'{row['username']}' will be required to change their password on next login.",
        "success",
    )
    return redirect(url_for(_ADMIN_USERS_ENDPOINT))


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
            return render_template(_ADMIN_USER_FORM_TEMPLATE, form=form, edit=False)

        role_row = query_db(
            "SELECT id FROM roles WHERE name = ?", (role_name,), one=True
        )
        if not role_row or not isinstance(role_row, dict):
            flash("Invalid role.", "danger")
            return render_template(_ADMIN_USER_FORM_TEMPLATE, form=form, edit=False)

        existing = query_db(
            "SELECT id FROM users WHERE username = ?", (username,), one=True
        )
        if existing:
            flash("Username already exists.", "danger")
            return render_template(_ADMIN_USER_FORM_TEMPLATE, form=form, edit=False)

        pw_hash = generate_password_hash(password)
        execute_db(
            "INSERT INTO users (username, password_hash, role_id) VALUES (?, ?, ?)",
            (username, pw_hash, int(role_row["id"])),
        )
        flash(f"User '{username}' created successfully.", "success")
        return redirect(url_for(_ADMIN_USERS_ENDPOINT))

    return render_template(_ADMIN_USER_FORM_TEMPLATE, form=form, edit=False)


def _update_user(
    user_id: int,
    new_password: str,
    new_role: str,
    new_active: bool,
    force_reset: bool,
) -> None:
    """Persist an edited user record.

    Args:
        user_id:      Primary key of the user to update.
        new_password: New plaintext password, or empty string to keep the
            existing password.
        new_role:     Role name to assign.
        new_active:   Whether the account should be active.
        force_reset:  Whether the user must change their password on next
            login.
    """
    role_row = query_db(
        "SELECT id FROM roles WHERE name = ?", (new_role,), one=True
    )
    if not role_row or not isinstance(role_row, dict):
        raise ValueError("Invalid role.")

    if new_password:
        pw_hash = generate_password_hash(new_password)
        execute_db(
            """
            UPDATE users
            SET password_hash = ?, role_id = ?, active = ?, must_change_password = ?
            WHERE id = ?
            """,
            (
                pw_hash,
                int(role_row["id"]),
                int(new_active),
                int(force_reset or 1),
                user_id,
            ),
        )
    else:
        execute_db(
            """
            UPDATE users
            SET role_id = ?, active = ?, must_change_password = ?
            WHERE id = ?
            """,
            (int(role_row["id"]), int(new_active), int(force_reset), user_id),
        )


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
        SELECT u.id, u.username, r.name AS role_name, u.active,
               u.must_change_password
        FROM users u
        JOIN roles r ON r.id = u.role_id
        WHERE u.id = ?
        """,
        (user_id,),
        one=True,
    )
    if not row or not isinstance(row, dict):
        flash("User not found.", "danger")
        return redirect(url_for(_ADMIN_USERS_ENDPOINT))

    form = UserEditForm()
    if request.method == "GET":
        form.role.data = str(row["role_name"])
        form.active.data = bool(row["active"])
        form.force_password_reset.data = bool(row.get("must_change_password", 0))

    if form.validate_on_submit():
        new_password: str = str(form.password.data or "").strip()
        confirm_password: str = str(form.confirm_password.data or "").strip()

        if new_password and new_password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template(
                _ADMIN_USER_FORM_TEMPLATE, form=form, edit=True, target_user=row
            )

        try:
            _update_user(
                user_id=user_id,
                new_password=new_password,
                new_role=str(form.role.data or "user"),
                new_active=bool(form.active.data),
                force_reset=bool(form.force_password_reset.data),
            )
        except ValueError as exc:
            flash(str(exc), "danger")
            return render_template(
                _ADMIN_USER_FORM_TEMPLATE, form=form, edit=True, target_user=row
            )

        flash("User updated successfully.", "success")
        return redirect(url_for(_ADMIN_USERS_ENDPOINT))

    return render_template(
        _ADMIN_USER_FORM_TEMPLATE, form=form, edit=True, target_user=row
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

    config_rows = query_db(
        "SELECT key, value FROM app_config WHERE key IN (?)", (known_keys[0],)
    )
    config_map: dict[str, str] = {}
    if isinstance(config_rows, list):
        for r in config_rows:
            if isinstance(r, dict):
                config_map[str(r["key"])] = str(r["value"])

    return render_template("admin/settings.html", config=config_map)


def _load_api_config_form(form: APIConfigForm) -> None:
    """Populate the API config form from the stored default row."""
    current_row = query_db(
        "SELECT base_url, timeout, verify_ssl FROM api_credentials WHERE label = 'default' LIMIT 1",
        one=True,
    )
    if current_row and isinstance(current_row, dict):
        form.base_url.data = str(current_row.get("base_url", ""))
        form.timeout.data = int(current_row.get("timeout", 30))
        form.verify_ssl.data = bool(current_row.get("verify_ssl", 0))


def _fetch_backend_auth_status() -> dict[str, Any]:
    """Return the backend's auth status, or defaults if unreachable."""
    auth_status: dict[str, Any] = {"auth_enabled": False, "has_users": False}
    try:
        from services.api_client import get_api_client

        client = get_api_client()
        resp = client.get_auth_status()
        if resp.status_code == 200:
            auth_status = resp.json()
    except Exception:
        pass  # Backend unreachable — show defaults
    return auth_status


def _encrypt_credentials(username: str, password: str) -> tuple[str, str] | tuple[None, None]:
    """Encrypt API credentials, flashing and returning None on failure."""
    try:
        from db.crypto import encrypt

        return encrypt(username), encrypt(password)
    except RuntimeError as exc:
        flash(str(exc), "danger")
        return None, None


def _save_api_credentials(
    base_url: str,
    timeout: int,
    verify_ssl: int,
    enc_user: str | None,
    enc_pass: str | None,
) -> None:
    """Upsert the default API credential row.

    When both encrypted values are supplied the row is fully updated;
    otherwise only ``base_url``, ``timeout``, and ``verify_ssl`` are
    touched, preserving any existing encrypted credentials.
    """
    if enc_user and enc_pass:
        execute_db(
            """
            INSERT INTO api_credentials
                (label, base_url, encrypted_username, encrypted_password,
                 timeout, verify_ssl)
            VALUES ('default', ?, ?, ?, ?, ?)
            ON CONFLICT(label) DO UPDATE SET
                base_url           = excluded.base_url,
                encrypted_username = excluded.encrypted_username,
                encrypted_password = excluded.encrypted_password,
                timeout            = excluded.timeout,
                verify_ssl         = excluded.verify_ssl
            """,
            (base_url, enc_user, enc_pass, timeout, verify_ssl),
        )
    else:
        execute_db(
            """
            INSERT INTO api_credentials
                (label, base_url, encrypted_username, encrypted_password,
                 timeout, verify_ssl)
            VALUES ('default', ?, NULL, NULL, ?, ?)
            ON CONFLICT(label) DO UPDATE SET
                base_url   = excluded.base_url,
                timeout    = excluded.timeout,
                verify_ssl = excluded.verify_ssl
            """,
            (base_url, timeout, verify_ssl),
        )


@admin_bp.route("/api-config", methods=["GET", "POST"])
@admin_required
def api_config() -> str | Response:
    """Render and process the backend API connection configuration form.

    Credentials are encrypted at rest using Fernet before being stored in
    the ``api_credentials`` table.

    On GET, also queries the backend's auth status so the template can
    show whether auth is enabled and offer the "Enable Authentication"
    workflow when it is off.

    Returns:
        Rendered config template on GET or validation error; redirect on success.
    """
    form = APIConfigForm()
    _load_api_config_form(form)
    auth_status = _fetch_backend_auth_status()

    if not form.validate_on_submit():
        return render_template("admin/api_config.html", form=form, auth_status=auth_status)

    base_url: str = str(form.base_url.data or "").rstrip("/")
    api_username: str = str(form.api_username.data or "").strip()
    api_password: str = str(form.api_password.data or "").strip()
    timeout: int = int(form.timeout.data or 30)
    verify_ssl: int = 1 if form.verify_ssl.data else 0

    enc_user: str | None = None
    enc_pass: str | None = None
    if api_username and api_password:
        enc_user, enc_pass = _encrypt_credentials(api_username, api_password)
        if enc_user is None or enc_pass is None:
            return render_template(
                "admin/api_config.html", form=form, auth_status=auth_status
            )

    _save_api_credentials(base_url, timeout, verify_ssl, enc_user, enc_pass)
    current_app.config["BACKEND_URL"] = base_url
    current_app.config["API_VERIFY_SSL"] = bool(verify_ssl)
    flash("API configuration saved.", "success")
    return redirect(url_for(_ADMIN_API_CONFIG_ENDPOINT))


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


@admin_bp.route("/api-config/enable-auth", methods=["POST"])
@admin_required
def api_config_enable_auth() -> Response:
    """Enable API authentication on the backend via the bootstrap endpoint.

    Reads the admin key, desired username, and API key from the form,
    calls the backend's ``POST /api-users/bootstrap`` endpoint, and
    stores the returned credentials encrypted in the frontend database.

    Returns:
        Redirect to the API config page with a flash message.
    """
    admin_key: str = str(request.form.get("admin_key", "")).strip()
    api_username: str = str(request.form.get("api_username", "")).strip()
    api_key: str = str(request.form.get("api_key", "")).strip()

    if not admin_key:
        flash("Admin key is required.", "danger")
        return redirect(url_for(_ADMIN_API_CONFIG_ENDPOINT))
    if not api_username or not api_key:
        flash("Username and API key are required.", "danger")
        return redirect(url_for(_ADMIN_API_CONFIG_ENDPOINT))

    from services.api_client import get_api_client

    client = get_api_client()
    try:
        resp = client.bootstrap_api_user(api_username, api_key, admin_key)
    except Exception as exc:
        flash(
            f"Could not connect to backend: {_sanitise_connection_error(str(exc))}",
            "danger",
        )
        return redirect(url_for(_ADMIN_API_CONFIG_ENDPOINT))

    if resp.status_code == 403:
        flash(
            "Invalid admin key. Verify the ADMIN_API_KEY in the backend .env.",
            "danger",
        )
        return redirect(url_for(_ADMIN_API_CONFIG_ENDPOINT))
    if resp.status_code == 409:
        flash(
            "API users already exist on the backend. Bootstrap is no longer available.",
            "warning",
        )
        return redirect(url_for(_ADMIN_API_CONFIG_ENDPOINT))
    if resp.status_code != 200:
        detail: str = "Unknown error."
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            logger.exception("Failed to parse bootstrap error response")
        flash(f"Bootstrap failed (HTTP {resp.status_code}): {detail}", "danger")
        return redirect(url_for(_ADMIN_API_CONFIG_ENDPOINT))

    # Success — store the credentials encrypted in the frontend DB
    try:
        from db.crypto import encrypt

        enc_user = encrypt(api_username)
        enc_pass = encrypt(api_key)
        execute_db(
            """
            UPDATE api_credentials
            SET encrypted_username = ?,
                encrypted_password = ?
            WHERE label = 'default'
            """,
            (enc_user, enc_pass),
        )
    except RuntimeError as exc:
        flash(str(exc), "danger")
        return redirect(url_for(_ADMIN_API_CONFIG_ENDPOINT))

    flash(
        "API authentication enabled successfully. "
        "Credentials stored — the frontend can now authenticate with the backend.",
        "success",
    )
    return redirect(url_for(_ADMIN_API_CONFIG_ENDPOINT))


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


# ── API Key User Management ───────────────────────────────────────────────────


@admin_bp.route("/api-keys")
@admin_required
def api_keys() -> str | Response:
    """List all API key users from the backend.

    Returns:
        Rendered HTML for the API key management list page, or a redirect
        with an error message when the backend is unreachable.
    """
    from services.api_client import get_api_client

    client = get_api_client()
    try:
        resp = client.list_api_users()
        if resp.status_code == 200:
            api_users: list[dict[str, Any]] = resp.json()
        else:
            flash(
                f"Failed to retrieve API users (HTTP {resp.status_code}).",
                "danger",
            )
            api_users = []
    except Exception as exc:
        flash(
            f"Could not connect to backend: {_sanitise_connection_error(str(exc))}",
            "danger",
        )
        api_users = []

    has_bootstrap: bool = any(u.get("bootstrap") for u in api_users)
    return render_template(
        "admin/api_keys.html",
        api_users=api_users,
        has_bootstrap=has_bootstrap,
    )


@admin_bp.route("/api-keys/new", methods=["GET", "POST"])
@admin_required
def api_key_new() -> str | Response:
    """Render and process the create-API-key-user form.

    On success, displays the plaintext API key once.

    Returns:
        Rendered form template on GET or validation error; rendered
        key-display template on successful creation.
    """
    form = APIKeyCreateForm()

    if form.validate_on_submit():
        username: str = str(form.username.data or "").strip()
        description: str = str(form.description.data or "").strip()
        is_admin: bool = bool(form.is_admin.data)

        from services.api_client import get_api_client

        client = get_api_client()
        try:
            resp = client.create_api_user(username, description, is_admin)
            if resp.status_code == 201:
                data: dict[str, Any] = resp.json()
                return render_template(
                    "admin/api_key_created.html",
                    username=username,
                    api_key=data.get("api_key", ""),
                )
            if resp.status_code == 409:
                flash(resp.json().get("detail", "Username already exists."), "danger")
            else:
                flash(
                    f"Failed to create API user (HTTP {resp.status_code}).",
                    "danger",
                )
        except Exception as exc:
            flash(
                f"Could not connect to backend: {_sanitise_connection_error(str(exc))}",
                "danger",
            )

    return render_template("admin/api_key_create.html", form=form)


@admin_bp.route("/api-keys/<int:user_id>/rotate", methods=["POST"])
@admin_required
def api_key_rotate(user_id: int) -> Response:
    """Rotate an API user's key.

    Args:
        user_id: Primary key of the API user.

    Returns:
        Redirect to the API keys list with the new key flashed once, or
        a redirect with an error message on failure.
    """
    from services.api_client import get_api_client

    client = get_api_client()
    try:
        # Check if this is the active user — warn if so
        active_username: str | None = client._api_username
        is_active_user: bool = False
        if active_username:
            list_resp = client.list_api_users()
            if list_resp.status_code == 200:
                target = next(
                    (u for u in list_resp.json() if u.get("id") == user_id), None
                )
                is_active_user = bool(
                    target and target.get("username") == active_username
                )

        resp = client.rotate_api_key(user_id)
        if resp.status_code == 200:
            data: dict[str, Any] = resp.json()
            new_key: str = data.get("api_key", "")
            if is_active_user:
                flash(
                    f"API key rotated successfully. New key (copy now — shown once): "
                    f"{new_key}",
                    "success",
                )
                # Auto-update the frontend's stored credentials so the
                # connection to the backend is not broken.
                from db.crypto import encrypt

                execute_db(
                    "UPDATE api_credentials"
                    " SET encrypted_username = ?, encrypted_password = ?"
                    " WHERE label = 'default'",
                    (encrypt(active_username), encrypt(new_key)),
                )
                flash(
                    "The frontend's API credentials have been updated automatically.",
                    "success",
                )
            else:
                flash(
                    f"API key rotated successfully. New key (copy now — shown once): "
                    f"{new_key}",
                    "success",
                )
        else:
            flash(
                f"Failed to rotate key (HTTP {resp.status_code}).",
                "danger",
            )
    except Exception as exc:
        flash(
            f"Could not connect to backend: {_sanitise_connection_error(str(exc))}",
            "danger",
        )

    return redirect(url_for(_ADMIN_API_KEYS_ENDPOINT))


@admin_bp.route("/api-keys/<int:user_id>/toggle", methods=["POST"])
@admin_required
def api_key_toggle(user_id: int) -> Response:
    """Enable or disable an API user.

    Args:
        user_id: Primary key of the API user.

    Returns:
        Redirect to the API keys list with a flash message.
    """
    from services.api_client import get_api_client

    enabled: bool = request.form.get("enabled", "").lower() in ("true", "1", "on")
    client = get_api_client()
    try:
        resp = client.toggle_api_user(user_id, enabled)
        if resp.status_code == 200:
            action: str = "enabled" if enabled else "disabled"
            flash(f"API user {action} successfully.", "success")
        else:
            flash(
                f"Failed to toggle user (HTTP {resp.status_code}).",
                "danger",
            )
    except Exception as exc:
        flash(
            f"Could not connect to backend: {_sanitise_connection_error(str(exc))}",
            "danger",
        )

    return redirect(url_for(_ADMIN_API_KEYS_ENDPOINT))


@admin_bp.route("/api-keys/<int:user_id>/admin", methods=["POST"])
@admin_required
def api_key_set_admin(user_id: int) -> Response:
    """Promote or demote an API user.

    Args:
        user_id: Primary key of the API user.

    Returns:
        Redirect to the API keys list with a flash message.
    """
    from services.api_client import get_api_client

    is_admin: bool = request.form.get("is_admin", "").lower() in ("true", "1", "on")
    client = get_api_client()
    try:
        resp = client.set_api_user_admin(user_id, is_admin)
        if resp.status_code == 200:
            action: str = "promoted to admin" if is_admin else "demoted to regular user"
            flash(f"API user {action} successfully.", "success")
        else:
            flash(
                f"Failed to update admin status (HTTP {resp.status_code}).",
                "danger",
            )
    except Exception as exc:
        flash(
            f"Could not connect to backend: {_sanitise_connection_error(str(exc))}",
            "danger",
        )

    return redirect(url_for(_ADMIN_API_KEYS_ENDPOINT))


@admin_bp.route("/api-keys/<int:user_id>/delete", methods=["POST"])
@admin_required
def api_key_delete(user_id: int) -> Response:
    """Permanently delete an API user.

    Prevents deletion of the API user whose credentials are currently
    stored in the frontend's ``api_credentials`` table — deleting that
    user would cause all backend requests to fail with 401 Unauthorized.

    Args:
        user_id: Primary key of the API user to delete.

    Returns:
        Redirect to the API keys list with a flash message.
    """
    from services.api_client import get_api_client

    # ── Guard: don't allow deleting the user the frontend is actively using ──
    try:
        client = get_api_client()
        current_username: str | None = client._api_username

        if current_username:
            list_resp = client.list_api_users()
            if list_resp.status_code == 200:
                users_list: list[dict[str, Any]] = list_resp.json()
                target_user = next(
                    (u for u in users_list if u.get("id") == user_id), None
                )
                if target_user and target_user.get("username") == current_username:
                    flash(
                        f"Cannot delete the API user '{current_username}' — it is "
                        "currently used by this frontend for backend authentication. "
                        "Create a replacement user, update the API Configuration "
                        "with the new credentials, then delete this account.",
                        "danger",
                    )
                    return redirect(url_for(_ADMIN_API_KEYS_ENDPOINT))
    except Exception:
        logger.exception("Failed to check active API user before deletion")

    client = get_api_client()
    try:
        resp = client.delete_api_user(user_id)
        if resp.status_code == 204:
            flash("API user deleted successfully.", "success")
        else:
            flash(
                f"Failed to delete user (HTTP {resp.status_code}).",
                "danger",
            )
    except Exception as exc:
        flash(
            f"Could not connect to backend: {_sanitise_connection_error(str(exc))}",
            "danger",
        )

    return redirect(url_for(_ADMIN_API_KEYS_ENDPOINT))
