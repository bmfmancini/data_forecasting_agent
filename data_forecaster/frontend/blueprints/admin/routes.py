"""
Route handlers for the administration blueprint.

All routes require the user to be authenticated AND to hold the ``admin``
role.  The ``admin_required`` decorator enforces this.
"""

from __future__ import annotations

import logging
import sqlite3
from functools import wraps
from typing import Any, Callable, TypeVar

import requests
from cryptography.fernet import InvalidToken
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
from db.crypto import decrypt, encrypt
from db.db import execute_db, query_db
from services.api_client import BackendAPIClient, get_api_client
from services.connection_errors import sanitize_connection_error
from services.report_service import (
    delete_all_reports_for_admin,
    delete_report_for_admin,
    list_report_owners,
    list_reports_for_user,
)

_F = TypeVar("_F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)

_ADMIN_USERS_ENDPOINT: str = "admin.users"
_ADMIN_USER_FORM_TEMPLATE: str = "admin/user_form.html"
_ADMIN_API_CONFIG_ENDPOINT: str = "admin.api_config"
_ADMIN_API_KEYS_ENDPOINT: str = "admin.api_keys"
_ADMIN_JOB_QUEUE_ENDPOINT: str = "admin.job_queue"
_API_KEY_PLACEHOLDER: str = "******"

_RETENTION_OPTIONS: dict[str, tuple[int | None, bool]] = {
    "1": (1, True),
    "7": (7, True),
    "14": (14, True),
    "30": (30, True),
    "90": (90, True),
    "180": (180, True),
    "indefinite": (None, True),
    "disabled": (None, False),
}


def _save_app_config_values(values: dict[str, str]) -> None:
    """Persist application config values after all validation succeeds."""
    for key, value in values.items():
        execute_db(
            """
            INSERT INTO app_config (key, value, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                           updated_at = excluded.updated_at
            """,
            (key, value),
        )


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
        client = get_api_client()
        resp = client.list_api_users()
        if resp.status_code == 200:
            api_users_list: list[dict[str, Any]] = resp.json()
            api_key_count = len(api_users_list)
            has_bootstrap = any(u.get("bootstrap") for u in api_users_list)
    except (requests.RequestException, ValueError):
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
        """
        UPDATE users
        SET must_change_password = 1, session_version = session_version + 1
        WHERE id = ?
        """,
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
            """
            INSERT INTO users
                (username, password_hash, role_id, must_change_password)
            VALUES (?, ?, ?, 1)
            """,
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
    role_row = query_db("SELECT id FROM roles WHERE name = ?", (new_role,), one=True)
    if not role_row or not isinstance(role_row, dict):
        raise ValueError("Invalid role.")

    if new_password:
        pw_hash = generate_password_hash(new_password)
        execute_db(
            """
            UPDATE users
            SET password_hash = ?, role_id = ?, active = ?,
                must_change_password = ?,
                session_version = session_version + 1
            WHERE id = ?
            """,
            (
                pw_hash,
                int(role_row["id"]),
                int(new_active),
                int(force_reset),
                user_id,
            ),
        )
    else:
        execute_db(
            """
            UPDATE users
            SET role_id = ?, active = ?, must_change_password = ?,
                session_version = CASE
                    WHEN active != ? OR ? = 1 THEN session_version + 1
                    ELSE session_version
                END
            WHERE id = ?
            """,
            (
                int(role_row["id"]),
                int(new_active),
                int(force_reset),
                int(new_active),
                int(force_reset),
                user_id,
            ),
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
    known_keys: list[str] = ["app_name", "max_reports_per_user", "max_upload_mb"]

    if request.method == "POST":
        try:
            if int(request.form.get("max_reports_per_user", "")) < 1:
                raise ValueError
        except ValueError:
            flash("Enter a report limit of at least 1.", "danger")
            return redirect(url_for("admin.settings"))
        try:
            max_upload_mb = int(request.form.get("max_upload_mb", ""))
            if max_upload_mb < 1 or max_upload_mb > 256:
                raise ValueError
        except ValueError:
            flash("Enter an upload limit from 1 to 256 MB.", "danger")
            return redirect(url_for("admin.settings"))
        app_config_updates = {
            key: value
            for key in known_keys
            if (value := str(request.form.get(key, "")).strip())
        }
        try:
            max_running_jobs = int(request.form.get("max_running_jobs_per_user", ""))
            max_queued_jobs = int(request.form.get("max_queued_jobs_per_user", ""))
            retention_option = str(request.form.get("job_retention", "30"))
            retention_days, cleanup_enabled = _RETENTION_OPTIONS[retention_option]
            if max_running_jobs < 1:
                raise ValueError
            if max_queued_jobs < 1:
                raise ValueError
        except (KeyError, ValueError):
            flash(
                "Enter job limits of at least 1 and select a valid retention option.",
                "danger",
            )
            return redirect(url_for("admin.settings"))
        try:
            response = get_api_client().update_job_settings(
                {
                    "max_running_jobs_per_user": max_running_jobs,
                    "max_queued_jobs_per_user": max_queued_jobs,
                    "retention_days": retention_days,
                    "cleanup_enabled": cleanup_enabled,
                }
            )
            if response.status_code != 200:
                flash("Forecast job settings could not be saved.", "danger")
                return redirect(url_for("admin.settings"))
        except requests.RequestException:
            logger.exception("Failed to save forecast job settings")
            flash(
                "The backend is unavailable; forecast job settings were not saved.",
                "danger",
            )
            return redirect(url_for("admin.settings"))
        _save_app_config_values(app_config_updates)
        current_app.config["MAX_CONTENT_LENGTH"] = max_upload_mb * 1024 * 1024
        flash("Settings saved.", "success")
        return redirect(url_for("admin.settings"))

    config_rows = query_db(
        "SELECT key, value FROM app_config WHERE key IN (?, ?, ?)",
        (known_keys[0], known_keys[1], known_keys[2]),
    )
    config_map: dict[str, str] = {}
    if isinstance(config_rows, list):
        for r in config_rows:
            if isinstance(r, dict):
                config_map[str(r["key"])] = str(r["value"])

    job_settings = _load_forecast_job_settings()
    return render_template(
        "admin/settings.html",
        config=config_map,
        job_settings=job_settings,
        retention_option=_retention_option(job_settings),
    )


@admin_bp.route("/reports")
@admin_required
def report_management() -> str:
    """List users who currently own one or more saved reports."""
    return render_template("admin/report_management.html", owners=list_report_owners())


@admin_bp.route("/reports/users/<int:user_id>")
@admin_required
def user_reports(user_id: int) -> str | Response:
    """List every saved report owned by the selected application user."""
    target_user = query_db(
        "SELECT id, username FROM users WHERE id = ?", (user_id,), one=True
    )
    if not isinstance(target_user, dict):
        flash("User not found.", "danger")
        return redirect(url_for("admin.report_management"))
    return render_template(
        "admin/user_reports.html",
        target_user=target_user,
        reports=list_reports_for_user(user_id),
    )


@admin_bp.route("/reports/<int:report_id>/delete", methods=["POST"])
@admin_required
def report_delete(report_id: int) -> Response:
    """Delete one saved report as an administrator."""
    if not delete_report_for_admin(report_id):
        flash("Report not found.", "danger")
        return redirect(url_for("admin.report_management"))
    flash("Report deleted.", "success")
    return redirect(url_for("admin.report_management"))


@admin_bp.route("/reports/users/<int:user_id>/delete-all", methods=["POST"])
@admin_required
def user_reports_delete_all(user_id: int) -> Response:
    """Delete every saved report owned by one application user."""
    target_user = query_db(
        "SELECT id, username FROM users WHERE id = ?", (user_id,), one=True
    )
    if not isinstance(target_user, dict):
        flash("User not found.", "danger")
        return redirect(url_for("admin.report_management"))
    deleted_count = delete_all_reports_for_admin(user_id)
    flash(
        f"Deleted {deleted_count} report(s) for '{target_user['username']}'.",
        "success",
    )
    return redirect(url_for("admin.report_management"))


def _load_forecast_job_settings() -> dict[str, Any]:
    """Load backend-managed forecast job settings with safe defaults."""
    defaults: dict[str, Any] = {
        "max_running_jobs_per_user": 1,
        "retention_days": 30,
        "cleanup_enabled": True,
    }
    try:
        response = get_api_client().get_job_settings()
        if response.status_code == 200:
            return response.json()
    except (requests.RequestException, ValueError):
        logger.exception("Failed to load forecast job settings")
    return defaults


def _retention_option(job_settings: dict[str, Any]) -> str:
    """Map backend retention fields to the settings form option value."""
    if not job_settings.get("cleanup_enabled", True):
        return "disabled"
    retention_days = job_settings.get("retention_days")
    return "indefinite" if retention_days is None else str(retention_days)


@admin_bp.route("/job-queue")
@admin_required
def job_queue() -> str:
    """Render the 25 most recent forecast jobs for administrators."""
    jobs: list[dict[str, Any]] = []
    queue_error: str | None = None
    try:
        response = get_api_client().list_recent_jobs()
        if response.status_code == 200:
            jobs = response.json()
        else:
            queue_error = "The backend could not provide job history."
    except (requests.RequestException, ValueError):
        logger.exception("Failed to fetch forecast job queue")
        queue_error = "The backend is unavailable. Try again once it is online."
    return render_template("admin/job_queue.html", jobs=jobs, queue_error=queue_error)


@admin_bp.route("/job-queue/clear-terminal", methods=["POST"])
@admin_required
def clear_terminal_jobs() -> Response:
    """Clear all completed and failed jobs through the protected backend API."""
    try:
        response = get_api_client().clear_terminal_jobs()
        if response.status_code == 200:
            deleted_count = int(response.json().get("deleted_count", 0))
            flash(f"Cleared {deleted_count} completed or failed job(s).", "success")
        else:
            flash("Completed and failed jobs could not be cleared.", "danger")
    except (requests.RequestException, ValueError):
        logger.exception("Failed to clear terminal forecast jobs")
        flash("The backend is unavailable; terminal jobs were not cleared.", "danger")
    return redirect(url_for(_ADMIN_JOB_QUEUE_ENDPOINT))


def _load_current_api_config() -> dict[str, Any] | None:
    """Return the saved API config summary without exposing the API key."""
    current_row = query_db(
        """
        SELECT base_url, timeout, verify_ssl, encrypted_username,
               encrypted_password
        FROM api_credentials WHERE label = 'default' LIMIT 1
        """,
        one=True,
    )
    if not current_row or not isinstance(current_row, dict):
        return None

    username = ""
    enc_user = current_row.get("encrypted_username")
    if enc_user:
        try:
            username = decrypt(str(enc_user))
        except (InvalidToken, RuntimeError, ValueError):
            username = ""

    return {
        "base_url": str(current_row.get("base_url", "")),
        "username": username,
        "timeout": int(current_row.get("timeout", 30)),
        "verify_ssl": bool(current_row.get("verify_ssl", 0)),
        "has_key": bool(current_row.get("encrypted_password")),
    }


def _load_api_config_form(
    form: APIConfigForm, current_config: dict[str, Any] | None = None
) -> None:
    """Populate the API config form from the stored default row."""
    current_config = current_config or _load_current_api_config()
    if current_config:
        form.base_url.data = str(current_config.get("base_url", ""))
        form.api_username.data = str(current_config.get("username", ""))
        form.timeout.data = int(current_config.get("timeout", 30))
        form.verify_ssl.data = bool(current_config.get("verify_ssl", False))
        form.api_password.data = ""


def _render_api_config(
    form: APIConfigForm,
    auth_status: dict[str, Any],
    status_code: int = 200,
) -> str | tuple[str, int]:
    """Render API Config with the current saved summary."""
    current_config = _load_current_api_config()
    rendered = render_template(
        "admin/api_config.html",
        form=form,
        auth_status=auth_status,
        current_api_config=current_config,
        api_key_placeholder=_API_KEY_PLACEHOLDER,
    )
    return rendered if status_code == 200 else (rendered, status_code)


def _fetch_backend_auth_status() -> dict[str, Any]:
    """Return the backend's auth status, or defaults if unreachable."""
    auth_status: dict[str, Any] = {"auth_enabled": False, "has_users": False}
    try:
        client = get_api_client()
        resp = client.get_auth_status()
        if resp.status_code == 200:
            auth_status = resp.json()
    except (requests.RequestException, ValueError):
        pass  # Backend unreachable — show defaults
    return auth_status


def _encrypt_credentials(
    username: str, password: str
) -> tuple[str, str] | tuple[None, None]:
    """Encrypt API credentials, flashing and returning None on failure."""
    try:
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
    preserve_existing_key: bool = False,
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
    elif enc_user and preserve_existing_key:
        execute_db(
            """
            INSERT INTO api_credentials
                (label, base_url, encrypted_username, encrypted_password,
                 timeout, verify_ssl)
            VALUES ('default', ?, ?, NULL, ?, ?)
            ON CONFLICT(label) DO UPDATE SET
                base_url           = excluded.base_url,
                encrypted_username = excluded.encrypted_username,
                timeout            = excluded.timeout,
                verify_ssl         = excluded.verify_ssl
            """,
            (base_url, enc_user, timeout, verify_ssl),
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


def _client_from_api_config_form() -> BackendAPIClient | None:
    """Build a temporary API client from posted API Config form values."""
    base_url = str(request.form.get("base_url", "")).strip().rstrip("/")
    if not base_url:
        return None

    username = str(request.form.get("api_username", "")).strip()
    api_key = str(request.form.get("api_password", "")).strip()
    verify_ssl = request.form.get("verify_ssl") in {"y", "on", "true", "1"}

    if api_key and not username:
        return None

    if username and not api_key:
        row = query_db(
            """
            SELECT encrypted_password
            FROM api_credentials
            WHERE label = 'default'
            LIMIT 1
            """,
            one=True,
        )
        if not row or not isinstance(row, dict) or not row.get("encrypted_password"):
            return None
        try:
            api_key = decrypt(str(row["encrypted_password"]))
        except (InvalidToken, RuntimeError, ValueError):
            return None

    return BackendAPIClient(
        base_url=base_url,
        api_username=username or None,
        api_key=api_key or None,
        verify=verify_ssl,
    )


def _client_for_api_config_test() -> BackendAPIClient:
    """Resolve the client requested by an API Config connection test.

    The summary-row button tests the saved configuration exactly as the rest
    of the application uses it. The edit-form button tests posted changes,
    falling back to the saved client only when no usable form URL was posted.
    """
    if request.form.get("use_saved_credentials") == "1":
        return get_api_client()
    return _client_from_api_config_form() or get_api_client()


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
    auth_status = _fetch_backend_auth_status()

    if request.method == "GET":
        current_config = _load_current_api_config()
        _load_api_config_form(form, current_config)
        return _render_api_config(form, auth_status)

    if not form.validate_on_submit():
        return _render_api_config(form, auth_status)

    base_url: str = str(form.base_url.data or "").rstrip("/")
    api_username: str = str(form.api_username.data or "").strip()
    api_password: str = str(form.api_password.data or "").strip()
    timeout: int = int(form.timeout.data or 30)
    verify_ssl: int = 1 if form.verify_ssl.data else 0
    current_config = _load_current_api_config()
    has_existing_key = bool(current_config and current_config.get("has_key"))

    if api_password == _API_KEY_PLACEHOLDER:
        api_password = ""

    if api_password and not api_username:
        flash("Enter an API username when providing a new API key.", "danger")
        return _render_api_config(form, auth_status)

    if api_username and not api_password and not has_existing_key:
        flash("Enter an API key for the configured API username.", "danger")
        return _render_api_config(form, auth_status)

    enc_user: str | None = None
    enc_pass: str | None = None
    preserve_existing_key = False
    if api_username and api_password:
        enc_user, enc_pass = _encrypt_credentials(api_username, api_password)
        if enc_user is None or enc_pass is None:
            return _render_api_config(form, auth_status)
    elif api_username and has_existing_key:
        try:
            enc_user = encrypt(api_username)
        except RuntimeError as exc:
            flash(str(exc), "danger")
            return _render_api_config(form, auth_status)
        preserve_existing_key = True
    elif not api_username and has_existing_key:
        preserve_existing_key = True

    _save_api_credentials(
        base_url,
        timeout,
        verify_ssl,
        enc_user,
        enc_pass,
        preserve_existing_key,
    )
    current_app.config["BACKEND_URL"] = base_url
    current_app.config["API_VERIFY_SSL"] = bool(verify_ssl)
    flash("API configuration saved.", "success")
    return redirect(url_for(_ADMIN_API_CONFIG_ENDPOINT))


@admin_bp.route("/api-config/test", methods=["POST"])
@admin_required
def api_config_test() -> Response:
    """Test connectivity to the currently configured backend API.

    Calls ``GET /auth-check`` using the saved credentials when requested by
    the summary-row button. Otherwise it tests the values currently typed in
    the form, using the saved API key when the key field is blank.
    Returns a sanitised result without exposing raw credential or
    connection details.

    Returns:
        JSON with ``ok`` (bool) and ``message`` (str) keys.
    """
    try:
        client = _client_for_api_config_test()
        resp = client.auth_check()
        if resp.status_code == 200:
            data = resp.json()
            if data.get("authenticated"):
                return jsonify(
                    {
                        "ok": True,
                        "message": "Connection and API credentials successful.",
                    }
                )
            return jsonify(
                {
                    "ok": True,
                    "message": "Connection successful; backend auth is disabled.",
                }
            )
        if resp.status_code == 401:
            return jsonify(
                {
                    "ok": False,
                    "message": "Authentication failed. Check the configured credentials.",
                }
            )
        return jsonify(
            {
                "ok": False,
                "message": f"Backend returned HTTP {resp.status_code}.",
            }
        )
    except requests.RequestException as exc:
        safe_message = _sanitise_connection_error(str(exc))
        return jsonify({"ok": False, "message": safe_message})
    except ValueError:
        return jsonify(
            {"ok": False, "message": "Backend returned an invalid response."}
        )


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

    client = get_api_client()
    try:
        resp = client.bootstrap_api_user(api_username, api_key, admin_key)
    except requests.RequestException as exc:
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
        except ValueError:
            logger.exception("Failed to parse bootstrap error response")
        flash(f"Bootstrap failed (HTTP {resp.status_code}): {detail}", "danger")
        return redirect(url_for(_ADMIN_API_CONFIG_ENDPOINT))

    # Success — store the credentials encrypted in the frontend DB
    try:
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
        client = get_api_client()
        resp = client.health_check()
        return resp.status_code == 200
    except requests.RequestException:
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
    return sanitize_connection_error(error_message)


# ── API Key User Management ───────────────────────────────────────────────────


@admin_bp.route("/api-keys")
@admin_required
def api_keys() -> str | Response:
    """List all API key users from the backend.

    Returns:
        Rendered HTML for the API key management list page, or a redirect
        with an error message when the backend is unreachable.
    """
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
    except (requests.RequestException, ValueError) as exc:
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
        except (requests.RequestException, ValueError) as exc:
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
                # connection to the backend is not broken.  Isolate this
                # in its own try/except so a credential-update failure does
                # not mask the successful backend rotation.
                try:
                    execute_db(
                        "UPDATE api_credentials"
                        " SET encrypted_username = ?, encrypted_password = ?"
                        " WHERE label = 'default'",
                        (encrypt(active_username), encrypt(new_key)),
                    )
                    flash(
                        "The frontend's API credentials have been updated "
                        "automatically.",
                        "success",
                    )
                except (RuntimeError, sqlite3.DatabaseError) as cred_exc:
                    logger.exception(
                        "Failed to update frontend credentials after rotation"
                    )
                    flash(
                        f"Key rotated but frontend credentials update failed: "
                        f"{cred_exc}. Manually update the stored frontend "
                        f"credentials to match the new key.",
                        "warning",
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
    except (requests.RequestException, ValueError) as exc:
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
    except (requests.RequestException, ValueError) as exc:
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
    is_admin: bool = request.form.get("is_admin", "").lower() in ("true", "1", "on")
    client = get_api_client()
    try:
        # Guard: don't allow demoting the user the frontend is actively
        # using for backend authentication — that would lock the admin
        # panel out of backend user-management calls.
        if not is_admin:
            active_username: str | None = client._api_username
            if active_username:
                list_resp = client.list_api_users()
                if list_resp.status_code == 200:
                    target = next(
                        (u for u in list_resp.json() if u.get("id") == user_id),
                        None,
                    )
                    if target and target.get("username") == active_username:
                        flash(
                            f"Cannot demote the API user '{active_username}' — it is "
                            "currently used by this frontend for backend "
                            "authentication.",
                            "danger",
                        )
                        return redirect(url_for(_ADMIN_API_KEYS_ENDPOINT))

        resp = client.set_api_user_admin(user_id, is_admin)
        if resp.status_code == 200:
            action: str = "promoted to admin" if is_admin else "demoted to regular user"
            flash(f"API user {action} successfully.", "success")
        else:
            flash(
                f"Failed to update admin status (HTTP {resp.status_code}).",
                "danger",
            )
    except (requests.RequestException, ValueError) as exc:
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
    except (requests.RequestException, ValueError):
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
    except requests.RequestException as exc:
        flash(
            f"Could not connect to backend: {_sanitise_connection_error(str(exc))}",
            "danger",
        )

    return redirect(url_for(_ADMIN_API_KEYS_ENDPOINT))
