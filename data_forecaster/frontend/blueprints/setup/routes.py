"""
Route handlers for the first-run setup wizard blueprint.

The wizard runs while the backend reports ``setup_complete = false`` and
walks the operator through: backend connection → LLM provider → enable
API auth → model selection → create first admin → done.  Progress is
tracked in the Flask session; every step re-validates backend
reachability and flashes errors on failure.

Security notes:
- The LLM API key collected in step 2 is forwarded to the backend and is
  never persisted by the frontend (no DB write, no logging).
- The bootstrap admin API user doubles as the frontend↔backend service
  account — after ``POST /setup/bootstrap`` succeeds, its credentials are
  stored (Fernet-encrypted) in ``api_credentials`` so the frontend can
  talk to the now-auth-enabled backend.  This avoids a second
  ``/api-users/create`` round-trip during first-run setup.
"""

from __future__ import annotations

import logging
import secrets
from typing import Any

import requests
from flask import (
    current_app,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.wrappers import Response

from blueprints.decorators import get_backend_setup_status
from blueprints.setup import setup_bp
from blueprints.setup.forms import (
    AdminCreateForm,
    BackendConnectionForm,
    EnableAuthForm,
    LLMProviderForm,
    ModelsForm,
)
from db.crypto import encrypt
from services.api_client import (
    BackendAPIClient,
    get_api_client,
    resolve_backend_connection,
)
from services.connection_errors import sanitize_connection_error
from services.credentials_service import save_api_credentials

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT: int = 30
_AUTH_LOGIN_ENDPOINT: str = "auth.login"
_TEMPLATE_BACKEND: str = "setup/backend.html"
_TEMPLATE_LLM: str = "setup/llm.html"
_TEMPLATE_AUTH: str = "setup/auth.html"
_TEMPLATE_MODELS: str = "setup/models.html"
_TEMPLATE_ADMIN: str = "setup/admin.html"
_TEMPLATE_DONE: str = "setup/done.html"


def _setup_complete() -> bool:
    """Return whether the backend reports setup as complete."""
    return bool(get_backend_setup_status().get("setup_complete"))


def _response_detail(resp: requests.Response) -> str:
    """Extract the ``detail`` message from an error response."""
    try:
        detail: str = resp.json().get("detail", "Unknown error.")
        return detail
    except ValueError:
        return "Unknown error."


def _render(template: str, status_code: int = 200, **context: Any) -> str | tuple[str, int]:
    """Render a wizard template with an optional non-200 status code."""
    rendered = render_template(template, **context)
    return rendered if status_code == 200 else (rendered, status_code)


@setup_bp.route("/")
def index() -> Response:
    """Redirect to the first wizard step, or to login when done."""
    if _setup_complete():
        return redirect(url_for(_AUTH_LOGIN_ENDPOINT))
    return redirect(url_for("setup.backend"))


@setup_bp.route("/backend", methods=["GET", "POST"])
def backend() -> str | tuple[str, int] | Response:
    """Step 1 — configure and verify the backend connection.

    Saves the verified backend URL (and TLS verification preference) to
    the ``api_credentials`` table without touching stored credentials.
    """
    if _setup_complete():
        return redirect(url_for(_AUTH_LOGIN_ENDPOINT))

    form = BackendConnectionForm()
    if request.method == "GET":
        # Resolve from the DB first so every worker pre-fills the same
        # URL (a different worker may have verified the connection).
        base_url, verify_ssl = resolve_backend_connection()
        form.base_url.data = base_url
        form.verify_ssl.data = verify_ssl
        return _render(_TEMPLATE_BACKEND, form=form)

    if not form.validate_on_submit():
        return _render(_TEMPLATE_BACKEND, 400, form=form)

    base_url: str = str(form.base_url.data or "").rstrip("/")
    verify_ssl: bool = bool(form.verify_ssl.data)

    probe = BackendAPIClient(base_url=base_url, verify=verify_ssl)
    try:
        resp = probe.get_setup_status()
    except requests.RequestException as exc:
        flash(
            f"Could not connect to backend: "
            f"{sanitize_connection_error(str(exc))}",
            "danger",
        )
        return _render(_TEMPLATE_BACKEND, 200, form=form)

    if resp.status_code != 200:
        flash(f"Backend responded with HTTP {resp.status_code}.", "danger")
        return _render(_TEMPLATE_BACKEND, 200, form=form)

    save_api_credentials(base_url, _DEFAULT_TIMEOUT, int(verify_ssl), None, None)
    current_app.config["BACKEND_URL"] = base_url
    current_app.config["API_VERIFY_SSL"] = verify_ssl
    session["setup_backend_ok"] = True
    flash("Backend connection verified.", "success")
    return redirect(url_for("setup.llm"))


def _prefill_llm_form(form: LLMProviderForm) -> None:
    """Populate the LLM form from the backend's current configuration.

    Args:
        form: The form instance to populate in place.
    """
    try:
        resp = get_api_client().get_llm_config()
        if resp.status_code != 200:
            return
        config: dict[str, Any] = resp.json()
    except (requests.RequestException, ValueError):
        flash("Backend unreachable — verify the connection step.", "warning")
        return
    form.provider.data = str(config.get("provider", "gemini"))
    form.model.data = str(config.get("model", ""))
    form.base_url.data = str(config.get("base_url") or "")
    form.temperature.data = float(config.get("temperature", 0.1))


def _submit_llm_config(form: LLMProviderForm) -> str | tuple[str, int] | None:
    """Validate and then forward the LLM configuration to the backend.

    Args:
        form: The validated LLM provider form.

    Returns:
        ``None`` on success, otherwise a rendered error response.
    """
    payload: dict[str, Any] = {
        "provider": str(form.provider.data),
        "model": str(form.model.data or "").strip(),
        "temperature": float(form.temperature.data or 0.1),
    }
    base_url: str = str(form.base_url.data or "").strip()
    if base_url:
        payload["base_url"] = base_url
    api_key: str = str(form.api_key.data or "").strip()
    if api_key:
        payload["api_key"] = api_key

    client = get_api_client()
    try:
        test_resp = client.test_llm_config(payload)
    except requests.RequestException as exc:
        flash(
            f"Could not connect to backend: "
            f"{sanitize_connection_error(str(exc))}",
            "danger",
        )
        return _render(_TEMPLATE_LLM, 200, form=form)

    if test_resp.status_code != 200:
        flash(
            f"Could not test LLM configuration (HTTP {test_resp.status_code}): "
            f"{_response_detail(test_resp)}",
            "danger",
        )
        return _render(_TEMPLATE_LLM, 200, form=form)
    try:
        test_result: dict[str, Any] = test_resp.json()
    except ValueError:
        flash("Backend returned an invalid LLM test response.", "danger")
        return _render(_TEMPLATE_LLM, 200, form=form)
    if not test_result.get("ok"):
        flash(
            str(test_result.get("message", "LLM connection test failed.")),
            "danger",
        )
        return _render(
            _TEMPLATE_LLM,
            200,
            form=form,
            llm_test_result=test_result,
        )

    try:
        resp = client.put_llm_config(payload)
    except requests.RequestException as exc:
        flash(
            f"LLM test passed, but the configuration could not be saved: "
            f"{sanitize_connection_error(str(exc))}",
            "danger",
        )
        return _render(_TEMPLATE_LLM, 200, form=form)
    if resp.status_code != 200:
        flash(
            f"LLM test passed, but the configuration could not be saved "
            f"(HTTP {resp.status_code}): {_response_detail(resp)}",
            "danger",
        )
        return _render(_TEMPLATE_LLM, 200, form=form)
    return None


@setup_bp.route("/llm", methods=["GET", "POST"])
def llm() -> str | tuple[str, int] | Response:
    """Step 2 — configure the LLM provider on the backend.

    The API key is forwarded to the backend only; the frontend never
    persists it.  Backend auth is still off at this point, so the
    unauthenticated ``PUT /config/llm`` succeeds.
    """
    if _setup_complete():
        return redirect(url_for(_AUTH_LOGIN_ENDPOINT))
    if not session.get("setup_backend_ok"):
        return redirect(url_for("setup.backend"))

    form = LLMProviderForm()
    if request.method == "GET":
        _prefill_llm_form(form)
        return _render(_TEMPLATE_LLM, form=form)

    if not form.validate_on_submit():
        return _render(_TEMPLATE_LLM, 400, form=form)

    error = _submit_llm_config(form)
    if error is not None:
        return error

    session["setup_llm_ok"] = True
    flash("LLM connection verified and configuration saved.", "success")
    return redirect(url_for("setup.auth"))


@setup_bp.route("/auth", methods=["GET", "POST"])
def auth() -> str | tuple[str, int] | Response:
    """Step 3 — confirm enabling API authentication.

    Generates the strong service-account key (``secrets.token_urlsafe``)
    that step 5 uses when creating the first admin API user.
    """
    if _setup_complete():
        return redirect(url_for(_AUTH_LOGIN_ENDPOINT))
    if not session.get("setup_llm_ok"):
        return redirect(url_for("setup.llm"))

    if "setup_api_key" not in session:
        session["setup_api_key"] = secrets.token_urlsafe(32)

    form = EnableAuthForm()
    if form.validate_on_submit():
        session["setup_auth_ok"] = True
        return redirect(url_for("setup.models"))
    return _render(_TEMPLATE_AUTH, form=form)


def _fetch_model_list(client: BackendAPIClient) -> list[dict[str, Any]]:
    """Fetch the model list from the backend, flashing errors.

    Args:
        client: The configured backend API client.

    Returns:
        The list of model state dicts, or an empty list on failure.
    """
    try:
        resp = client.get_models()
    except requests.RequestException as exc:
        flash(
            f"Could not connect to backend: "
            f"{sanitize_connection_error(str(exc))}",
            "danger",
        )
        return []
    if resp.status_code != 200:
        flash(
            f"Could not load models (HTTP {resp.status_code}): "
            f"{_response_detail(resp)}",
            "danger",
        )
        return []
    try:
        model_states: list[dict[str, Any]] = resp.json().get("models", [])
        return model_states
    except ValueError:
        flash("Backend returned an invalid model list.", "danger")
        return []


def _apply_model_selection(
    client: BackendAPIClient,
    model_list: list[dict[str, Any]],
    selected: set[str],
    form: ModelsForm,
) -> str | tuple[str, int] | None:
    """PUT the desired enabled state for every changed model.

    Args:
        client:     The configured backend API client.
        model_list: Current model states from the backend.
        selected:   Names of the models that should be enabled.
        form:       The form instance (for error re-rendering).

    Returns:
        ``None`` on success, otherwise a rendered error response.
    """
    for model in model_list:
        desired: bool = model["name"] in selected
        if desired == bool(model.get("enabled")):
            continue
        try:
            update_resp = client.put_model(str(model["name"]), desired)
        except requests.RequestException as exc:
            flash(
                f"Could not connect to backend: "
                f"{sanitize_connection_error(str(exc))}",
                "danger",
            )
            return _render(_TEMPLATE_MODELS, 200, form=form, models=model_list)
        if update_resp.status_code != 200:
            flash(
                f"Could not update {model.get('display_name', model['name'])}: "
                f"{_response_detail(update_resp)}",
                "danger",
            )
            return _render(_TEMPLATE_MODELS, 200, form=form, models=model_list)
    return None


@setup_bp.route("/models", methods=["GET", "POST"])
def models() -> str | tuple[str, int] | Response:
    """Step 4 — enable the forecasting models to use.

    At least one model must remain enabled; enforced both client-side
    (template script) and server-side here, in addition to the backend's
    own last-model guard.
    """
    if _setup_complete():
        return redirect(url_for(_AUTH_LOGIN_ENDPOINT))
    if not session.get("setup_auth_ok"):
        return redirect(url_for("setup.auth"))

    form = ModelsForm()
    client = get_api_client()
    model_list = _fetch_model_list(client)

    if request.method == "GET" or not model_list:
        return _render(_TEMPLATE_MODELS, form=form, models=model_list)

    if not form.validate_on_submit():
        return _render(_TEMPLATE_MODELS, 400, form=form, models=model_list)

    selected = set(request.form.getlist("model_enabled"))
    if not selected:
        flash("At least one model must remain enabled.", "danger")
        return _render(_TEMPLATE_MODELS, 200, form=form, models=model_list)

    error = _apply_model_selection(client, model_list, selected, form)
    if error is not None:
        return error

    session["setup_models_ok"] = True
    flash("Model selection saved.", "success")
    return redirect(url_for("setup.admin"))


@setup_bp.route("/admin", methods=["GET", "POST"])
def admin() -> str | tuple[str, int] | Response:
    """Step 5 — create the first admin API user via atomic bootstrap.

    On success the same credentials are stored (Fernet-encrypted) in the
    frontend ``api_credentials`` table so the frontend can authenticate
    against the now-auth-enabled backend.
    """
    if _setup_complete():
        return redirect(url_for(_AUTH_LOGIN_ENDPOINT))
    if not session.get("setup_models_ok"):
        return redirect(url_for("setup.models"))

    form = AdminCreateForm()
    if request.method == "GET":
        form.api_key.data = str(session.get("setup_api_key", ""))
        return _render(_TEMPLATE_ADMIN, form=form)

    if not form.validate_on_submit():
        return _render(_TEMPLATE_ADMIN, 400, form=form)

    username: str = str(form.username.data or "").strip()
    api_key: str = str(form.api_key.data or "").strip()

    client = get_api_client()
    try:
        resp = client.setup_bootstrap(username, api_key)
    except requests.RequestException as exc:
        flash(
            f"Could not connect to backend: "
            f"{sanitize_connection_error(str(exc))}",
            "danger",
        )
        return _render(_TEMPLATE_ADMIN, 200, form=form)

    if resp.status_code == 409:
        flash("Setup was already completed on the backend.", "info")
        session["setup_done"] = True
        return redirect(url_for("setup.done"))
    if resp.status_code != 200:
        flash(
            f"Bootstrap failed (HTTP {resp.status_code}): "
            f"{_response_detail(resp)}",
            "danger",
        )
        return _render(_TEMPLATE_ADMIN, 200, form=form)

    try:
        # Resolve from the DB (not in-process config): a different
        # gunicorn worker may have handled step 1, leaving this worker's
        # config empty.  Using config here would wipe the stored base_url.
        base_url, verify_ssl = resolve_backend_connection()
        if not base_url:
            flash(
                "Backend URL is missing — complete the connection step "
                "again before creating the admin user.",
                "danger",
            )
            return _render(_TEMPLATE_ADMIN, 200, form=form)
        save_api_credentials(
            base_url,
            _DEFAULT_TIMEOUT,
            int(verify_ssl),
            encrypt(username),
            encrypt(api_key),
        )
    except RuntimeError as exc:
        flash(str(exc), "danger")
        return _render(_TEMPLATE_ADMIN, 200, form=form)

    logger.info("Setup bootstrap completed for admin user '%s'.", username)
    session["setup_done"] = True
    return redirect(url_for("setup.done"))


@setup_bp.route("/done")
def done() -> str | tuple[str, int] | Response:
    """Step 6 — completion summary."""
    if not session.get("setup_done") and not _setup_complete():
        return redirect(url_for("setup.index"))
    return _render(_TEMPLATE_DONE)
