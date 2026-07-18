"""
Route handlers for the main application blueprint.

Covers all page routes (chat, overview, quality, stats, model, forecast,
trace, report, get-started) and all AJAX API endpoints (upload, preflight,
columns, preflight-choices, analyze, job-status, chat, clear).
"""

from __future__ import annotations

import io
import json
import logging
import sqlite3
from functools import wraps
from typing import Any, Callable, TypeVar

import pandas
import requests
from flask import (
    abort,
    current_app,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from flask_login import current_user, login_required
from werkzeug.wrappers import Response

from blueprints.decorators import password_change_required
from blueprints.main import main_bp
from services.api_client import BackendAPIClient, get_api_client
from services.pdf_service import report_to_pdf
from services.report_identity import (
    ReportTitleValidationError,
    normalize_report_title,
    report_download_filename,
    resolve_report_identity,
)
from services.report_rendering import render_analysis_report
from services.report_service import (
    ReportLimitError,
    delete_report_for_user,
    get_report_ids_by_job_ids,
    get_report_for_user,
    list_reports_for_user,
    rename_report_for_user,
    report_usage_for_user,
    save_report,
)

_F = TypeVar("_F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)

_login_required: Callable[[_F], _F] = login_required  # type: ignore[assignment]

_FORECAST_SETUP_ENDPOINT: str = "main.forecast_setup"
_BACKEND_CONN_ERROR: str = "Backend connection error."
_JOB_STATUS_TIMEOUT_ERROR: str = (
    "The forecast status check timed out. The backend may still be busy; "
    "please try running the forecast again in a moment."
)


def _safe_error_detail(resp: Any, fallback: str = "Request failed.") -> str:
    """Safely extract an error detail from a backend response.

    Args:
        resp: The backend response object.
        fallback: Message to use when the body is not JSON or has no
            ``detail`` key.

    Returns:
        The error detail string, or ``fallback`` if it cannot be parsed.
    """
    try:
        body = resp.json()
        if isinstance(body, dict):
            return str(body.get("detail", fallback))
        return fallback
    except ValueError:
        return fallback


def analysis_required(f: _F) -> _F:
    """Decorator that redirects to the chat page when no analysis result exists.

    Apply to any route that should only be accessible after a completed
    forecast run.

    Args:
        f: The route function to wrap.

    Returns:
        The wrapped function that enforces the analysis precondition.
    """

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not session.get("analysis_result"):
            flash("Please run an analysis first.", "warning")
            return redirect(url_for(_FORECAST_SETUP_ENDPOINT))
        return f(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


_SETTING_LABELS: dict[str, str] = {
    "frequency": "Frequency alignment",
    "duplicate_strategy": "Duplicate timestamps",
    "missing_strategy": "Missing values",
    "outlier_strategy": "Outlier treatment",
    "smoothing": "Smoothing",
    "data_domain": "Data domain",
}

_DEFAULT_SETTING_VALUES = {
    "",
    "Let AI Decide",
    "Skip / Let AI Guess",
    "None",
    "none",
    "continue",
}


def _custom_settings_from_session() -> list[dict[str, str]]:
    """Return explicit setup choices formatted for the final report."""
    settings: list[dict[str, str]] = []
    horizon = int(session.get("forecast_horizon") or 12)
    settings.append({"label": "Forecast horizon", "value": f"{horizon} periods"})

    model = str(session.get("model_choice") or "Auto (AI selects)")
    if model != "Auto (AI selects)":
        settings.append({"label": "Requested model", "value": model})

    context = str(session.get("user_prompt") or "").strip()
    if context:
        settings.append({"label": "Data context", "value": context})

    options: dict[str, Any] = session.get("preflight_options") or {}
    for key, label in _SETTING_LABELS.items():
        value = str(options.get(key) or "")
        if value not in _DEFAULT_SETTING_VALUES:
            settings.append({"label": label, "value": value})

    disabled_tests = (options.get("statistical_tuning") or {}).get("disabled_tests", [])
    if isinstance(disabled_tests, list) and disabled_tests:
        settings.append(
            {
                "label": "Disabled statistical checks",
                "value": ", ".join(
                    str(test).replace("_", " ") for test in disabled_tests
                ),
            }
        )
    return settings


@main_bp.route("/")
@_login_required
def index() -> Response:
    """Redirect the root URL to the chat tab.

    Returns:
        A redirect response to ``/chat``.
    """
    return redirect(url_for("main.chat"))


@main_bp.route("/chat")
@_login_required
def chat() -> str:
    """Render the chat (Data Explorer) tab.

    Returns:
        Rendered HTML for the chat page.
    """
    chat_history: list[dict[str, Any]] = session.get("chat_history") or []
    return render_template(
        "main/chat.html",
        chat_history=chat_history,
    )


@main_bp.route("/forecast-setup")
@_login_required
def forecast_setup() -> str:
    """Render the forecast setup and configuration page.

    Returns:
        Rendered HTML for the forecast setup page.
    """
    upload_info: dict[str, Any] = session.get("upload_info") or {}
    setup_state = {
        "forecast_horizon": int(session.get("forecast_horizon") or 12),
        "model_choice": str(session.get("model_choice") or "Auto (AI selects)"),
        "user_prompt": str(session.get("user_prompt") or ""),
        "report_title": str(session.get("report_title") or ""),
    }
    return render_template(
        "main/forecast_setup.html",
        upload_info=upload_info,
        setup_state=setup_state,
        setup_error=session.pop("analysis_error", None),
    )


@main_bp.route("/forecast-progress")
@_login_required
def forecast_progress() -> Response | str:
    """Render the dedicated progress screen for an active forecast job."""
    if not session.get("job_running") or not session.get("job_id"):
        flash("There is no forecast currently running.", "warning")
        return redirect(url_for(_FORECAST_SETUP_ENDPOINT))
    return render_template("main/forecast_progress.html")


@main_bp.route("/started")
@_login_required
def started() -> str:
    """Render the Get Started informational tab.

    Returns:
        Rendered HTML for the get started page.
    """
    return render_template("main/started.html")


@main_bp.route("/jobs")
@_login_required
def jobs() -> str:
    """Render the per-user forecast job queue page.

    Shows the user's most recent jobs with live-updating progress bars,
    cancel buttons for active jobs, and view/finalize actions for completed
    jobs.  The page polls ``/api/jobs/mine`` via ``jobs.js``.
    """
    return render_template("main/jobs.html")


@main_bp.route("/overview")
@_login_required
@analysis_required
def overview() -> str:
    """Render the Data Overview tab.

    Returns:
        Rendered HTML for the overview page.
    """
    result: dict[str, Any] = session.get("analysis_result") or {}
    upload_info: dict[str, Any] = session.get("upload_info") or {}
    preview_data: list[dict[str, Any]] = session.get("preview_data") or []
    chart_json: str | None = (
        json.dumps(result["chart_historical"])
        if result.get("chart_historical")
        else None
    )
    return render_template(
        "main/overview.html",
        upload_info=upload_info,
        preview_data=preview_data,
        chart_json=chart_json,
    )


@main_bp.route("/quality")
@_login_required
@analysis_required
def quality() -> str:
    """Render the Data Quality tab.

    Returns:
        Rendered HTML for the data quality page.
    """
    result: dict[str, Any] = session.get("analysis_result") or {}
    validation: dict[str, Any] = result.get("validation") or {}
    return render_template("main/quality.html", v=validation)


@main_bp.route("/stats")
@_login_required
@analysis_required
def stats() -> str:
    """Render the Statistical Analysis tab.

    Returns:
        Rendered HTML for the statistical analysis page.
    """
    result: dict[str, Any] = session.get("analysis_result") or {}
    statistical: dict[str, Any] = result.get("statistical") or {}
    stl_json: str | None = (
        json.dumps(result["chart_stl"]) if result.get("chart_stl") else None
    )
    acf_b64: str | None = result.get("chart_acf_pacf")
    return render_template(
        "main/stats.html",
        s=statistical,
        stl_json=stl_json,
        acf_b64=acf_b64,
    )


@main_bp.route("/model")
@_login_required
@analysis_required
def model() -> str:
    """Render the Forecast Model Selection tab.

    Returns:
        Rendered HTML for the model selection page.
    """
    result: dict[str, Any] = session.get("analysis_result") or {}
    model_sel: dict[str, Any] = result.get("model_selection") or {}
    statistical: dict[str, Any] = result.get("statistical") or {}
    comparison_json: str | None = (
        json.dumps(result["chart_model_comparison"])
        if result.get("chart_model_comparison")
        else None
    )
    rejected: dict[str, str] = {
        k: v
        for k, v in {
            "Holt-Winters": model_sel.get("holt_winters_rejected_reason", ""),
            "ARIMA": model_sel.get("arima_rejected_reason", ""),
            "SARIMA": model_sel.get("sarima_rejected_reason", ""),
            "EWMA": model_sel.get("ewma_rejected_reason", ""),
        }.items()
        if v and k != model_sel.get("selected_model")
    }
    return render_template(
        "main/model.html",
        m=model_sel,
        s=statistical,
        rejected=rejected,
        comparison_json=comparison_json,
    )


@main_bp.route("/forecast")
@_login_required
@analysis_required
def forecast() -> str:
    """Render the Forecast tab.

    Returns:
        Rendered HTML for the forecast page.
    """
    result: dict[str, Any] = session.get("analysis_result") or {}
    fc: dict[str, Any] = result.get("forecast") or {}
    forecast_json: str | None = (
        json.dumps(result["chart_forecast"]) if result.get("chart_forecast") else None
    )
    forecast_rows: list[dict[str, Any]] = []
    dates = fc.get("forecast_dates", [])
    values = fc.get("forecast", [])
    lower = fc.get("lower_ci", [])
    upper = fc.get("upper_ci", [])
    for i, date in enumerate(dates):
        forecast_rows.append(
            {
                "date": date,
                "forecast": round(float(values[i]), 4) if i < len(values) else None,
                "lower_ci": round(float(lower[i]), 4) if i < len(lower) else None,
                "upper_ci": round(float(upper[i]), 4) if i < len(upper) else None,
            }
        )
    return render_template(
        "main/forecast.html",
        fc=fc,
        forecast_rows=forecast_rows,
        forecast_json=forecast_json,
    )


@main_bp.route("/trace")
@_login_required
@analysis_required
def trace() -> str:
    """Render the AI Reasoning Trace tab.

    Returns:
        Rendered HTML for the AI reasoning trace page.
    """
    result: dict[str, Any] = session.get("analysis_result") or {}
    agents: list[dict[str, Any]] = [
        {
            "label": "1. Data Validation Agent",
            "steps": (result.get("validation") or {}).get("reasoning_steps", []),
        },
        {
            "label": "2. Statistical Analysis Agent",
            "steps": (result.get("statistical") or {}).get("reasoning_steps", []),
        },
        {
            "label": "3. Model Selection Agent",
            "steps": (result.get("model_selection") or {}).get("reasoning_steps", []),
        },
        {
            "label": "4. Forecasting Agent",
            "steps": (result.get("forecast") or {}).get("reasoning_steps", []),
        },
        {
            "label": "5. Report Generation Agent",
            "steps": result.get("report_reasoning", []),
        },
    ]
    token_usage: dict[str, Any] = result.get("pipeline_token_usage") or {}
    return render_template("main/trace.html", agents=agents, token_usage=token_usage)


@main_bp.route("/report")
@_login_required
@analysis_required
def report() -> str:
    """Render the Report tab with inline charts.

    Parses ``[VISUAL:TAG]`` tokens from the backend report and converts them
    to structured segment data for the template.

    Returns:
        Rendered HTML for the report page.
    """
    result: dict[str, Any] = session.get("analysis_result") or {}
    upload_info: dict[str, Any] = session.get("upload_info") or {}
    custom_settings = session.get("active_report_custom_settings")
    if custom_settings is None:
        custom_settings = _custom_settings_from_session()
    return render_analysis_report(
        result,
        str(upload_info.get("filename", "data")),
        url_for("main.report_export"),
        custom_settings,
        str(current_user.username),
    )


@main_bp.route("/report/export", methods=["POST"])
@_login_required
@analysis_required
def report_export() -> Response:
    """Generate and stream a PDF of the current report.

    Returns:
        A file download response containing the PDF bytes.
    """
    result: dict[str, Any] = session.get("analysis_result") or {}
    upload_info: dict[str, Any] = session.get("upload_info") or {}
    return _send_report_pdf(result, str(upload_info.get("filename", "data")))


def _send_report_pdf(result: dict[str, Any], filename: str) -> Response:
    """Generate a PDF response from a current or persisted final report."""
    report_text: str = result.get("report", "Report not available.")
    identity = resolve_report_identity(result, filename, str(current_user.username))
    pdf_filename = report_download_filename(identity["title"])
    pdf_bytes = report_to_pdf(
        report_text,
        title=identity["title"],
        result=result,
        prepared_by=identity["prepared_by"],
        creation_date=identity["creation_date"],
    )
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=pdf_filename,
    )


@main_bp.route("/reports")
@_login_required
def reports() -> str:
    """Render the authenticated user's saved-report list."""
    count, limit = report_usage_for_user(int(current_user.id))
    return render_template(
        "main/reports.html",
        reports=list_reports_for_user(int(current_user.id)),
        report_count=count,
        report_limit=limit,
    )


@main_bp.route("/reports/<int:report_id>")
@_login_required
def saved_report(report_id: int) -> str:
    """Activate and render one owner-scoped saved report.

    Making the selected report the active analysis keeps all analysis tabs
    available and ensures they render data from this report rather than a
    previously viewed forecast.
    """
    stored = get_report_for_user(report_id, int(current_user.id))
    if stored is None:
        abort(404)
    session["analysis_result"] = stored
    session["upload_info"] = {
        **(session.get("upload_info") or {}),
        "filename": str(stored["source_filename"]),
    }
    session["forecast_horizon"] = int(stored.get("forecast_horizon") or 12)
    session["active_report_custom_settings"] = stored.get("custom_settings") or []
    return render_analysis_report(
        stored,
        str(stored["source_filename"]),
        url_for("main.saved_report_export", report_id=report_id),
        stored.get("custom_settings") or [],
        str(current_user.username),
    )


@main_bp.route("/reports/<int:report_id>/export", methods=["POST"])
@_login_required
def saved_report_export(report_id: int) -> Response:
    """Export one owner-scoped saved report as PDF."""
    stored = get_report_for_user(report_id, int(current_user.id))
    if stored is None:
        abort(404)
    return _send_report_pdf(stored, str(stored["source_filename"]))


@main_bp.route("/reports/<int:report_id>/delete", methods=["POST"])
@_login_required
def saved_report_delete(report_id: int) -> Response:
    """Delete one owner-scoped saved report."""
    if not delete_report_for_user(report_id, int(current_user.id)):
        abort(404)
    flash("Report deleted.", "success")
    return redirect(url_for("main.reports"))


@main_bp.route("/reports/<int:report_id>/rename", methods=["POST"])
@_login_required
def saved_report_rename(report_id: int) -> Response:
    """Rename one owner-scoped saved report."""
    title = str(request.form.get("title", "")).strip()
    if not title or len(title) > 200:
        flash("Report names must be between 1 and 200 characters.", "danger")
        return redirect(url_for("main.reports"))
    if not rename_report_for_user(report_id, int(current_user.id), title):
        abort(404)
    flash("Report renamed.", "success")
    return redirect(url_for("main.reports"))


@main_bp.route("/load-demo", methods=["GET", "POST"])
@_login_required
def load_demo() -> Response:
    """Upload the bundled airline-passengers demo dataset to the backend.

    Reads ``demo_data.csv`` from the configured ``DEMO_DATA_PATH``, forwards
    it to the backend upload endpoint, and stores the returned upload info in
    the session.

    Returns:
        A redirect response to the chat page.
    """
    demo_path: str = current_app.config.get("DEMO_DATA_PATH", "")
    try:
        with open(demo_path, "rb") as fh:
            demo_bytes = fh.read()
    except FileNotFoundError:
        flash("Demo data file not found. Check DEMO_DATA_PATH configuration.", "danger")
        return redirect(url_for(_FORECAST_SETUP_ENDPOINT))

    _clear_analysis_state()

    try:
        client = get_api_client()
        resp = client.upload_file(
            "sample_airline_passengers.csv", demo_bytes, "text/csv"
        )
        if resp.status_code == 200:
            upload_info = resp.json()
            session["upload_info"] = upload_info
            session["date_col"] = upload_info.get("detected_date_col")
            session["value_col"] = upload_info.get("detected_value_col")
            session["preview_data"] = _parse_preview(
                demo_bytes, "sample_airline_passengers.csv"
            )
            flash(
                f"Demo data loaded — {upload_info.get('rows', 0)} rows "
                "(airline passengers 1949-1960).",
                "success",
            )
        else:
            detail = resp.json().get("detail", "Demo upload failed.")
            flash(detail, "danger")
    except (requests.RequestException, ValueError):
        logger.exception("Backend connection error during demo data load")
        flash(
            "Backend connection error. Verify Admin API Config and service availability.",
            "danger",
        )

    return redirect(url_for(_FORECAST_SETUP_ENDPOINT))


def _forward_upload(file: Any) -> Response:
    """Forward a validated upload to the backend and store session state.

    Streams the file to the backend without loading the entire file into
    memory.  A small preview is read separately for the overview tab.
    """
    filename: str = file.filename
    content_type: str = file.content_type or "application/octet-stream"

    _clear_analysis_state()

    try:
        client = get_api_client()
        resp = client.upload_file_stream(filename, file.stream, content_type)
        if resp.status_code == 200:
            upload_info: dict[str, Any] = resp.json()
            session["upload_info"] = upload_info
            session["date_col"] = upload_info.get("detected_date_col")
            session["value_col"] = upload_info.get("detected_value_col")
            # Read a small preview for the overview tab.  This seeks back
            # to the beginning of the stream after the upload completes.
            file.stream.seek(0)
            session["preview_data"] = _parse_preview(file.stream, filename)
            return make_response(jsonify(upload_info), 200)
        return make_response(
            jsonify({"error": _safe_error_detail(resp, "Upload failed.")}),
            resp.status_code,
        )
    except (requests.RequestException, ValueError):
        logger.exception("Backend connection error during upload")
        return make_response(jsonify({"error": _BACKEND_CONN_ERROR}), 503)


@main_bp.route("/api/upload", methods=["POST"])
@_login_required
def api_upload() -> Response:
    """Accept a file upload from the browser and forward it to the backend.

    Expects a multipart form field named ``file``.

    Returns:
        JSON with upload info on success, or an error object on failure.
    """
    if "file" not in request.files:
        return make_response(jsonify({"error": "No file provided"}), 400)

    file = request.files["file"]
    if not file.filename:
        return make_response(jsonify({"error": "Empty filename"}), 400)

    return _forward_upload(file)


@main_bp.route("/api/columns", methods=["POST"])
@_login_required
def api_columns() -> Response:
    """Persist the user's column selection and return updated preflight info.

    Expects a JSON body with ``date_col`` and ``value_col`` keys.

    Returns:
        JSON with the preflight result or an error object.
    """
    data: dict[str, Any] = request.get_json(silent=True) or {}
    date_col: str = str(data.get("date_col", ""))
    value_col: str = str(data.get("value_col", ""))

    if not date_col or not value_col:
        return make_response(
            jsonify({"error": "date_col and value_col are required"}), 400
        )

    session["date_col"] = date_col
    session["value_col"] = value_col

    upload_info: dict[str, Any] = session.get("upload_info") or {}
    file_id: str = upload_info.get("file_id", "")
    horizon: int = int(session.get("forecast_horizon") or 12)

    if not file_id:
        return jsonify({"preflight": None})

    try:
        client = get_api_client()
        resp = client.get_preflight(file_id, horizon, date_col, value_col)
        if resp.status_code == 200:
            preflight: dict[str, Any] = resp.json()
            session["preflight_result"] = preflight
            session["preflight_options"] = _preflight_defaults(preflight)
            return jsonify({"preflight": preflight})
        return make_response(
            jsonify({"error": _safe_error_detail(resp, "Preflight failed.")}),
            resp.status_code,
        )
    except (requests.RequestException, ValueError):
        logger.exception("Backend connection error during preflight")
        return make_response(jsonify({"error": _BACKEND_CONN_ERROR}), 503)


@main_bp.route("/api/preflight-choices", methods=["POST"])
@_login_required
def api_preflight_choices() -> Response:
    """Store the user's preflight decision choices in the session.

    Expects a JSON body with a ``choices`` object mapping decision keys to
    selected values.

    Returns:
        JSON ``{"ok": true}`` on success.
    """
    data: dict[str, Any] = request.get_json(silent=True) or {}
    choices: dict[str, Any] = data.get("choices", {})
    session["preflight_options"] = choices
    return jsonify({"ok": True})


@main_bp.route("/api/setup-state", methods=["POST"])
@_login_required
def api_setup_state() -> Response:
    """Persist non-sensitive wizard configuration while the user works."""
    data: dict[str, Any] = request.get_json(silent=True) or {}
    horizon = data.get("forecast_horizon")
    if horizon is not None:
        try:
            session["forecast_horizon"] = max(1, min(100, int(horizon)))
        except (TypeError, ValueError):
            return make_response(jsonify({"error": "Invalid forecast horizon"}), 400)
    if "model_choice" in data:
        session["model_choice"] = str(data["model_choice"])
    if "user_prompt" in data:
        session["user_prompt"] = str(data["user_prompt"]).strip()
    if "report_title" in data:
        raw_title = str(data["report_title"] or "").strip()
        try:
            normalize_report_title(
                raw_title,
                str((session.get("upload_info") or {}).get("filename") or "data"),
            )
        except ReportTitleValidationError as exc:
            return make_response(
                jsonify({"error": str(exc), "field": "report_title"}), 400
            )
        session["report_title"] = raw_title
    return jsonify({"ok": True})


def _build_analyze_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Build the backend analysis payload from request/session state."""
    upload_info: dict[str, Any] = session.get("upload_info") or {}
    file_id: str = upload_info.get("file_id", "")

    date_col: str = str(data.get("date_col") or session.get("date_col") or "")
    value_col: str = str(data.get("value_col") or session.get("value_col") or "")
    horizon: int = int(
        data.get("forecast_horizon") or session.get("forecast_horizon") or 12
    )
    model_choice: str = str(data.get("model_choice") or "Auto (AI selects)")
    user_prompt: str = str(data.get("user_prompt") or "").strip()
    raw_report_title: str = str(
        data.get("report_title")
        if "report_title" in data
        else session.get("report_title") or ""
    )
    preflight_options: dict[str, Any] = (
        data.get("preflight_options") or session.get("preflight_options") or {}
    )
    source_filename: str = str(upload_info.get("filename", "data"))
    report_name = normalize_report_title(raw_report_title, source_filename)

    session["forecast_horizon"] = horizon
    session["model_choice"] = model_choice
    session["user_prompt"] = user_prompt
    session["report_title"] = raw_report_title.strip()
    session["preflight_options"] = preflight_options

    forced_model: str | None = (
        None if model_choice == "Auto (AI selects)" else model_choice
    )

    return {
        "file_id": file_id,
        "forecast_horizon": horizon,
        "date_col": date_col,
        "value_col": value_col,
        "forced_model": forced_model,
        "user_prompt": user_prompt or None,
        "preflight_options": preflight_options,
        "report_name": report_name,
        "source_filename": source_filename,
        "custom_settings": _custom_settings_from_session(),
    }


@main_bp.route("/api/analyze", methods=["POST"])
@_login_required
def api_analyze() -> Response:
    """Submit an analysis job to the backend and store the job ID in session.

    Expects a JSON body with ``date_col``, ``value_col``, ``forecast_horizon``,
    ``model_choice``, ``user_prompt``, and ``preflight_options``.

    Returns:
        JSON with ``job_id`` on success (HTTP 202) or an error object.
    """
    data: dict[str, Any] = request.get_json(silent=True) or {}
    try:
        payload = _build_analyze_payload(data)
    except ReportTitleValidationError as exc:
        return make_response(jsonify({"error": str(exc), "field": "report_title"}), 400)

    if not payload["file_id"]:
        return make_response(jsonify({"error": "No file uploaded"}), 400)

    payload["application_user_id"] = current_user.id
    payload["application_username"] = current_user.username
    payload["application_user_is_admin"] = current_user.is_admin

    try:
        client = get_api_client()
        resp = client.submit_analysis(payload)
        if resp.status_code == 202:
            job_id: str = resp.json()["job_id"]
            session["job_id"] = job_id
            session["job_running"] = True
            session["job_progress"] = 0
            session["job_step"] = "Queued - waiting for an available slot..."
            session["analysis_error"] = None
            session.pop("active_report_custom_settings", None)
            return make_response(
                jsonify(
                    {
                        "job_id": job_id,
                        "redirect": url_for("main.forecast_progress"),
                    }
                ),
                202,
            )
        return make_response(
            jsonify({"error": _safe_error_detail(resp, "Failed to submit job.")}),
            resp.status_code,
        )
    except (requests.RequestException, ValueError):
        logger.exception("Backend connection error during preflight")
        return make_response(jsonify({"error": _BACKEND_CONN_ERROR}), 503)


def _handle_done_job(
    client: BackendAPIClient, job_id: str, status_data: dict[str, Any]
) -> Response:
    """Fetch full results for a completed job and update session state.

    Report metadata (source filename, horizon, custom settings) is read from
    the backend job record via :meth:`get_job_results` so that it is correct
    even when multiple forecasts have been submitted (the Flask session would
    be stale for earlier jobs).
    """
    results_resp = client.get_job_results(job_id, application_user_id=current_user.id)
    if results_resp.status_code != 200:
        return _handle_error_job(client, job_id, status_data)
    result_data = results_resp.json().get("result", {})
    session["analysis_result"] = result_data
    session["llm_fallback"] = result_data.get("llm_fallback", False)
    session["job_running"] = False
    session["job_id"] = None
    session["analysis_error"] = None
    session.pop("active_report_custom_settings", None)

    # Read per-job metadata from the backend job record (not the session,
    # which may have been overwritten by later submissions).
    job_record = results_resp.json()
    source_filename = str(job_record.get("source_filename") or "data")
    report_title = str(job_record.get("report_name") or "").strip() or None
    forecast_horizon = int(job_record.get("forecast_horizon") or 12)
    custom_settings_raw = job_record.get("custom_settings_json")
    if isinstance(custom_settings_raw, str):
        try:
            custom_settings = json.loads(custom_settings_raw)
        except ValueError:
            custom_settings = _custom_settings_from_session()
    else:
        custom_settings = _custom_settings_from_session()

    try:
        save_report(
            user_id=int(current_user.id),
            result=result_data,
            source_filename=source_filename,
            forecast_horizon=forecast_horizon,
            custom_settings=custom_settings,
            job_id=job_id,
            report_title=report_title,
        )
    except ReportLimitError:
        flash(
            "This report was not saved because you reached your report limit. "
            "Delete a saved report before running another forecast.",
            "warning",
        )
    except (sqlite3.Error, TypeError, ValueError):
        logger.exception(
            "Failed to persist completed report for user_id=%s", current_user.id
        )
        flash("The report was generated but could not be saved.", "warning")
    return jsonify(
        {
            "status": status_data.get("status", ""),
            "progress": int(status_data.get("progress", 0)),
            "step": str(status_data.get("step", "")),
            "done": True,
            "redirect": url_for("main.report"),
            **_job_activity_payload(status_data),
        }
    )


def _handle_error_job(
    client: BackendAPIClient, job_id: str, status_data: dict[str, Any]
) -> Response:
    """Fetch a terminal error/cancellation message and clear session state."""
    results_resp = client.get_job_results(job_id, application_user_id=current_user.id)
    status = str(status_data.get("status", ""))
    default_message = (
        "Forecast cancelled." if status == "cancelled" else "Analysis failed."
    )
    error_msg = default_message
    if results_resp.status_code == 200:
        error_msg = str(results_resp.json().get("error") or default_message)
    session["job_running"] = False
    session["job_id"] = None
    session["analysis_error"] = error_msg
    return jsonify(
        {
            "status": status_data.get("status", ""),
            "progress": int(status_data.get("progress", 0)),
            "step": str(status_data.get("step", "")),
            "done": False,
            "error": error_msg,
            **_job_activity_payload(status_data),
        }
    )


def _job_activity_payload(status_data: dict[str, Any]) -> dict[str, Any]:
    """Return backend-owned heartbeat fields without reclassifying them."""
    return {
        "heartbeat_at": status_data.get("heartbeat_at"),
        "progress_updated_at": status_data.get("progress_updated_at"),
        "elapsed_seconds": int(status_data.get("elapsed_seconds", 0)),
        "heartbeat_age_seconds": status_data.get("heartbeat_age_seconds"),
        "stage_age_seconds": status_data.get("stage_age_seconds"),
        "liveness": status_data.get("liveness", "queued"),
    }


def _transient_job_poll_response(message: str, status_code: int) -> Response:
    """Report a recoverable polling interruption while retaining job state."""
    return make_response(
        jsonify(
            {
                "status": "reconnecting",
                "progress": int(session.get("job_progress", 0)),
                "step": str(session.get("job_step", "Processing…")),
                "done": False,
                "transient": True,
                "error": message,
                "liveness": "delayed",
            }
        ),
        status_code,
    )


@main_bp.route("/api/jobs/status")
@_login_required
def api_job_status() -> Response:
    """Poll the backend for the current job's progress.

    Reads ``job_id`` from the session, proxies the status request to the
    backend, and updates session state when the job completes or fails.

    Returns:
        JSON with ``status``, ``progress``, ``step``, ``done``, and
        optionally ``error`` or ``redirect`` keys.
    """
    job_id: str = session.get("job_id") or ""
    if not job_id:
        return make_response(jsonify({"error": "No active job"}), 400)

    try:
        client = get_api_client()

        status_resp = client.get_job_status_lightweight(
            job_id, application_user_id=current_user.id
        )
        if status_resp.status_code != 200:
            return _transient_job_poll_response(
                "Reconnecting to the forecast worker…", 502
            )

        status_data: dict[str, Any] = status_resp.json()
        status: str = status_data.get("status", "")
        progress: int = int(status_data.get("progress", 0))
        step: str = str(status_data.get("step", ""))

        session["job_progress"] = progress
        session["job_step"] = step

        if status == "done":
            return _handle_done_job(client, job_id, status_data)
        if status in ("error", "cancelled"):
            return _handle_error_job(client, job_id, status_data)

        return jsonify(
            {
                "status": status,
                "progress": progress,
                "step": step,
                "done": False,
                **_job_activity_payload(status_data),
            }
        )

    except requests.exceptions.Timeout:
        logger.exception("Status poll timed out")
        return _transient_job_poll_response(_JOB_STATUS_TIMEOUT_ERROR, 504)
    except requests.exceptions.RequestException:
        logger.exception("Status poll request error")
        return _transient_job_poll_response(_BACKEND_CONN_ERROR, 503)
    except (KeyError, ValueError):
        logger.exception("Status poll error")
        return _transient_job_poll_response("Status response was interrupted.", 503)


# ── Per-user job queue AJAX endpoints ─────────────────────────────────────────


@main_bp.route("/api/jobs/mine")
@_login_required
def api_jobs_mine() -> Response:
    """Return the current user's recent jobs with report-linkage enrichment.

    Fetches the job list from the backend (scoped by
    ``X-Application-User-ID`` derived from ``current_user.id``), then
    enriches each item with ``report_id``, ``report_ready``, and
    ``finalization_error`` by looking up the frontend ``forecast_reports``
    table.  For ``done`` jobs without a report, finalization is attempted
    automatically so that closing the browser mid-forecast does not
    prevent report creation.

    Returns:
        JSON list of user-queue DTOs.
    """
    try:
        client = get_api_client()
        resp = client.list_my_jobs(int(current_user.id))
        if resp.status_code != 200:
            return make_response(
                jsonify({"error": _safe_error_detail(resp, "Failed to list jobs.")}),
                resp.status_code,
            )
        jobs_list: list[dict[str, Any]] = resp.json()
    except (requests.RequestException, ValueError):
        logger.exception("Backend connection error during job list")
        return make_response(jsonify({"error": _BACKEND_CONN_ERROR}), 503)

    job_ids = [str(job.get("job_id", "")) for job in jobs_list]
    report_ids = get_report_ids_by_job_ids(job_ids, int(current_user.id))
    enriched: list[dict[str, Any]] = []
    for job in jobs_list:
        job_id = str(job.get("job_id", ""))
        status = str(job.get("status", ""))
        report_id: int | None = None
        report_ready = False
        finalization_error: str | None = None

        if job_id:
            existing_report_id = report_ids.get(job_id)
            if existing_report_id is not None:
                report_id = existing_report_id
                report_ready = True
            elif status == "done":
                # Auto-finalize: the pipeline completed but the report has
                # not been saved yet (e.g. browser was closed).
                finalize_result = _finalize_job_report(client, job_id)
                if finalize_result.get("report_id") is not None:
                    report_id = int(finalize_result["report_id"])
                    report_ready = True
                elif finalize_result.get("finalization_error"):
                    finalization_error = str(finalize_result["finalization_error"])

        enriched.append(
            {
                "job_id": job_id,
                "report_name": job.get("report_name", ""),
                "status": status,
                "progress": int(job.get("progress", 0)),
                "step": job.get("step", ""),
                "queued_at": job.get("queued_at", ""),
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
                "error": job.get("error"),
                "can_cancel": bool(job.get("can_cancel", False)),
                "forecast_horizon": int(job.get("forecast_horizon", 0)),
                "forced_model": job.get("forced_model"),
                "report_id": report_id,
                "report_ready": report_ready,
                "finalization_error": finalization_error,
                **_job_activity_payload(job),
            }
        )
    active_session_job_id = str(session.get("job_id") or "")
    if active_session_job_id and any(
        job["job_id"] == active_session_job_id
        and job["status"] in ("done", "error", "cancelled")
        for job in enriched
    ):
        session["job_running"] = False
        session["job_id"] = None
    return jsonify(enriched)


@main_bp.route("/api/jobs/mine/terminal", methods=["POST"])
@_login_required
def api_my_terminal_jobs_clear() -> Response:
    """Delete only the current frontend user's terminal forecast jobs."""
    try:
        response = get_api_client().clear_my_terminal_jobs(
            application_user_id=int(current_user.id)
        )
        if response.status_code != 200:
            return make_response(
                jsonify(
                    {
                        "error": _safe_error_detail(
                            response, "Failed to clear completed jobs."
                        )
                    }
                ),
                response.status_code,
            )
        return jsonify({"deleted_count": int(response.json().get("deleted_count", 0))})
    except (requests.RequestException, TypeError, ValueError):
        logger.exception("Backend connection error during terminal job cleanup")
        return make_response(jsonify({"error": _BACKEND_CONN_ERROR}), 503)


def _finalize_job_report(client: BackendAPIClient, job_id: str) -> dict[str, Any]:
    """Finalize a completed job into a saved report.

    Fetches durable results from the backend, reads per-job metadata from
    the job record, and calls :func:`save_report` with ``job_id`` for
    idempotency.  This is used both by the auto-finalization in
    :func:`api_jobs_mine` and by the explicit ``/api/jobs/<job_id>/finalize``
    endpoint.

    Args:
        client: The backend API client.
        job_id: The job identifier to finalize.

    Returns:
        A dict with ``report_id`` on success, or ``finalization_error`` on
        failure.
    """
    try:
        results_resp = client.get_job_results(
            job_id, application_user_id=current_user.id
        )
        if results_resp.status_code != 200:
            return {"finalization_error": "backend_error"}
        job_record = results_resp.json()
        result_data = job_record.get("result")
        if not result_data:
            return {"finalization_error": "no_result"}
        source_filename = str(job_record.get("source_filename") or "data")
        report_title = str(job_record.get("report_name") or "").strip() or None
        forecast_horizon = int(job_record.get("forecast_horizon") or 12)
        custom_settings_raw = job_record.get("custom_settings_json")
        if isinstance(custom_settings_raw, str):
            try:
                custom_settings = json.loads(custom_settings_raw)
            except ValueError:
                custom_settings = []
        else:
            custom_settings = []
        report_id = save_report(
            user_id=int(current_user.id),
            result=result_data,
            source_filename=source_filename,
            forecast_horizon=forecast_horizon,
            custom_settings=custom_settings,
            job_id=job_id,
            report_title=report_title,
        )
        return {"report_id": report_id}
    except ReportLimitError:
        return {"finalization_error": "report_limit"}
    except (sqlite3.Error, TypeError, ValueError, requests.RequestException):
        logger.exception(
            "Failed to finalize report for job_id=%s user_id=%s",
            job_id,
            current_user.id,
        )
        return {"finalization_error": "save_failed"}


@main_bp.route("/api/jobs/<job_id>/finalize", methods=["POST"])
@_login_required
def api_job_finalize(job_id: str) -> Response:
    """Finalize a completed job into a saved report.

    This endpoint is called explicitly by the user (via the "Finalize"
    button) when auto-finalization failed due to the report limit.  After
    the user deletes an old report, they can retry finalization here.

    Returns:
        JSON with ``report_id`` on success, or ``error`` and
        ``finalization_error`` on failure.
    """
    try:
        client = get_api_client()
        result = _finalize_job_report(client, job_id)
    except (requests.RequestException, ValueError):
        return make_response(jsonify({"error": _BACKEND_CONN_ERROR}), 503)
    if result.get("report_id") is not None:
        return jsonify({"report_id": result["report_id"]})
    error_type = str(result.get("finalization_error", "unknown"))
    if error_type == "report_limit":
        return make_response(
            jsonify(
                {
                    "error": (
                        "Report limit reached. Delete a saved report and try again."
                    ),
                    "finalization_error": "report_limit",
                }
            ),
            409,
        )
    return make_response(
        jsonify(
            {
                "error": "Could not finalize this report.",
                "finalization_error": error_type,
            }
        ),
        502,
    )


@main_bp.route("/api/jobs/<job_id>/cancel", methods=["POST"])
@_login_required
def api_job_cancel(job_id: str) -> Response:
    """Request cooperative cancellation of a forecast job.

    The ``application_user_id`` is derived from ``current_user.id`` and sent
    as the ``X-Application-User-ID`` header to the backend.  The backend
    scopes the cancellation by both the backend credential and the
    application user ID.

    Returns:
        JSON with ``cancel_status`` on success, or ``error`` on failure.
    """
    try:
        client = get_api_client()
        resp = client.cancel_job(job_id, application_user_id=current_user.id)
        if resp.status_code == 200:
            return jsonify(
                {"job_id": job_id, "cancel_status": resp.json().get("cancel_status")}
            )
        if resp.status_code == 404:
            return make_response(
                jsonify({"error": "Job not found or not owned by you."}), 404
            )
        if resp.status_code == 409:
            return make_response(jsonify({"error": "Job is already complete."}), 409)
        return make_response(
            jsonify({"error": _safe_error_detail(resp, "Failed to cancel job.")}),
            resp.status_code,
        )
    except (requests.RequestException, ValueError):
        return make_response(jsonify({"error": _BACKEND_CONN_ERROR}), 503)


@main_bp.route("/api/llm-health")
@_login_required
def api_llm_health() -> Response:
    """Proxy the LLM health check request to the backend.

    Requires authentication to avoid exposing backend LLM state to
    public callers.

    Returns:
        JSON response from the backend's `/llm-health` endpoint.
    """
    try:
        client = get_api_client()
        resp = client.get_llm_health()
        return make_response(jsonify(resp.json()), resp.status_code)
    except (requests.RequestException, ValueError):
        logger.exception("Failed to proxy LLM health check")
        return make_response(jsonify({"error": "Failed to check LLM health."}), 503)


@main_bp.route("/api/chat", methods=["POST"])
@_login_required
def api_chat() -> Response:
    """Forward a chat query to the backend and store the result in history.

    Expects a JSON body with a ``query`` string.

    Returns:
        JSON with ``answer``, ``visualization_type``, and
        ``visualization_data`` keys, mirroring the backend response.
    """
    data: dict[str, Any] = request.get_json(silent=True) or {}
    query: str = str(data.get("query", "")).strip()

    if not query:
        return make_response(jsonify({"error": "Query is required"}), 400)

    upload_info: dict[str, Any] = session.get("upload_info") or {}
    file_id: str | None = upload_info.get("file_id") or None

    try:
        client = get_api_client()
        resp = client.send_chat(file_id, query)
        if resp.status_code == 200:
            result: dict[str, Any] = resp.json()
            chat_history: list[dict[str, Any]] = session.get("chat_history") or []
            chat_history.append({"role": "user", "content": query})
            chat_history.append(
                {
                    "role": "assistant",
                    "content": result.get("answer", ""),
                    "visualization_type": result.get("visualization_type"),
                    "visualization_data": result.get("visualization_data"),
                }
            )
            if len(chat_history) > 100:
                chat_history = chat_history[-100:]
            session["chat_history"] = chat_history
            return jsonify(result)
        return make_response(
            jsonify({"error": _safe_error_detail(resp, "Chat request failed.")}),
            resp.status_code,
        )
    except (requests.RequestException, ValueError):
        logger.exception("Backend connection error during chat")
        return make_response(jsonify({"error": _BACKEND_CONN_ERROR}), 503)


@main_bp.route("/api/clear", methods=["POST"])
@_login_required
def api_clear() -> Response:
    """Clear all analysis state from the session.

    Returns:
        JSON ``{"ok": true}``.
    """
    _clear_analysis_state()
    return jsonify({"ok": True})


def _clear_analysis_state() -> None:
    """Reset all analysis-related session keys without affecting auth state."""
    for key in (
        "upload_info",
        "analysis_result",
        "job_id",
        "job_running",
        "job_progress",
        "job_step",
        "analysis_error",
        "active_report_custom_settings",
        "report_title",
        "preflight_result",
        "preflight_options",
        "preview_data",
        "date_col",
        "value_col",
    ):
        session.pop(key, None)


def _preflight_defaults(preflight: dict[str, Any]) -> dict[str, Any]:
    """Extract the default option values from a preflight response.

    Args:
        preflight: Preflight dict containing a ``decisions`` list.

    Returns:
        Dict mapping each decision key to its default value.
    """
    return {
        decision["key"]: decision["default"]
        for decision in preflight.get("decisions", [])
    }


def _parse_preview(content: Any, filename: str) -> list[dict[str, Any]]:
    """Parse the first 20 rows of an uploaded file for the overview tab.

    Args:
        content:  Raw file bytes or a seekable file-like object.
        filename: Original filename used to determine the parser.

    Returns:
        List of row dicts (at most 20 rows), or an empty list on parse error.
    """
    try:
        if hasattr(content, "seek"):
            content.seek(0)
            buffer = content
        else:
            buffer = io.BytesIO(content)
        lower = filename.lower()
        if lower.endswith(".xlsx"):
            df = pandas.read_excel(buffer).head(20)
        elif lower.endswith(".json"):
            df = pandas.read_json(buffer).head(20)
        else:
            df = pandas.read_csv(buffer).head(20)
        return df.astype(str).to_dict(orient="records")
    except (OSError, ValueError):
        return []
