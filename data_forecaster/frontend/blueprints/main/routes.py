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
import re
from functools import wraps
from typing import Any, Callable, TypeVar

import bleach
import markdown as md_lib
import pandas
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
from services.api_client import get_api_client
from services.report_service import (
    ReportLimitError,
    delete_report_for_user,
    get_report_for_user,
    list_reports_for_user,
    rename_report_for_user,
    report_usage_for_user,
    save_report,
)

_F = TypeVar("_F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)

_VISUAL_TAG_RE: re.Pattern[str] = re.compile(r"\[VISUAL:([A-Z_]+)\]")

_login_required: Callable[[_F], _F] = login_required  # type: ignore[assignment]

_FORECAST_SETUP_ENDPOINT: str = "main.forecast_setup"
_BACKEND_CONN_ERROR: str = "Backend connection error."


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
    except Exception:
        return fallback


_CHART_FIELD_BY_TAG: dict[str, str] = {
    "HISTORICAL": "chart_historical",
    "STL": "chart_stl",
    "ACF_PACF": "chart_acf_pacf",
    "FORECAST": "chart_forecast",
    "COMPARISON": "chart_model_comparison",
}

_BLEACH_ALLOWED_TAGS: list[str] = [
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "ul",
    "ol",
    "li",
    "strong",
    "em",
    "code",
    "pre",
    "blockquote",
    "hr",
    "a",
    "br",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
]

_BLEACH_ALLOWED_ATTRS: dict[str, list[str]] = {
    "a": ["href", "title", "rel"],
    "th": ["scope"],
    "td": ["colspan", "rowspan"],
}


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


def _markdown_to_html(text: str) -> str:
    """Convert a markdown string to sanitised HTML.

    Uses the ``markdown`` library with the ``tables`` and ``fenced_code``
    extensions, then passes the result through ``bleach`` to remove any
    potentially unsafe markup.

    Args:
        text: Markdown-formatted string.

    Returns:
        Safe HTML string.
    """
    raw_html: str = md_lib.markdown(
        text,
        extensions=["tables", "fenced_code", "nl2br"],
    )
    return bleach.clean(
        raw_html,
        tags=_BLEACH_ALLOWED_TAGS,
        attributes=_BLEACH_ALLOWED_ATTRS,
        strip=True,
    )


def _build_chart_segment(tag: str, result: dict[str, Any]) -> dict[str, Any]:
    """Return a chart segment descriptor for the given visual tag."""
    field = _CHART_FIELD_BY_TAG.get(tag)
    chart_data = result.get(field) if field else None
    if tag == "ACF_PACF":
        return {"type": "chart", "tag": tag, "acf_b64": chart_data}
    return {
        "type": "chart",
        "tag": tag,
        "chart_json": chart_data if chart_data else None,
    }


def _parse_report_segments(
    report_text: str,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Split a report into alternating text and chart segments.

    The backend embeds ``[VISUAL:TAG]`` tokens in the report markdown.  This
    function splits on those tokens and returns a list of segment descriptors
    that the Jinja template can iterate over.

    Args:
        report_text: Raw report markdown string from the backend.
        result:      Complete analysis result dict containing chart data.

    Returns:
        List of segment dicts.  Each dict has a ``type`` key of either
        ``'text'`` (with an ``html`` key) or ``'chart'`` (with ``tag`` and
        optionally ``chart_json`` or ``acf_b64`` keys).
    """
    parts = _VISUAL_TAG_RE.split(report_text)
    segments: list[dict[str, Any]] = []

    for idx, segment in enumerate(parts):
        if idx % 2 == 0:
            if segment.strip():
                segments.append({"type": "text", "html": _markdown_to_html(segment)})
        else:
            segments.append(_build_chart_segment(segment, result))

    return segments


def _remove_web_dashboard_section(report_text: str) -> str:
    """Remove the markdown dashboard section from the web report body.

    The report page renders the executive dashboard as a richer tile-based
    overview.  PDF/export still receives the full markdown report, including
    the dashboard table, so audit/export behavior remains unchanged.
    """
    sections = report_text.split("\n\n---\n\n")
    filtered = [
        section
        for section in sections
        if not section.lstrip().startswith("## 1. Executive Dashboard")
    ]
    return "\n\n---\n\n".join(filtered)


def _render_report(
    result: dict[str, Any],
    source_filename: str,
    export_url: str,
) -> str:
    """Render a current or persisted final report using shared presentation."""
    executive_report: dict[str, Any] | None = result.get("executive_report")
    report_md: str = result.get("report", "Report not available.")
    web_report_md = _remove_web_dashboard_section(report_md)
    base_name = (
        source_filename.rsplit(".", 1)[0]
        if "." in source_filename
        else source_filename
    )
    pdf_filename = f"forecast_report_{base_name or 'data'}.pdf"
    return render_template(
        "main/report.html",
        segments=_parse_report_segments(web_report_md, result),
        pdf_filename=pdf_filename,
        er=executive_report,
        llm_fallback=bool(result.get("llm_fallback", False)),
        export_url=export_url,
    )


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
        }.items()
        if v
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
    return render_template(
        "main/trace.html", agents=agents, token_usage=token_usage
    )


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
    return _render_report(
        result,
        str(upload_info.get("filename", "data")),
        url_for("main.report_export"),
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
    from services.pdf_service import report_to_pdf

    report_text: str = result.get("report", "Report not available.")
    base_name = filename.rsplit(".", 1)[0] if "." in filename else filename
    pdf_filename = f"forecast_report_{base_name or 'data'}.pdf"
    pdf_bytes = report_to_pdf(
        report_text,
        title=pdf_filename.replace("_", " ").replace(".pdf", ""),
        result=result,
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
    """Render one owner-scoped saved report."""
    stored = get_report_for_user(report_id, int(current_user.id))
    if stored is None:
        abort(404)
    return _render_report(
        stored,
        str(stored["source_filename"]),
        url_for("main.saved_report_export", report_id=report_id),
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
    except Exception:
        logger.exception("Backend connection error during demo data load")
        flash(
            "Backend connection error. Verify BACKEND_URL and service availability.",
            "danger",
        )

    return redirect(url_for(_FORECAST_SETUP_ENDPOINT))


def _forward_upload(file: Any) -> Response:
    """Forward a validated upload to the backend and store session state."""
    filename: str = file.filename
    content: bytes = file.read()
    content_type: str = file.content_type or "application/octet-stream"

    _clear_analysis_state()

    try:
        client = get_api_client()
        resp = client.upload_file(filename, content, content_type)
        if resp.status_code == 200:
            upload_info: dict[str, Any] = resp.json()
            session["upload_info"] = upload_info
            session["date_col"] = upload_info.get("detected_date_col")
            session["value_col"] = upload_info.get("detected_value_col")
            session["preview_data"] = _parse_preview(content, filename)
            return make_response(jsonify(upload_info), 200)
        return make_response(
            jsonify({"error": _safe_error_detail(resp, "Upload failed.")}),
            resp.status_code,
        )
    except Exception:
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
    except Exception:
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
    preflight_options: dict[str, Any] = (
        data.get("preflight_options") or session.get("preflight_options") or {}
    )

    session["forecast_horizon"] = horizon
    session["model_choice"] = model_choice
    session["user_prompt"] = user_prompt

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
    payload = _build_analyze_payload(data)

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
            return make_response(jsonify({"job_id": job_id}), 202)
        return make_response(
            jsonify({"error": _safe_error_detail(resp, "Failed to submit job.")}),
            resp.status_code,
        )
    except Exception:
        logger.exception("Backend connection error during preflight")
        return make_response(jsonify({"error": _BACKEND_CONN_ERROR}), 503)


def _handle_done_job(
    client: BackendAPIClient, job_id: str, status_data: dict[str, Any]
) -> Response:
    """Fetch full results for a completed job and update session state."""
    results_resp = client.get_job_results(job_id)
    if results_resp.status_code != 200:
        return _handle_error_job(client, job_id, status_data)
    result_data = results_resp.json().get("result", {})
    session["analysis_result"] = result_data
    session["llm_fallback"] = result_data.get("llm_fallback", False)
    session["job_running"] = False
    session["job_id"] = None
    session["analysis_error"] = None
    upload_info: dict[str, Any] = session.get("upload_info") or {}
    try:
        save_report(
            user_id=int(current_user.id),
            result=result_data,
            source_filename=str(upload_info.get("filename", "data")),
            forecast_horizon=int(session.get("forecast_horizon") or 12),
        )
    except ReportLimitError:
        flash(
            "This report was not saved because you reached your report limit. "
            "Delete a saved report before running another forecast.",
            "warning",
        )
    except Exception:  # pylint: disable=broad-exception-caught
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
        }
    )


def _handle_error_job(
    client: BackendAPIClient, job_id: str, status_data: dict[str, Any]
) -> Response:
    """Fetch the error message for a failed job and update session state."""
    results_resp = client.get_job_results(job_id)
    error_msg: str = "Analysis failed."
    if results_resp.status_code == 200:
        error_msg = str(results_resp.json().get("error", "Analysis failed."))
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
        }
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

        status_resp = client.get_job_status_lightweight(job_id)
        if status_resp.status_code != 200:
            return make_response(jsonify({"error": "Failed to poll job status."}), 502)

        status_data: dict[str, Any] = status_resp.json()
        status: str = status_data.get("status", "")
        progress: int = int(status_data.get("progress", 0))
        step: str = str(status_data.get("step", ""))

        session["job_progress"] = progress
        session["job_step"] = step

        if status == "done":
            return _handle_done_job(client, job_id, status_data)
        if status == "error":
            return _handle_error_job(client, job_id, status_data)

        return jsonify(
            {
                "status": status,
                "progress": progress,
                "step": step,
                "done": False,
            }
        )

    except Exception:
        logger.exception("Status poll error")
        return make_response(jsonify({"error": "Status poll error."}), 503)


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
    except Exception:
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
    except Exception:
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


def _parse_preview(content: bytes, filename: str) -> list[dict[str, Any]]:
    """Parse the first 20 rows of an uploaded file for the overview tab.

    Args:
        content:  Raw file bytes (CSV or XLSX).
        filename: Original filename used to determine the parser.

    Returns:
        List of row dicts (at most 20 rows), or an empty list on parse error.
    """
    try:
        buffer = io.BytesIO(content)
        if filename.lower().endswith(".xlsx"):
            df = pandas.read_excel(buffer).head(20)
        else:
            df = pandas.read_csv(buffer).head(20)
        return df.astype(str).to_dict(orient="records")
    except Exception:
        return []
