"""Main FastAPI application module for the Data Forecaster API.

Route handlers are kept thin — business logic lives in the
``backend/services/`` package (file storage, job queue, pipeline
orchestration, chat, and RAG).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Annotated

import aiofiles
import anyio
import httpx

from fastapi import (
    Body,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware

import core.config as settings
from auth.application_identity import verify_application_user_signature
from auth.api_key_db import (
    create_api_user,
    create_first_user,
    delete_api_user,
    has_any_users,
    has_bootstrap_user,
    list_api_users,
    reconcile_service_user,
    rotate_api_key,
    set_user_admin,
    set_user_enabled,
)
from auth.dependency import require_admin_api_key, require_api_key
from core.config import (
    GOOGLE_API_KEY,
    OLLAMA_API_KEY,
    OLLAMA_MODEL,
    USE_OLLAMA,
    USE_OLLAMA_CLOUD,
    set_api_key_enabled,
)
from core.database import init_database
from core.logging_config import get_logger
from schemas import (
    APIKeyRotatedResponse,
    APIUserCreateRequest,
    APIUserCreatedResponse,
    APIUserResponse,
    APIUserSetAdminRequest,
    APIUserToggleRequest,
    AnalyzeRequest,
    AuthStatusResponse,
    BootstrapRequest,
    BootstrapResponse,
    ChatRequest,
    ChatResponse,
    DeletedJobsResponse,
    ForecastJobQueueItem,
    ForecastJobSettings,
    JobStatusResponse,
    JobSubmitResponse,
    PreflightResponse,
    UploadResponse,
    UserJobQueueItem,
)
from services.chat_service import chat_general, chat_with_data
from services.file_service import get_file, init_storage, store_file
from services.job_service import (
    QueuedJobLimitError,
    create_job,
    clear_terminal_jobs,
    cleanup_terminal_jobs,
    get_job,
    get_job_settings,
    get_job_status_only,
    init_job_queue,
    is_queue_ready,
    job_worker,
    list_jobs_for_user,
    list_recent_jobs,
    request_cancel,
    update_job_settings,
)
from utils.data_parser import parse_upload, parse_upload_from_path
from utils.preflight import run_preflight_checks

logger = get_logger(__name__)

_JSON_MEDIA_TYPE: str = "application/json"
_JOB_CLEANUP_INTERVAL_SECONDS: int = 24 * 60 * 60


def _application_user_id_header(
    x_application_user_id: Annotated[
        str | None,
        Header(alias="X-Application-User-ID"),
    ] = None,
    x_application_user_signature: Annotated[
        str | None,
        Header(alias="X-Application-User-Signature"),
    ] = None,
) -> int | None:
    """Extract and authenticate the delegated application-user identity.

    This header is set by the Flask proxy from ``current_user.id`` (server-side)
    and is never controlled by the browser.  It scopes job operations to the
    authenticated frontend user, since multiple app users may share the same
    backend API credential.

    Args:
        x_application_user_id: The raw application-user header value.
        x_application_user_signature: HMAC signature supplied by the frontend.

    Returns:
        The parsed integer user ID, or ``None`` if the header is absent.

    Raises:
        HTTPException: If the identity headers are incomplete, malformed, or
            fail signature verification.
    """
    if not x_application_user_id and not x_application_user_signature:
        return None
    if not x_application_user_id or not x_application_user_signature:
        raise HTTPException(
            status_code=400,
            detail="Both application-user identity headers are required.",
        )
    try:
        application_user_id = int(x_application_user_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="X-Application-User-ID must be an integer.",
        ) from exc
    if application_user_id < 1:
        raise HTTPException(
            status_code=400,
            detail="X-Application-User-ID must be a positive integer.",
        )
    if not verify_application_user_signature(
        application_user_id,
        x_application_user_signature,
        settings.APPLICATION_IDENTITY_SECRET,
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid application-user identity signature.",
        )
    return application_user_id


async def _cleanup_job_history() -> None:
    """Run retention cleanup daily without blocking API request handling."""
    while True:
        await asyncio.sleep(_JOB_CLEANUP_INTERVAL_SECONDS)
        deleted_count = await asyncio.to_thread(cleanup_terminal_jobs)
        if deleted_count:
            logger.info("Deleted %d expired forecast job records.", deleted_count)


def _service_credentials_configured() -> bool:
    """Return whether frontend service-user credentials are configured."""
    return bool(settings.FRONTEND_API_USERNAME and settings.FRONTEND_API_KEY)


def _reconcile_frontend_service_user() -> None:
    """Reconcile the frontend service user from environment configuration."""
    if not _service_credentials_configured():
        return

    try:
        reconciled = reconcile_service_user(
            settings.FRONTEND_API_USERNAME,
            settings.FRONTEND_API_KEY,
        )
    except ValueError as exc:
        logger.warning("Failed to reconcile frontend API user: %s", exc)
        return

    if reconciled:
        logger.info(
            "Frontend API user '%s' reconciled from env.",
            settings.FRONTEND_API_USERNAME,
        )


def _create_frontend_service_user_from_env() -> None:
    """Create the first API user from frontend service env vars when present."""
    if not _service_credentials_configured():
        logger.info("No API users — auth disabled (open mode).")
        return

    try:
        create_first_user(
            username=settings.FRONTEND_API_USERNAME,
            api_key=settings.FRONTEND_API_KEY,
        )
    except ValueError as exc:
        logger.warning("Failed to auto-create frontend API user: %s", exc)
        logger.info("No API users — auth disabled (open mode).")
        return

    set_api_key_enabled(True)
    logger.info(
        "Frontend API user '%s' auto-created from env vars — auth enabled.",
        settings.FRONTEND_API_USERNAME,
    )
    if settings.FRONTEND_API_KEY == "frontend":
        logger.warning(
            "SECURITY: The frontend API key is the default 'frontend'. "
            "Rotate it via the admin panel and update the stored "
            "frontend credentials before production use."
        )
        return

    logger.warning(
        "The initial API key was sourced from the FRONTEND_API_KEY "
        "env var.  For production security, rotate this key via the "
        "admin panel and update the stored frontend credentials."
    )


def _configure_startup_auth() -> None:
    """Enable or bootstrap API authentication during startup."""
    if has_any_users():
        _reconcile_frontend_service_user()
        set_api_key_enabled(True)
        logger.info("API users found — auth enabled.")
        return

    _create_frontend_service_user_from_env()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Manage the lifecycle of the job worker queue.

    Initialises the API key database, re-enables auth if users from a
    prior deployment exist, starts the job queue, and cancels the worker
    on shutdown.
    """
    logger.info("Initializing API key database…")
    init_database()
    cleanup_terminal_jobs()
    init_storage()
    _configure_startup_auth()
    init_job_queue()
    worker_tasks = [
        asyncio.create_task(job_worker()) for _ in range(settings.MAX_CONCURRENT_JOBS)
    ]
    cleanup_task = asyncio.create_task(_cleanup_job_history())
    yield
    for worker_task in worker_tasks:
        worker_task.cancel()
    cleanup_task.cancel()
    await asyncio.gather(*worker_tasks, cleanup_task, return_exceptions=True)


app = FastAPI(title="Data Forecaster API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=[
        "X-API-Username",
        "X-API-Key",
        "X-Application-User-ID",
        "X-Application-User-Signature",
        "Content-Type",
        "X-Admin-Key",
    ],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next: Any) -> Any:
    """Add standard security headers to every response.

    Sets ``X-Content-Type-Options``, ``X-Frame-Options``, and
    ``Strict-Transport-Security`` to mitigate MIME sniffing, clickjacking,
    and protocol downgrade attacks.
    """
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> Response:
    """Catch any uncaught exception and return a generic 500 response.

    Logs the full exception server-side so details are available for
    debugging without leaking them to the client.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return Response(
        content='{"detail": "Internal server error."}',
        media_type=_JSON_MEDIA_TYPE,
        status_code=500,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/auth-check")
def auth_check(
    current_user: Annotated[dict[str, Any], Depends(require_api_key)],
) -> dict[str, Any]:
    """Validate API credentials without performing a privileged operation."""
    return {
        "authenticated": bool(current_user),
        "username": current_user.get("username"),
        "is_admin": bool(current_user.get("is_admin")),
    }


@app.get("/llm-health")
def llm_health_endpoint() -> dict[str, Any]:
    """Return a minimal LLM liveness status.

    The public response is intentionally limited to ``llm_configured``
    and ``llm_reachable`` booleans so that provider names, configuration
    details, and error messages are not exposed to unauthenticated
    callers.  The full :func:`llm_health` result is available for
    internal/server-side use only.

    Returns:
        A JSON dict with keys ``llm_configured`` and ``llm_reachable``.
    """
    full = llm_health()
    return {
        "llm_configured": full.get("llm_configured", False),
        "llm_reachable": full.get("llm_reachable", False),
    }


async def _check_ollama_reachable() -> bool:
    """Return whether the configured Ollama endpoint responds.

    Handles both Ollama Cloud (with optional API key) and local Ollama.
    """
    try:
        if USE_OLLAMA_CLOUD:
            ollama_url = f"{settings.OLLAMA_BASE_URL}/api/version"
            headers = {"Content-Type": _JSON_MEDIA_TYPE}
            if OLLAMA_API_KEY:
                headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
            async with httpx.AsyncClient() as client:
                response = await client.get(ollama_url, headers=headers)
                return response.status_code == 200
        ollama_url = f"{settings.OLLAMA_BASE_URL}/api/tags"
        async with httpx.AsyncClient() as client:
            response = await client.get(ollama_url)
            return response.status_code == 200
    except httpx.HTTPError:
        return False


async def _check_gemini_reachable() -> bool:
    """Return whether the Gemini API responds to a lightweight probe."""
    try:
        gemini_url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{settings.GEMINI_MODEL}:countTokens"
        )
        headers = {"Content-Type": _JSON_MEDIA_TYPE}
        async with httpx.AsyncClient() as client:
            response = await client.post(
                gemini_url,
                json={"contents": [{"parts": [{"text": "ping"}]}]},
                headers=headers,
                params={"key": GOOGLE_API_KEY},
            )
            return response.status_code == 200
    except httpx.HTTPError:
        return False


def llm_health() -> dict[str, Any]:
    """Check LLM connectivity and configuration.

    Returns:
        dict: A dictionary with keys:
            - "llm_configured": bool indicating if an LLM provider is configured.
            - "llm_reachable": bool indicating if the LLM is reachable.
            - "llm_provider": str indicating the configured LLM provider ("gemini" or "ollama").
            - "error": str containing error message if any, otherwise None.
    """
    result: dict[str, Any] = {
        "llm_configured": False,
        "llm_reachable": False,
        "llm_provider": None,
        "error": None,
    }

    if USE_OLLAMA:
        result["llm_provider"] = "ollama"
        if not OLLAMA_MODEL:
            result["error"] = "OLLAMA_MODEL is not set."
            return result
        result["llm_configured"] = True
        result["llm_reachable"] = asyncio.run(_check_ollama_reachable())
        if not result["llm_reachable"]:
            result["error"] = "Ollama server is not reachable."
    elif GOOGLE_API_KEY:
        result["llm_provider"] = "gemini"
        result["llm_configured"] = True
        result["llm_reachable"] = asyncio.run(_check_gemini_reachable())
        if not result["llm_reachable"]:
            result["error"] = "Gemini API is not reachable."
    else:
        result["error"] = (
            "No LLM provider configured. Either set USE_OLLAMA=true with a "
            "running Ollama instance, or set GOOGLE_API_KEY for Gemini."
        )

    return result


# ── Auth Status & Bootstrap (unauthenticated, guarded) ────────────────────────


@app.get("/auth-status", response_model=AuthStatusResponse)
def auth_status() -> dict[str, Any]:
    """Return whether API auth is enabled and whether any users exist.

    This endpoint is unauthenticated so the frontend can determine
    whether to show the "Enable Authentication" workflow.  It reveals
    only boolean flags — no sensitive data.
    """
    return {
        "auth_enabled": settings.API_KEY_ENABLED,
        "has_users": has_any_users(),
    }


@app.post(
    "/api-users/bootstrap",
    response_model=BootstrapResponse,
    responses={
        400: {"description": "Invalid username or API key"},
        403: {"description": "ADMIN_API_KEY missing or invalid"},
        409: {"description": "API users already exist"},
    },
)
def api_users_bootstrap(
    request: BootstrapRequest,
    http_request: Request,
) -> dict[str, Any]:
    """Create the first API user and enable authentication.

    This is a one-time setup endpoint protected by the ``ADMIN_API_KEY``
    deployment secret (sent via the ``X-Admin-Key`` header).  It only
    succeeds when:

    - ``ADMIN_API_KEY`` is set in the backend environment.
    - The supplied ``X-Admin-Key`` header matches.
    - No API users exist yet (bootstrap is one-time only).

    On success, creates the user with the admin-supplied username and
    key, enables ``API_KEY_ENABLED``, and returns the user dict.

    Raises:
        HTTPException: 403 when the admin key is missing or mismatched.
        HTTPException: 409 when users already exist (bootstrap expired).
    """
    if not settings.ADMIN_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="ADMIN_API_KEY is not set on the backend. "
            "Configure it in the backend .env to use bootstrap.",
        )
    supplied_key: str | None = http_request.headers.get("X-Admin-Key")
    if not supplied_key or supplied_key != settings.ADMIN_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing admin key.",
        )

    if has_any_users():
        raise HTTPException(
            status_code=409,
            detail="API users already exist — bootstrap is no longer available.",
        )

    try:
        user: dict[str, Any] = create_first_user(
            username=request.username,
            api_key=request.api_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    set_api_key_enabled(True)
    logger.info(
        "API auth enabled by bootstrap. User '%s' created.",
        request.username,
    )

    return {"user": user, "auth_enabled": True}


# ── File Upload & Analysis ────────────────────────────────────────────────────

# Chunk size for streaming uploads to disk (1 MB).
_UPLOAD_CHUNK_SIZE: int = 1024 * 1024


def _validate_upload_metadata(file: UploadFile, file_size: int) -> str:
    """Validate upload metadata (content type, extension, size).

    Args:
        file:      The uploaded file object (metadata only, not contents).
        file_size: Size of the streamed file in bytes.

    Returns:
        The lower-cased file extension.

    Raises:
        HTTPException: 400 when content type, extension, or size is invalid.
    """
    logger.info(
        "POST /upload  filename=%s  content_type=%s  size=%d",
        file.filename,
        file.content_type,
        file_size,
    )

    if file.content_type not in settings.ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported content type '{file.content_type}'. "
                f"Allowed: {settings.ALLOWED_MIME_TYPES}"
            ),
        )

    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File extension '.{ext}' not allowed. "
                f"Allowed: {settings.ALLOWED_EXTENSIONS}"
            ),
        )

    if file_size > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File too large ({file_size // 1024} KB). "
                f"Maximum allowed: {settings.MAX_UPLOAD_MB} MB."
            ),
        )

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return ext


def _validate_file_signature(path: str, ext: str) -> None:
    """Validate file magic bytes from a disk path.

    Args:
        path: Path to the temporary file on disk.
        ext:  Lower-cased file extension.

    Raises:
        HTTPException: 400 when the file content does not match the
            expected format.
    """
    with open(path, "rb") as fh:
        header = fh.read(4096)

    if ext == "xlsx" and header[:4] != b"PK\x03\x04":
        raise HTTPException(
            status_code=400,
            detail="File content does not match XLSX format (expected ZIP signature).",
        )

    if ext == "csv":
        try:
            header.decode("utf-8")
        except UnicodeDecodeError:
            try:
                header.decode("latin-1")
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="File content does not appear to be a valid text/CSV file.",
                ) from None

    if ext == "json":
        try:
            header.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File content does not appear to be valid UTF-8 JSON.",
            ) from None


async def _stream_upload_to_disk(file: UploadFile) -> tuple[str, int]:
    """Stream an upload to a temporary file on disk, returning path and size.

    Reads the upload in fixed-size chunks to avoid loading the entire
    file into memory.  Synchronous file operations are offloaded to a
    thread pool via ``anyio.to_thread.run_sync`` to avoid blocking the
    async event loop on large uploads.

    Args:
        file: The FastAPI UploadFile to stream.

    Returns:
        A tuple of ``(temp_file_path, total_bytes_written)``.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".upload")
    os.close(tmp_fd)  # Close the raw fd; aiofiles opens its own handle.
    total = 0
    try:
        async with aiofiles.open(tmp_path, "wb") as fh:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                await fh.write(chunk)
                total += len(chunk)
    except (OSError, RuntimeError):
        await anyio.to_thread.run_sync(os.unlink, tmp_path)
        raise
    return tmp_path, total


@app.post(
    "/upload",
    responses={
        400: {"description": "Invalid file content, size, or format"},
        503: {"description": "File storage is full and cannot be evicted"},
        500: {"description": "File parsing failed"},
    },
)
async def upload_file(
    file: Annotated[UploadFile, File(...)],
    _user: Annotated[dict, Depends(require_api_key)],
) -> UploadResponse:
    """Handle file uploads, validate types, and store on disk.

    Streams the upload to a temporary file to avoid loading the entire
    file into memory, then parses from disk.
    """
    tmp_path, file_size = await _stream_upload_to_disk(file)
    try:
        ext = _validate_upload_metadata(file, file_size)
        _validate_file_signature(tmp_path, ext)

        try:
            df, date_col, value_col, freq = parse_upload_from_path(
                tmp_path, file.filename or "upload.csv"
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            logger.exception("Unexpected error during file parsing")
            raise HTTPException(
                status_code=500,
                detail="An internal error occurred while parsing the file.",
            ) from exc

        try:
            file_id = store_file(
                df,
                date_col,
                value_col,
                freq,
                file.filename,
                owner_id=_user.get("id"),
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return UploadResponse(
            file_id=file_id,
            filename=file.filename or "",
            rows=len(df),
            columns=df.columns.tolist(),
            detected_date_col=date_col,
            detected_value_col=value_col,
            detected_frequency=freq,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            logger.warning("Failed to clean up temp file: %s", tmp_path)


@app.post(
    "/preflight",
    responses={
        400: {"description": "Preflight check failed"},
        404: {"description": "Session data not found"},
    },
)
async def preflight_check(
    body: Annotated[AnalyzeRequest, Body()],
    _user: Annotated[dict, Depends(require_api_key)],
) -> PreflightResponse:
    """Run data quality checks before starting the full analysis pipeline."""
    stored = get_file(body.file_id, requester=_user)
    if not stored:
        raise HTTPException(
            status_code=404,
            detail="Session expired or data cleared from memory. Please re-upload your file.",
        )

    date_col = body.date_col or stored["date_col"]
    value_col = body.value_col or stored["value_col"]

    try:
        return run_preflight_checks(
            stored["df"], date_col, value_col, body.forecast_horizon
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.exception("Preflight failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/analyze",
    status_code=202,
    responses={
        404: {"description": "Session data not found"},
        429: {"description": "Per-user queued-job limit exceeded"},
        503: {"description": "Background worker service not ready"},
    },
)
def analyze(
    body: Annotated[AnalyzeRequest, Body()],
    _user: Annotated[dict, Depends(require_api_key)],
    app_user_id: Annotated[int | None, Depends(_application_user_id_header)] = None,
) -> JobSubmitResponse:
    """Enqueue a forecasting analysis job."""
    logger.info(
        "POST /analyze  file_id=%s horizon=%d",
        body.file_id,
        body.forecast_horizon,
    )

    if not is_queue_ready():
        raise HTTPException(status_code=503, detail="Service not ready.")

    if body.application_user_id != app_user_id:
        raise HTTPException(
            status_code=403,
            detail="Application-user identity does not match the request body.",
        )

    stored = get_file(body.file_id, requester=_user)
    if stored is None:
        raise HTTPException(
            status_code=404, detail=f"File ID '{body.file_id}' not found."
        )

    date_col = body.date_col or stored["date_col"]
    value_col = body.value_col or stored["value_col"]

    try:
        job_id = create_job(
            file_id=body.file_id,
            date_col=date_col,
            value_col=value_col,
            forecast_horizon=body.forecast_horizon,
            forced_model=body.forced_model,
            user_prompt=body.user_prompt,
            preflight_options=body.preflight_options,
            owner_id=_user.get("id"),
            application_user_id=body.application_user_id,
            application_username=body.application_username,
            application_user_is_admin=(
                body.application_user_is_admin and bool(_user.get("is_admin"))
            ),
            report_name=body.report_name,
            source_filename=body.source_filename,
            custom_settings=body.custom_settings,
        )
    except QueuedJobLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JobSubmitResponse(job_id=job_id, status="pending")


# ── Job Status & Results ──────────────────────────────────────────────────────


@app.get(
    "/jobs/recent",
    response_model=list[ForecastJobQueueItem],
    responses={403: {"description": "Authenticated user is not an admin"}},
)
def recent_jobs(
    _user: Annotated[dict, Depends(require_admin_api_key)],
) -> list[dict[str, Any]]:
    """Return the 25 most recent jobs for the administrator queue."""
    return list_recent_jobs()


@app.delete(
    "/jobs/terminal",
    response_model=DeletedJobsResponse,
    responses={403: {"description": "Authenticated user is not an admin"}},
)
def terminal_jobs_clear(
    _user: Annotated[dict, Depends(require_admin_api_key)],
) -> dict[str, int]:
    """Delete completed and failed forecast jobs at an administrator's request."""
    return {"deleted_count": clear_terminal_jobs()}


@app.get(
    "/job-settings",
    response_model=ForecastJobSettings,
    responses={403: {"description": "Authenticated user is not an admin"}},
)
def job_settings_get(
    _user: Annotated[dict, Depends(require_admin_api_key)],
) -> dict[str, Any]:
    """Return administrator-managed scheduler and retention settings."""
    return get_job_settings()


@app.put(
    "/job-settings",
    response_model=ForecastJobSettings,
    responses={403: {"description": "Authenticated user is not an admin"}},
)
def job_settings_update(
    job_settings: ForecastJobSettings,
    _user: Annotated[dict, Depends(require_admin_api_key)],
) -> dict[str, Any]:
    """Update scheduler and retention settings, then clean expired history."""
    updated = update_job_settings(
        job_settings.max_running_jobs_per_user,
        job_settings.max_queued_jobs_per_user,
        job_settings.retention_days,
        job_settings.cleanup_enabled,
    )
    cleanup_terminal_jobs()
    return updated


@app.get(
    "/jobs/mine",
    response_model=list[UserJobQueueItem],
    responses={403: {"description": "Authenticated user is not an admin"}},
)
def my_jobs(
    _user: Annotated[dict, Depends(require_api_key)],
    app_user_id: Annotated[int | None, Depends(_application_user_id_header)] = None,
) -> list[dict[str, Any]]:
    """Return the most recent jobs for the authenticated application user.

    The ``X-Application-User-ID`` header is set by the Flask proxy from
    ``current_user.id`` and scopes the result to the frontend user's own
    jobs.  When the header is absent, a 400 error is raised.
    """
    if app_user_id is None:
        raise HTTPException(
            status_code=400,
            detail="X-Application-User-ID header is required.",
        )
    jobs = list_jobs_for_user(
        app_user_id,
        backend_owner_id=_user.get("id"),
    )
    result: list[dict[str, Any]] = []
    for job in jobs:
        status = job.get("status", "")
        can_cancel = status in ("pending", "running", "cancelling")
        result.append(
            {
                "job_id": job["job_id"],
                "report_name": job.get("report_name") or "",
                "status": status,
                "progress": int(job.get("progress", 0)),
                "step": job.get("step", ""),
                "queued_at": job.get("queued_at", ""),
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
                "error": job.get("error"),
                "can_cancel": can_cancel,
                "forecast_horizon": int(job.get("forecast_horizon", 0)),
                "forced_model": job.get("forced_model"),
            }
        )
    return result


@app.delete(
    "/jobs/{job_id}",
    responses={
        404: {"description": "Job ID not found or not owned by the user"},
        409: {"description": "Job is already terminal"},
    },
)
def cancel_job(
    job_id: str,
    _user: Annotated[dict, Depends(require_api_key)],
    app_user_id: Annotated[int | None, Depends(_application_user_id_header)] = None,
) -> dict[str, str]:
    """Request cooperative cancellation of a forecast job.

    Pending jobs are cancelled immediately.  Running jobs transition to
    ``cancelling`` and are cancelled at the next pipeline stage boundary.
    Cancellation is idempotent for ``cancelling`` jobs.  Terminal jobs
    return 409.
    """
    result = request_cancel(job_id, requester=_user, application_user_id=app_user_id)
    if result == "not_found":
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if result == "conflict":
        raise HTTPException(status_code=409, detail="Job is already complete.")
    return {"job_id": job_id, "cancel_status": result}


@app.get("/jobs/{job_id}/status", responses={404: {"description": "Job ID not found"}})
def get_job_status_lightweight(
    job_id: str,
    _user: Annotated[dict, Depends(require_api_key)],
    app_user_id: Annotated[int | None, Depends(_application_user_id_header)] = None,
) -> dict[str, Any]:
    """Return only status/progress/step for lightweight polling.

    This endpoint excludes the full ``result`` payload (which can be
    large with base64 charts) so polling is fast.  Use
    ``GET /jobs/{job_id}`` to retrieve the complete results once the
    job is done.
    """
    status = get_job_status_only(
        job_id, requester=_user, application_user_id=app_user_id
    )
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return {"job_id": job_id, **status}


@app.get("/jobs/{job_id}", responses={404: {"description": "Job ID not found"}})
def get_job_status(
    job_id: str,
    _user: Annotated[dict, Depends(require_api_key)],
    app_user_id: Annotated[int | None, Depends(_application_user_id_header)] = None,
) -> JobStatusResponse:
    """Retrieve the full status and results of a specific background job."""
    job = get_job(job_id, requester=_user, application_user_id=app_user_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        step=job["step"],
        result=job.get("result"),
        error=job.get("error"),
        report_name=str(job.get("report_name") or ""),
        source_filename=str(job.get("source_filename") or ""),
        forecast_horizon=int(job.get("forecast_horizon") or 0),
        custom_settings_json=str(job.get("custom_settings_json") or "[]"),
    )


# ── Chat ──────────────────────────────────────────────────────────────────────


@app.post(
    "/chat",
    responses={
        404: {"description": "Session data not found"},
        500: {"description": "Chat agent processing error"},
    },
)
async def chat_explorer(
    request: Annotated[ChatRequest, Body()],
    _user: Annotated[dict, Depends(require_api_key)],
) -> ChatResponse:
    """Allow users to chat with the agent about the uploaded data and results."""
    if request.file_id:
        stored = get_file(request.file_id, requester=_user)
        if not stored:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Chat session lost: the associated data is no longer "
                    "in the server memory. Please re-run the analysis."
                ),
            )

        try:
            response = await asyncio.to_thread(
                chat_with_data,
                query=request.query,
                df=stored["df"],
                file_id=request.file_id,
                owner_id=_user.get("id"),
                chroma_persist_dir=settings.CHROMA_PERSIST_DIR,
            )
            return ChatResponse(**response)
        except Exception as exc:
            logger.exception("Chat exploration failed")
            raise HTTPException(
                status_code=500,
                detail="An internal error occurred while processing the chat request.",
            ) from exc
    else:
        try:
            response = await asyncio.to_thread(
                chat_general,
                query=request.query,
                chroma_persist_dir=settings.CHROMA_PERSIST_DIR,
            )
            return ChatResponse(**response)
        except Exception:
            logger.exception("General chat failed")
            return ChatResponse(
                answer=(
                    "I encountered an error while processing your request. "
                    "Please try again later."
                )
            )


# ── API User Management Endpoints (admin) ─────────────────────────────────────


@app.get(
    "/api-users",
    response_model=list[APIUserResponse],
    responses={403: {"description": "Authenticated user is not an admin"}},
)
def api_users_list(
    _user: Annotated[dict, Depends(require_admin_api_key)],
) -> list[dict[str, Any]]:
    """List all API key users (never includes key hashes)."""
    return list_api_users()


@app.post(
    "/api-users",
    response_model=APIUserCreatedResponse,
    status_code=201,
    responses={
        409: {"description": "Username already exists"},
        500: {"description": "User created but not found"},
    },
)
def api_users_create(
    request: APIUserCreateRequest,
    _user: Annotated[dict, Depends(require_admin_api_key)],
) -> dict[str, Any]:
    """Create a new API user and return the plaintext key once."""
    try:
        plaintext_key: str = create_api_user(
            username=request.username,
            description=request.description,
            is_admin=request.is_admin,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    users: list[dict[str, Any]] = list_api_users()
    new_user: dict[str, Any] | None = next(
        (u for u in users if u["username"] == request.username), None
    )
    if new_user is None:
        raise HTTPException(status_code=500, detail="User created but not found.")

    return {"user": new_user, "api_key": plaintext_key}


@app.post(
    "/api-users/{user_id}/rotate",
    response_model=APIKeyRotatedResponse,
    responses={404: {"description": "API user not found"}},
)
def api_users_rotate(
    user_id: int,
    _user: Annotated[dict, Depends(require_admin_api_key)],
) -> dict[str, Any]:
    """Rotate an API user's key and return the new plaintext key once."""
    try:
        plaintext_key: str = rotate_api_key(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"user_id": user_id, "api_key": plaintext_key}


@app.post(
    "/api-users/{user_id}/toggle",
    response_model=APIUserResponse,
    responses={404: {"description": "API user not found"}},
)
def api_users_toggle(
    user_id: int,
    request: APIUserToggleRequest,
    _user: Annotated[dict, Depends(require_admin_api_key)],
) -> dict[str, Any]:
    """Enable or disable an API user."""
    try:
        set_user_enabled(user_id, request.enabled)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    user: dict[str, Any] | None = next(
        (u for u in list_api_users() if u["id"] == user_id), None
    )
    if user is None:
        raise HTTPException(status_code=404, detail="User not found after update.")
    return user


@app.post(
    "/api-users/{user_id}/admin",
    response_model=APIUserResponse,
    responses={404: {"description": "API user not found"}},
)
def api_users_set_admin(
    user_id: int,
    request: APIUserSetAdminRequest,
    _user: Annotated[dict, Depends(require_admin_api_key)],
) -> dict[str, Any]:
    """Promote or demote an API user."""
    try:
        set_user_admin(user_id, request.is_admin)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    user: dict[str, Any] | None = next(
        (u for u in list_api_users() if u["id"] == user_id), None
    )
    if user is None:
        raise HTTPException(status_code=404, detail="User not found after update.")
    return user


@app.delete(
    "/api-users/{user_id}",
    status_code=204,
    response_class=Response,
    responses={404: {"description": "API user not found"}},
)
def api_users_delete(
    user_id: int,
    _user: Annotated[dict, Depends(require_admin_api_key)],
) -> Response:
    """Permanently delete an API user."""
    try:
        delete_api_user(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)


@app.get(
    "/api-users/bootstrap-status",
    responses={403: {"description": "Authenticated user is not an admin"}},
)
def api_users_bootstrap_status(
    _user: Annotated[dict, Depends(require_admin_api_key)],
) -> dict[str, bool]:
    """Check whether a bootstrap API user still exists."""
    return {"has_bootstrap": has_bootstrap_user()}


# ── Chart PNG Export ─────────────────────────────────────────────────────────
# Chart PNGs are now pre-computed in the pipeline (see
# utils.visualization.chart_dict_to_png_b64) and stored as base64 strings
# in the AnalysisResponse.  The /api/charts/png endpoint has been removed
# — no network round-trips are needed for PDF export.
