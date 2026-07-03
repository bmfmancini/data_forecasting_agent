"""Main FastAPI application module for the Data Forecaster API.

Route handlers are kept thin — business logic lives in the
``backend/services/`` package (file storage, job queue, pipeline
orchestration, chat, and RAG).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import core.config as settings
from auth.api_key_db import (
    create_api_user,
    create_first_user,
    delete_api_user,
    has_any_users,
    has_bootstrap_user,
    init_db as init_api_key_db,
    list_api_users,
    rotate_api_key,
    set_user_enabled,
)
from auth.dependency import require_api_key
from core.config import set_api_key_enabled
from core.logging_config import get_logger
from schemas import (
    APIKeyRotatedResponse,
    APIUserCreateRequest,
    APIUserCreatedResponse,
    APIUserResponse,
    APIUserToggleRequest,
    AuthStatusResponse,
    BootstrapRequest,
    BootstrapResponse,
    ChatRequest,
    ChatResponse,
    JobStatusResponse,
    JobSubmitResponse,
    PreflightResponse,
    UploadResponse,
)
from services.chat_service import chat_general, chat_with_data
from services.file_service import get_file, store_file
from services.job_service import (
    create_job,
    get_job,
    get_job_status_only,
    init_job_queue,
    is_queue_ready,
    job_worker,
)
from utils.data_parser import parse_upload
from utils.preflight import run_preflight_checks

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Manage the lifecycle of the job worker queue.

    Initialises the API key database, re-enables auth if users from a
    prior deployment exist, starts the job queue, and cancels the worker
    on shutdown.
    """
    logger.info("Initializing API key database…")
    init_api_key_db()
    if has_any_users():
        set_api_key_enabled(True)
        logger.info("API users found — auth enabled.")
    else:
        logger.info("No API users — auth disabled (open mode).")
    init_job_queue()
    worker_task = asyncio.create_task(job_worker())
    yield
    worker_task.cancel()
    await worker_task


app = FastAPI(title="Data Forecaster API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "http://frontend:5000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=[
        "X-API-Username",
        "X-API-Key",
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
        media_type="application/json",
        status_code=500,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


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


@app.post("/api-users/bootstrap", response_model=BootstrapResponse)
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


@app.post(
    "/upload",
    responses={
        400: {"description": "Invalid file content, size, or format"},
        500: {"description": "File parsing failed"},
    },
)
async def upload_file(
    file: Annotated[UploadFile, File(...)],
    _user: Annotated[dict, Depends(require_api_key)],
) -> UploadResponse:
    """Handle file uploads, validate types, and store in memory."""
    logger.info(
        "POST /upload  filename=%s  content_type=%s", file.filename, file.content_type
    )

    # ── Validate content-type ─────────────────────────────────────────────────
    if file.content_type not in settings.ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported content type '{file.content_type}'. "
                f"Allowed: {settings.ALLOWED_MIME_TYPES}"
            ),
        )

    # ── Validate extension ────────────────────────────────────────────────────
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File extension '.{ext}' not allowed. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    # ── Read & validate size ──────────────────────────────────────────────────
    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File too large ({len(contents) // 1024} KB). "
                f"Maximum allowed: {settings.MAX_UPLOAD_MB} MB."
            ),
        )

    # ── Validate file signature (magic bytes) ─────────────────────────────────
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if ext == "xlsx" and contents[:4] != b"PK\x03\x04":
        raise HTTPException(
            status_code=400,
            detail="File content does not match XLSX format (expected ZIP signature).",
        )
    if ext == "csv":
        try:
            contents[:4096].decode("utf-8")
        except UnicodeDecodeError:
            try:
                contents[:4096].decode("latin-1")
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="File content does not appear to be a valid text/CSV file.",
                ) from None

    # ── Parse ─────────────────────────────────────────────────────────────────
    try:
        df, date_col, value_col, freq = parse_upload(
            contents, file.filename or "upload.csv"
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during file parsing")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while parsing the file.",
        ) from exc

    # ── Store & return ────────────────────────────────────────────────────────
    file_id = store_file(df, date_col, value_col, freq, file.filename)

    return UploadResponse(
        file_id=file_id,
        filename=file.filename or "",
        rows=len(df),
        columns=df.columns.tolist(),
        detected_date_col=date_col,
        detected_value_col=value_col,
        detected_frequency=freq,
    )


@app.post(
    "/preflight",
    responses={
        400: {"description": "Preflight check failed"},
        404: {"description": "Session data not found"},
    },
)
async def preflight_check(
    request: AnalyzeRequest,
    _user: Annotated[dict, Depends(require_api_key)],
) -> PreflightResponse:
    """Run data quality checks before starting the full analysis pipeline."""
    stored = get_file(request.file_id)
    if not stored:
        raise HTTPException(
            status_code=404,
            detail="Session expired or data cleared from memory. Please re-upload your file.",
        )

    date_col = request.date_col or stored["date_col"]
    value_col = request.value_col or stored["value_col"]

    try:
        return run_preflight_checks(
            stored["df"], date_col, value_col, request.forecast_horizon
        )
    except Exception as exc:
        logger.exception("Preflight failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/analyze",
    status_code=202,
    responses={
        404: {"description": "Session data not found"},
        503: {"description": "Background worker service not ready"},
    },
)
def analyze(
    request: AnalyzeRequest,
    _user: Annotated[dict, Depends(require_api_key)],
) -> JobSubmitResponse:
    """Enqueue a forecasting analysis job."""
    logger.info(
        "POST /analyze  file_id=%s horizon=%d",
        request.file_id,
        request.forecast_horizon,
    )

    if not is_queue_ready():
        raise HTTPException(status_code=503, detail="Service not ready.")

    stored = get_file(request.file_id)
    if stored is None:
        raise HTTPException(
            status_code=404, detail=f"File ID '{request.file_id}' not found."
        )

    date_col = request.date_col or stored["date_col"]
    value_col = request.value_col or stored["value_col"]

    job_id = create_job(
        file_id=request.file_id,
        date_col=date_col,
        value_col=value_col,
        forecast_horizon=request.forecast_horizon,
        forced_model=request.forced_model,
        user_prompt=request.user_prompt,
        preflight_options=request.preflight_options,
    )
    return JobSubmitResponse(job_id=job_id, status="pending")


# ── Job Status & Results ──────────────────────────────────────────────────────


@app.get("/jobs/{job_id}/status", responses={404: {"description": "Job ID not found"}})
def get_job_status_lightweight(
    job_id: str,
    _user: Annotated[dict, Depends(require_api_key)],
) -> dict[str, Any]:
    """Return only status/progress/step for lightweight polling.

    This endpoint excludes the full ``result`` payload (which can be
    large with base64 charts) so polling is fast.  Use
    ``GET /jobs/{job_id}`` to retrieve the complete results once the
    job is done.
    """
    status = get_job_status_only(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return {"job_id": job_id, **status}


@app.get("/jobs/{job_id}", responses={404: {"description": "Job ID not found"}})
def get_job_status(
    job_id: str,
    _user: Annotated[dict, Depends(require_api_key)],
) -> JobStatusResponse:
    """Retrieve the full status and results of a specific background job."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        step=job["step"],
        result=job.get("result"),
        error=job.get("error"),
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
    request: ChatRequest,
    _user: Annotated[dict, Depends(require_api_key)],
) -> ChatResponse:
    """Allow users to chat with the agent about the uploaded data and results."""
    if request.file_id:
        stored = get_file(request.file_id)
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


@app.get("/api-users", response_model=list[APIUserResponse])
def api_users_list(
    _user: Annotated[dict, Depends(require_api_key)],
) -> list[dict[str, Any]]:
    """List all API key users (never includes key hashes)."""
    return list_api_users()


@app.post("/api-users", response_model=APIUserCreatedResponse, status_code=201)
def api_users_create(
    request: APIUserCreateRequest,
    _user: Annotated[dict, Depends(require_api_key)],
) -> dict[str, Any]:
    """Create a new API user and return the plaintext key once."""
    try:
        plaintext_key: str = create_api_user(
            username=request.username,
            description=request.description,
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
)
def api_users_rotate(
    user_id: int,
    _user: Annotated[dict, Depends(require_api_key)],
) -> dict[str, Any]:
    """Rotate an API user's key and return the new plaintext key once."""
    try:
        plaintext_key: str = rotate_api_key(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"user_id": user_id, "api_key": plaintext_key}


@app.post("/api-users/{user_id}/toggle", response_model=APIUserResponse)
def api_users_toggle(
    user_id: int,
    request: APIUserToggleRequest,
    _user: Annotated[dict, Depends(require_api_key)],
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


@app.delete("/api-users/{user_id}", status_code=204, response_class=Response)
def api_users_delete(
    user_id: int,
    _user: Annotated[dict, Depends(require_api_key)],
) -> Response:
    """Permanently delete an API user."""
    try:
        delete_api_user(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)


@app.get("/api-users/bootstrap-status")
def api_users_bootstrap_status(
    _user: Annotated[dict, Depends(require_api_key)],
) -> dict[str, bool]:
    """Check whether a bootstrap API user still exists."""
    return {"has_bootstrap": has_bootstrap_user()}
