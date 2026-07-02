"""Main FastAPI application module for the Data Forecaster API."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, cast, Annotated

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
from orchestrator import chat_with_data, index_analysis_results, run_pipeline
from schemas import (
    APIKeyRotatedResponse,
    APIUserCreateRequest,
    APIUserCreatedResponse,
    APIUserResponse,
    APIUserToggleRequest,
    AnalysisResponse,
    AnalyzeRequest,
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
from utils.data_parser import parse_upload
from utils.preflight import run_preflight_checks

logger = get_logger(__name__)

# ── In-memory stores ──────────────────────────────────────────────────────────
# Bounds for in-memory storage to prevent OOM restarts
MAX_FILES = 50
MAX_JOBS = 100

# { file_id: { df, date_col, value_col, freq, filename } }
_file_store: Dict[str, Dict[str, Any]] = {}

# { job_id: { status, progress, step, request, result, error } }
_job_store: Dict[str, Dict[str, Any]] = {}

# ── Job queue & worker ────────────────────────────────────────────────────────
JOB_QUEUE: asyncio.Queue[str] = cast(asyncio.Queue, None)


def _update_job_progress(job_id: str, pct: int, step: str) -> None:
    """Called from the pipeline thread; CPython GIL makes dict updates thread-safe."""
    job = _job_store.get(job_id)
    if job:
        job["progress"] = pct
        job["step"] = step


async def _job_worker() -> None:
    """Processes one job at a time from the FIFO queue."""
    while True:
        job_id: str = await JOB_QUEUE.get()
        job = _job_store.get(job_id)
        if job is None:
            JOB_QUEUE.task_done()
            continue

        job["status"] = "running"
        req = job["request"]
        stored = _file_store.get(req["file_id"])

        if stored is None:
            job["status"] = "error"
            job["step"] = "Error: uploaded file not found."
            job["error"] = f"File ID '{req['file_id']}' not found in store."
            JOB_QUEUE.task_done()
            continue

        def _run_pipeline_sync(r=req, s=stored, j_id=job_id) -> AnalysisResponse:
            return run_pipeline(
                df=s["df"],
                file_id=r["file_id"],
                date_col=r["date_col"],
                value_col=r["value_col"],
                freq=s["freq"],
                forecast_horizon=r["forecast_horizon"],
                forced_model=r["forced_model"],
                user_prompt=r.get("user_prompt"),
                preflight_options=r.get("preflight_options"),
                chroma_persist_dir=settings.CHROMA_PERSIST_DIR,
                progress_callback=lambda pct, step: _update_job_progress(
                    j_id, pct, step
                ),
            )

        try:
            result: AnalysisResponse = await asyncio.to_thread(_run_pipeline_sync)
            job["status"] = "done"
            job["progress"] = 100
            job["step"] = "Analysis complete."
            job["result"] = result.model_dump()

            # Requirement: Store the output results in the agent's memory
            await asyncio.to_thread(
                index_analysis_results,
                file_id=req["file_id"],
                analysis_result=result,
                chroma_persist_dir=settings.CHROMA_PERSIST_DIR,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception(
                "Pipeline failed for job_id=%s file_id=%s", job_id, req["file_id"]
            )
            job["status"] = "error"
            job["step"] = "Pipeline failed."
            job["error"] = str(exc)
        finally:
            JOB_QUEUE.task_done()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Manage the lifecycle of the job worker queue."""
    global JOB_QUEUE  # pylint: disable=global-statement
    logger.info("Initializing API key database…")
    init_api_key_db()
    # If users exist from a prior deployment, re-enable auth automatically
    # so restarts don't accidentally open the API.
    if has_any_users():
        set_api_key_enabled(True)
        logger.info("API users found — auth enabled.")
    else:
        logger.info("No API users — auth disabled (open mode).")
    JOB_QUEUE = asyncio.Queue()
    worker_task = asyncio.create_task(_job_worker())
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
    # Verify the deployment-time admin key
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

    # Bootstrap is one-time only
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

    # Enable auth at runtime
    set_api_key_enabled(True)
    logger.info(
        "API auth enabled by bootstrap. User '%s' created.",
        request.username,
    )

    return {"user": user, "auth_enabled": True}


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
    if file.content_type not in settings.ALLOWED_MIME_TYPES + [
        "application/octet-stream"
    ]:
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
    if ext == "xlsx" and not contents[:4] == b"PK\x03\x04":
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
    if len(_file_store) >= MAX_FILES:
        # Evict oldest file to maintain memory stability and prevent OOM
        oldest_file = next(iter(_file_store))
        _file_store.pop(oldest_file)

    file_id = str(uuid.uuid4())
    _file_store[file_id] = {
        "df": df,
        "date_col": date_col,
        "value_col": value_col,
        "freq": freq,
        "filename": file.filename,
    }
    logger.info(
        "File stored: file_id=%s rows=%d date_col=%s value_col=%s freq=%s",
        file_id,
        len(df),
        date_col,
        value_col,
        freq,
    )

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
    stored = _file_store.get(request.file_id)
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

    if JOB_QUEUE is None:
        raise HTTPException(status_code=503, detail="Service not ready.")

    stored = _file_store.get(request.file_id)
    if stored is None:
        raise HTTPException(
            status_code=404, detail=f"File ID '{request.file_id}' not found."
        )

    date_col = request.date_col or stored["date_col"]
    value_col = request.value_col or stored["value_col"]

    if len(_job_store) >= MAX_JOBS:
        # Simple eviction of oldest job
        oldest_key = next(iter(_job_store))
        _job_store.pop(oldest_key)

    job_id = str(uuid.uuid4())
    _job_store[job_id] = {
        "status": "pending",
        "progress": 0,
        "step": "Queued — waiting for an available slot…",
        "request": {
            "file_id": request.file_id,
            "date_col": date_col,
            "value_col": value_col,
            "forecast_horizon": request.forecast_horizon,
            "forced_model": request.forced_model,
            "user_prompt": request.user_prompt,
            "preflight_options": request.preflight_options,
        },
        "result": None,
        "error": None,
    }
    JOB_QUEUE.put_nowait(job_id)
    logger.info("Job enqueued: job_id=%s file_id=%s", job_id, request.file_id)
    return JobSubmitResponse(job_id=job_id, status="pending")


@app.get("/jobs/{job_id}", responses={404: {"description": "Job ID not found"}})
def get_job_status(
    job_id: str,
    _user: Annotated[dict, Depends(require_api_key)],
) -> JobStatusResponse:
    """Retrieve the status and results of a specific background job."""
    job = _job_store.get(job_id)
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
    """Allows users to chat with the agent about the uploaded data and results."""
    # If file_id is provided, use the specific file data
    if request.file_id:
        stored = _file_store.get(request.file_id)
        if not stored:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Chat session lost: the associated data is no longer in the server memory. "
                    "Please re-run the analysis."
                ),
            )

        try:
            # Delegate to orchestrator to query both the data and the indexed memory
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
        # General chat without specific file - use RAG knowledge base
        try:
            from orchestrator import chat_general

            # Delegate to orchestrator for general questions using RAG
            response = await asyncio.to_thread(
                chat_general,
                query=request.query,
                chroma_persist_dir=settings.CHROMA_PERSIST_DIR,
            )
            return ChatResponse(**response)
        except Exception as exc:
            logger.exception("General chat failed")
            # Provide a basic response
            answer = (
                "I encountered an error while processing your request. "
                "Please try again later."
            )
            return ChatResponse(answer=answer)


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
