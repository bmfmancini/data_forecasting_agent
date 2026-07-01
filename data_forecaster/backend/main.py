"""Main FastAPI application module for the Data Forecaster API."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional, cast, Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import core.config as settings
from core.logging_config import get_logger
from orchestrator import chat_with_data, index_analysis_results, run_pipeline
from schemas import (
    AnalysisResponse,
    AnalyzeRequest,
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
    JOB_QUEUE = asyncio.Queue()
    worker_task = asyncio.create_task(_job_worker())
    yield
    worker_task.cancel()
    await worker_task


app = FastAPI(title="Data Forecaster API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://frontend:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post(
    "/upload",
    responses={
        400: {"description": "Invalid file content, size, or format"},
        500: {"description": "File parsing failed"},
    },
)
async def upload_file(file: Annotated[UploadFile, File(...)]) -> UploadResponse:
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
            status_code=500, detail=f"Failed to parse file: {exc}"
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
async def preflight_check(request: AnalyzeRequest) -> PreflightResponse:
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
def analyze(request: AnalyzeRequest) -> JobSubmitResponse:
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
def get_job_status(job_id: str) -> JobStatusResponse:
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
async def chat_explorer(request: ChatRequest) -> ChatResponse:
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
            raise HTTPException(status_code=500, detail=str(exc)) from exc
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
            answer = f"I can help with general time series forecasting questions. Technical details: {str(exc)}"
            return ChatResponse(answer=answer)
