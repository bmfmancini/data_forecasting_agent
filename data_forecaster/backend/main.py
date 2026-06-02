from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import core.config as settings
from core.logging_config import get_logger
from orchestrator import run_pipeline
from schemas import AnalysisResponse, AnalyzeRequest, JobStatusResponse, JobSubmitResponse, UploadResponse
from utils.data_parser import parse_upload

logger = get_logger(__name__)

# ── In-memory stores ──────────────────────────────────────────────────────────
# { file_id: { df, date_col, value_col, freq, filename } }
_file_store: dict[str, dict] = {}
# { job_id: { status, progress, step, request, result, error } }
_job_store: dict[str, dict] = {}

# ── Job queue & worker ────────────────────────────────────────────────────────
_job_queue: asyncio.Queue | None = None


def _update_job_progress(job_id: str, pct: int, step: str) -> None:
    """Called from the pipeline thread; CPython GIL makes dict updates thread-safe."""
    job = _job_store.get(job_id)
    if job:
        job["progress"] = pct
        job["step"] = step


async def _job_worker() -> None:
    """Processes one job at a time from the FIFO queue."""
    while True:
        job_id: str = await _job_queue.get()  # type: ignore[union-attr]
        job = _job_store.get(job_id)
        if job is None:
            _job_queue.task_done()  # type: ignore[union-attr]
            continue

        job["status"] = "running"
        req = job["request"]
        stored = _file_store.get(req["file_id"])

        if stored is None:
            job["status"] = "error"
            job["step"] = "Error: uploaded file not found."
            job["error"] = f"File ID '{req['file_id']}' not found in store."
            _job_queue.task_done()  # type: ignore[union-attr]
            continue

        def _run_pipeline_sync() -> AnalysisResponse:
            return run_pipeline(
                df=stored["df"],
                file_id=req["file_id"],
                date_col=req["date_col"],
                value_col=req["value_col"],
                freq=stored["freq"],
                forecast_horizon=req["forecast_horizon"],
                forced_model=req["forced_model"],
                user_prompt=req.get("user_prompt"),
                chroma_persist_dir=settings.CHROMA_PERSIST_DIR,
                progress_callback=lambda pct, step: _update_job_progress(job_id, pct, step),
            )

        try:
            result: AnalysisResponse = await asyncio.to_thread(_run_pipeline_sync)
            job["status"] = "done"
            job["progress"] = 100
            job["step"] = "Analysis complete."
            job["result"] = result.model_dump()
        except Exception as exc:
            logger.exception("Pipeline failed for job_id=%s file_id=%s", job_id, req["file_id"])
            job["status"] = "error"
            job["step"] = "Pipeline failed."
            job["error"] = str(exc)
        finally:
            _job_queue.task_done()  # type: ignore[union-attr]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _job_queue
    _job_queue = asyncio.Queue()
    worker_task = asyncio.create_task(_job_worker())
    yield
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


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
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    logger.info("POST /upload  filename=%s  content_type=%s", file.filename, file.content_type)

    # ── Validate content-type ─────────────────────────────────────────────────
    if file.content_type not in settings.ALLOWED_MIME_TYPES + ["application/octet-stream"]:
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
        df, date_col, value_col, freq = parse_upload(contents, file.filename or "upload.csv")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during file parsing")
        raise HTTPException(status_code=500, detail=f"Failed to parse file: {exc}")

    # ── Store & return ────────────────────────────────────────────────────────
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
        file_id, len(df), date_col, value_col, freq,
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


@app.post("/analyze", response_model=JobSubmitResponse, status_code=202)
def analyze(request: AnalyzeRequest) -> JobSubmitResponse:
    logger.info(
        "POST /analyze  file_id=%s horizon=%d", request.file_id, request.forecast_horizon
    )

    if _job_queue is None:
        raise HTTPException(status_code=503, detail="Service not ready.")

    stored = _file_store.get(request.file_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"File ID '{request.file_id}' not found.")

    date_col = request.date_col or stored["date_col"]
    value_col = request.value_col or stored["value_col"]

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
        },
        "result": None,
        "error": None,
    }
    _job_queue.put_nowait(job_id)
    logger.info("Job enqueued: job_id=%s file_id=%s", job_id, request.file_id)
    return JobSubmitResponse(job_id=job_id, status="pending")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
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
