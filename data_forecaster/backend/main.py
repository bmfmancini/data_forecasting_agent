from __future__ import annotations

import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import core.config as settings
from core.logging_config import get_logger
from orchestrator import run_pipeline
from schemas import AnalysisResponse, AnalyzeRequest, UploadResponse
from utils.data_parser import parse_upload

logger = get_logger(__name__)

app = FastAPI(title="Data Forecaster API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://frontend:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory file store ──────────────────────────────────────────────────────
# { file_id: { df, date_col, value_col, freq, filename } }
_file_store: dict[str, dict] = {}


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


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalyzeRequest) -> AnalysisResponse:
    logger.info(
        "POST /analyze  file_id=%s horizon=%d", request.file_id, request.forecast_horizon
    )

    stored = _file_store.get(request.file_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"File ID '{request.file_id}' not found.")

    date_col = request.date_col or stored["date_col"]
    value_col = request.value_col or stored["value_col"]

    try:
        result = run_pipeline(
            df=stored["df"],
            file_id=request.file_id,
            date_col=date_col,
            value_col=value_col,
            freq=stored["freq"],
            forecast_horizon=request.forecast_horizon,
            chroma_persist_dir=settings.CHROMA_PERSIST_DIR,
        )
    except Exception as exc:
        logger.exception("Pipeline failed for file_id=%s", request.file_id)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    return result
