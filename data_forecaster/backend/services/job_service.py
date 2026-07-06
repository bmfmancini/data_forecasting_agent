"""Job queue service for the Data Forecaster backend.

Manages the in-memory ``_job_store`` dict and the async FIFO job queue.
Encapsulates job creation, progress tracking, eviction, and the
background worker so route handlers remain thin.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from typing import Any, Callable, Dict, cast

import core.config as settings
from core.logging_config import get_logger
from schemas import AnalysisResponse
from services.file_service import get_file
from services.pipeline_service import run_pipeline
from services.rag_service import get_rag_kb

logger = get_logger(__name__)

MAX_JOBS: int = settings.MAX_INMEMORY_JOBS

# { job_id: { status, progress, step, request, result, error } }
_job_store: Dict[str, Dict[str, Any]] = {}

# Guards compound read-modify-write operations on _job_store.
_job_store_lock = threading.Lock()

# The async FIFO queue (initialised in ``init_job_queue``).
JOB_QUEUE: asyncio.Queue[str] = cast(asyncio.Queue, None)


def init_job_queue() -> None:
    """Initialise the job queue (called at app startup)."""
    global JOB_QUEUE  # pylint: disable=global-statement
    JOB_QUEUE = asyncio.Queue()


def is_queue_ready() -> bool:
    """Return ``True`` when the job queue has been initialised."""
    return JOB_QUEUE is not None


def create_job(
    file_id: str,
    date_col: str,
    value_col: str,
    forecast_horizon: int,
    forced_model: str | None,
    user_prompt: str | None,
    preflight_options: dict[str, Any] | None,
) -> str:
    """Create a pending job, enqueue it, and return the ``job_id``.

    Args:
        file_id:          Identifier of the uploaded file.
        date_col:         Date column name.
        value_col:        Value column name.
        forecast_horizon: Number of periods to forecast.
        forced_model:     Optional model override.
        user_prompt:      Optional extra report instructions.
        preflight_options: Optional preflight configuration dict.

    Returns:
        A UUID ``job_id`` string.
    """
    # Evict oldest job if at capacity (double-check inside lock)
    if len(_job_store) >= MAX_JOBS:
        with _job_store_lock:
            if len(_job_store) >= MAX_JOBS:
                oldest_key = next(iter(_job_store))
                _job_store.pop(oldest_key)

    job_id = str(uuid.uuid4())
    with _job_store_lock:
        _job_store[job_id] = {
            "status": "pending",
            "progress": 0,
            "step": "Queued — waiting for an available slot…",
            "request": {
                "file_id": file_id,
                "date_col": date_col,
                "value_col": value_col,
                "forecast_horizon": forecast_horizon,
                "forced_model": forced_model,
                "user_prompt": user_prompt,
                "preflight_options": preflight_options,
            },
            "result": None,
            "error": None,
        }
    JOB_QUEUE.put_nowait(job_id)
    logger.info("Job enqueued: job_id=%s file_id=%s", job_id, file_id)
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    """Retrieve a job dict by ``job_id``.

    Args:
        job_id: The UUID returned by :func:`create_job`.

    Returns:
        The job dict or ``None`` if not found.
    """
    return _job_store.get(job_id)


def get_job_status_only(job_id: str) -> dict[str, Any] | None:
    """Return only status/progress/step for lightweight polling.

    Args:
        job_id: The UUID returned by :func:`create_job`.

    Returns:
        A dict with ``status``, ``progress``, and ``step`` keys, or
        ``None`` if the job is not found.
    """
    job = _job_store.get(job_id)
    if job is None:
        return None
    return {
        "status": job["status"],
        "progress": job["progress"],
        "step": job["step"],
    }


def _update_job_progress(job_id: str, pct: int, step: str) -> None:
    """Called from the pipeline thread; CPython GIL makes dict updates thread-safe."""
    job = _job_store.get(job_id)
    if job:
        job["progress"] = pct
        job["step"] = step


async def job_worker() -> None:
    """Process jobs from the FIFO queue one at a time."""
    while True:
        job_id: str = await JOB_QUEUE.get()
        job = _job_store.get(job_id)
        if job is None:
            JOB_QUEUE.task_done()
            continue

        job["status"] = "running"
        req = job["request"]
        stored = get_file(req["file_id"])

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

            # Store the output results in the agent's memory
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


def index_analysis_results(
    file_id: str, analysis_result: AnalysisResponse, chroma_persist_dir: str
) -> None:
    """Index analysis results into the RAG knowledge base.

    Args:
        file_id:            The file identifier.
        analysis_result:    The completed :class:`AnalysisResponse`.
        chroma_persist_dir: Path to the ChromaDB persistence directory.
    """
    logger.info("Indexing analysis results for file_id=%s", file_id)
    rag_kb = get_rag_kb(chroma_persist_dir)

    summary = (
        f"Forecasting Analysis Results for {file_id}:\n"
        f"Summary Report: {analysis_result.report}\n"
        f"Model Selection: {analysis_result.model_selection.selected_model}\n"
        f"Statistical Insights: {analysis_result.statistical.model_dump_json()}\n"
    )

    if hasattr(rag_kb, "add_texts"):
        rag_kb.add_texts(
            texts=[summary],
            metadatas=[{"file_id": file_id, "type": "analysis_result"}],
        )
    else:
        logger.warning(
            "RAGKnowledgeBase does not support direct text indexing. "
            "Result memory not persisted."
        )
