"""File storage service for the Data Forecaster backend.

Manages uploaded DataFrames using **disk-backed parquet storage**.
An in-memory metadata index (``_file_index``) holds column names,
frequency, and filename for fast lookups.  The DataFrame itself is
serialised to a parquet file on disk and loaded lazily on demand,
keeping process memory low even with many concurrent uploads.

Thread-safe via a module-level lock around compound operations.
"""

from __future__ import annotations

import os
import threading
import uuid
from typing import Any

import pandas as pd

import core.config as settings
from core.database import get_connection
from core.logging_config import get_logger

logger = get_logger(__name__)

# Bounds imported from config to prevent unbounded disk usage.
MAX_FILES: int = settings.MAX_INMEMORY_FILES

# In-memory metadata index: { file_id: { owner_id, date_col, value_col, freq, filename } }
# The DataFrame is NOT held here — it lives on disk.
_file_index: dict[str, dict[str, Any]] = {}

# Guards compound read-modify-write operations (e.g. eviction) to prevent
# race conditions between the async event loop and background threads.
_file_store_lock = threading.Lock()


def _storage_dir() -> str:
    """Return the configured storage directory at call time."""
    return settings.FILE_STORAGE_DIR


def init_storage() -> None:
    """Create the storage directory if it doesn't exist.

    Called at application startup to ensure the parquet directory is
    writable before the first upload arrives.
    """
    storage_dir = _storage_dir()
    os.makedirs(storage_dir, exist_ok=True)
    with _file_store_lock:
        conn = get_connection()
        try:
            # Jobs are in-memory only, so reservations from a terminated
            # process cannot still correspond to live work after restart.
            conn.execute("UPDATE uploaded_files SET active_jobs = 0")
            rows = conn.execute(
                "SELECT file_id, owner_id, date_col, value_col, freq, filename, active_jobs "
                "FROM uploaded_files ORDER BY created_at, rowid"
            ).fetchall()
            _file_index.clear()
            for (
                file_id,
                owner_id,
                date_col,
                value_col,
                freq,
                filename,
                active_jobs,
            ) in rows:
                if os.path.exists(_file_path(str(file_id))):
                    _file_index[str(file_id)] = {
                        "owner_id": owner_id,
                        "date_col": date_col,
                        "value_col": value_col,
                        "freq": freq,
                        "filename": filename,
                        "active_jobs": active_jobs,
                    }
                else:
                    conn.execute(
                        "DELETE FROM uploaded_files WHERE file_id = ?", (file_id,)
                    )
            conn.commit()
        finally:
            conn.close()
    logger.info("File storage directory ready: %s", storage_dir)


def _file_path(file_id: str) -> str:
    """Return the on-disk parquet path for a given ``file_id``.

    Args:
        file_id: The UUID returned by :func:`store_file`.

    Returns:
        Absolute path to the parquet file.
    """
    return os.path.join(_storage_dir(), f"{file_id}.parquet")


def store_file(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str,
    filename: str | None,
    owner_id: int | None = None,
) -> str:
    """Persist an uploaded DataFrame to disk and return its ``file_id``.

    The DataFrame is written as a parquet file.  Metadata (column names,
    frequency, filename) is kept in an in-memory index for fast lookups.
    Evicts the oldest entry (both index and file) when the store is full.

    Args:
        df:         Parsed pandas DataFrame.
        date_col:   Detected or user-selected date column name.
        value_col:  Detected or user-selected value column name.
        freq:       Inferred frequency string.
        filename:   Original filename from the upload.

    Returns:
        A UUID ``file_id`` string.
    """
    with _file_store_lock:
        if len(_file_index) >= MAX_FILES:
            oldest_file = _oldest_evictable_file()
            if oldest_file is None:
                raise RuntimeError(
                    "All stored files are currently used by analysis jobs."
                )
            _file_index.pop(oldest_file)
            _remove_disk_file(oldest_file)
            _delete_metadata(oldest_file)

        file_id = str(uuid.uuid4())
        path = _file_path(file_id)
        df.to_parquet(path, index=False)
        _file_index[file_id] = {
            "owner_id": owner_id,
            "date_col": date_col,
            "value_col": value_col,
            "freq": freq,
            "filename": filename,
            "active_jobs": 0,
        }
        _upsert_metadata(file_id, _file_index[file_id])
    logger.info(
        "File stored: file_id=%s rows=%d date_col=%s value_col=%s freq=%s",
        file_id,
        len(df),
        date_col,
        value_col,
        freq,
    )
    return file_id


def get_file(
    file_id: str,
    requester: dict[str, Any] | None = None,
    *,
    selected_date_col: str | None = None,
    selected_value_col: str | None = None,
) -> dict[str, Any] | None:
    """Retrieve a stored file's metadata and DataFrame by ``file_id``.

    Loads the DataFrame lazily from the parquet file on disk. Only the
    user-selected date and value columns are read when supplied; otherwise
    the initially detected columns are used. This preserves columnar pruning
    while allowing the frontend to override the upload-time suggestion.

    Args:
        file_id: The UUID returned by :func:`store_file`.
        selected_date_col: Optional date-column override selected after upload.
        selected_value_col: Optional value-column override selected after upload.

    Returns:
        A dict with ``df``, ``date_col``, ``value_col``, ``freq``, and
        ``filename`` keys, or ``None`` if the file_id is not in the index
        or the parquet file is missing/corrupt.
    """
    meta = _file_index.get(file_id)
    if meta is None:
        return None

    # Open-mode requests have no owner.  Once authentication is enabled, a
    # regular user may access only their own uploads; administrators may audit
    # all uploads.
    owner_id = meta.get("owner_id")
    if requester and not requester.get("is_admin") and owner_id != requester.get("id"):
        return None

    path = _file_path(file_id)
    if not os.path.exists(path):
        logger.warning("Parquet file missing for file_id=%s", file_id)
        return None

    date_col = meta["date_col"]
    value_col = meta["value_col"]
    columns = list(
        dict.fromkeys(
            [
                selected_date_col or date_col,
                selected_value_col or value_col,
            ]
        )
    )

    try:
        # Columnar pruning: read only the columns needed for this request.
        # This avoids loading wide DataFrames entirely into memory.
        df = pd.read_parquet(path, columns=columns)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.exception("Failed to load parquet for file_id=%s: %s", file_id, exc)
        return None

    return {
        "df": df,
        "owner_id": owner_id,
        "date_col": date_col,
        "value_col": value_col,
        "freq": meta["freq"],
        "filename": meta["filename"],
    }


def load_file_from_disk(
    file_id: str,
    *,
    selected_date_col: str,
    selected_value_col: str,
) -> dict[str, Any] | None:
    """Load a reserved file using durable metadata in a spawned worker.

    Spawned forecast processes intentionally do not inherit the parent's
    in-memory metadata index. This read-only path avoids resetting reservation
    counters while retaining parquet column pruning.
    """
    connection = get_connection()
    try:
        row = connection.execute(
            "SELECT owner_id, date_col, value_col, freq, filename "
            "FROM uploaded_files WHERE file_id = ?",
            (file_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    path = _file_path(file_id)
    try:
        frame = pd.read_parquet(
            path,
            columns=list(dict.fromkeys([selected_date_col, selected_value_col])),
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        logger.exception("Spawned worker could not load file_id=%s", file_id)
        return None
    return {
        "df": frame,
        "owner_id": row["owner_id"],
        "date_col": row["date_col"],
        "value_col": row["value_col"],
        "freq": row["freq"],
        "filename": row["filename"],
    }


def _remove_disk_file(file_id: str) -> None:
    """Delete the parquet file for a given ``file_id`` from disk.

    Logs a warning if the file is already gone (e.g. manually deleted).

    Args:
        file_id: The UUID of the file to remove.
    """
    path = _file_path(file_id)
    try:
        os.remove(path)
    except FileNotFoundError:
        logger.warning("Parquet file already removed for file_id=%s", file_id)
    except OSError as exc:
        logger.warning("Failed to remove parquet for file_id=%s: %s", file_id, exc)


def _oldest_evictable_file() -> str | None:
    """Return the oldest cached file that is not reserved by a job."""
    for file_id, metadata in _file_index.items():
        if not metadata["active_jobs"]:
            return file_id
    return None


def reserve_file(file_id: str) -> bool:
    """Prevent an uploaded file from being evicted while a job uses it.

    Args:
        file_id: Identifier of the uploaded file to reserve.

    Returns:
        ``True`` when the file was reserved, or ``False`` when it is absent.
    """
    with _file_store_lock:
        metadata = _file_index.get(file_id)
        if metadata is None:
            return False
        metadata["active_jobs"] += 1
        _update_active_jobs(file_id, 1)
        return True


def release_file(file_id: str) -> None:
    """Release a job's eviction reservation for an uploaded file.

    Args:
        file_id: Identifier of the uploaded file to release.
    """
    with _file_store_lock:
        metadata = _file_index.get(file_id)
        if metadata is None or not metadata["active_jobs"]:
            return
        metadata["active_jobs"] -= 1
        _update_active_jobs(file_id, -1)


def _upsert_metadata(file_id: str, meta: dict[str, Any]) -> None:
    """Write file metadata durably after its parquet payload is created."""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO uploaded_files "
            "(file_id, owner_id, date_col, value_col, freq, filename, active_jobs) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                file_id,
                meta["owner_id"],
                meta["date_col"],
                meta["value_col"],
                meta["freq"],
                meta["filename"],
                meta["active_jobs"],
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _delete_metadata(file_id: str) -> None:
    """Remove durable metadata for an evicted upload."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM uploaded_files WHERE file_id = ?", (file_id,))
        conn.commit()
    finally:
        conn.close()


def _update_active_jobs(file_id: str, change: int) -> None:
    """Apply a bounded active-job counter update to durable metadata."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE uploaded_files SET active_jobs = MAX(0, active_jobs + ?) "
            "WHERE file_id = ?",
            (change, file_id),
        )
        conn.commit()
    finally:
        conn.close()
