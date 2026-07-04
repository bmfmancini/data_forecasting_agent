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
from typing import Any, Dict

import pandas as pd

import core.config as settings
from core.logging_config import get_logger

logger = get_logger(__name__)

# Bounds imported from config to prevent unbounded disk usage.
MAX_FILES: int = settings.MAX_INMEMORY_FILES

# Directory where parquet files are persisted.
_STORAGE_DIR: str = settings.FILE_STORAGE_DIR

# In-memory metadata index: { file_id: { date_col, value_col, freq, filename } }
# The DataFrame is NOT held here — it lives on disk.
_file_index: Dict[str, Dict[str, Any]] = {}

# Guards compound read-modify-write operations (e.g. eviction) to prevent
# race conditions between the async event loop and background threads.
_file_store_lock = threading.Lock()


def init_storage() -> None:
    """Create the storage directory if it doesn't exist.

    Called at application startup to ensure the parquet directory is
    writable before the first upload arrives.
    """
    os.makedirs(_STORAGE_DIR, exist_ok=True)
    logger.info("File storage directory ready: %s", _STORAGE_DIR)


def _file_path(file_id: str) -> str:
    """Return the on-disk parquet path for a given ``file_id``.

    Args:
        file_id: The UUID returned by :func:`store_file`.

    Returns:
        Absolute path to the parquet file.
    """
    return os.path.join(_STORAGE_DIR, f"{file_id}.parquet")


def store_file(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str,
    filename: str | None,
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
            oldest_file = next(iter(_file_index))
            _file_index.pop(oldest_file)
            _remove_disk_file(oldest_file)

        file_id = str(uuid.uuid4())
        path = _file_path(file_id)
        df.to_parquet(path, index=False)
        _file_index[file_id] = {
            "date_col": date_col,
            "value_col": value_col,
            "freq": freq,
            "filename": filename,
        }
    logger.info(
        "File stored: file_id=%s rows=%d date_col=%s value_col=%s freq=%s",
        file_id,
        len(df),
        date_col,
        value_col,
        freq,
    )
    return file_id


def get_file(file_id: str) -> dict[str, Any] | None:
    """Retrieve a stored file's metadata and DataFrame by ``file_id``.

    Loads the DataFrame lazily from the parquet file on disk.

    Args:
        file_id: The UUID returned by :func:`store_file`.

    Returns:
        A dict with ``df``, ``date_col``, ``value_col``, ``freq``, and
        ``filename`` keys, or ``None`` if the file_id is not in the index
        or the parquet file is missing/corrupt.
    """
    meta = _file_index.get(file_id)
    if meta is None:
        return None

    path = _file_path(file_id)
    if not os.path.exists(path):
        logger.warning("Parquet file missing for file_id=%s", file_id)
        return None

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.exception("Failed to load parquet for file_id=%s: %s", file_id, exc)
        return None

    return {
        "df": df,
        "date_col": meta["date_col"],
        "value_col": meta["value_col"],
        "freq": meta["freq"],
        "filename": meta["filename"],
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