"""File storage service for the Data Forecaster backend.

Manages the in-memory ``_file_store`` dict that holds uploaded
DataFrames.  Encapsulates eviction logic and thread-safe access so
route handlers remain thin.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Dict

import core.config as settings
from core.logging_config import get_logger

logger = get_logger(__name__)

# Bounds imported from config to prevent OOM restarts.
MAX_FILES: int = settings.MAX_INMEMORY_FILES

# { file_id: { df, date_col, value_col, freq, filename } }
_file_store: Dict[str, Dict[str, Any]] = {}

# Guards compound read-modify-write operations (e.g. eviction) to prevent
# race conditions between the async event loop and background threads.
_file_store_lock = threading.Lock()


def store_file(
    df: Any,
    date_col: str,
    value_col: str,
    freq: str,
    filename: str | None,
) -> str:
    """Store an uploaded DataFrame and return its ``file_id``.

    Evicts the oldest entry when the store is full.

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
        if len(_file_store) >= MAX_FILES:
            oldest_file = next(iter(_file_store))
            _file_store.pop(oldest_file)

        file_id = str(uuid.uuid4())
        _file_store[file_id] = {
            "df": df,
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
    """Retrieve a stored file dict by ``file_id``.

    Args:
        file_id: The UUID returned by :func:`store_file`.

    Returns:
        The stored dict (``df``, ``date_col``, ``value_col``, ``freq``,
        ``filename``) or ``None`` if not found.
    """
    return _file_store.get(file_id)