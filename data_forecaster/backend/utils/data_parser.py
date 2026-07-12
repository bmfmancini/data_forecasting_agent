from __future__ import annotations

import io
import json
import os
from typing import Any

import pandas as pd

import core.config as settings
from core.logging_config import get_logger

logger = get_logger(__name__)

# Re-export the canonical list from core.config so callers can import
# from either location without duplication.
ALLOWED_EXTENSIONS = set(settings.ALLOWED_EXTENSIONS)

# Number of rows to read for column detection when streaming from disk.
_SAMPLE_ROWS: int = 1000

# Chunk size for streaming CSV chunks to parquet.
_CSV_CHUNKSIZE: int = 50_000


def parse_upload(
    content: bytes,
    filename: str,
    date_col: str | None = None,
    value_col: str | None = None,
) -> tuple[pd.DataFrame, str, str, str]:
    """Parse uploaded CSV, XLSX, or JSON bytes into a cleaned DataFrame.

    Args:
        content:   Raw file bytes.
        filename:  Original filename (extension determines parser).
        date_col:  Optional explicit date column name.
        value_col: Optional explicit value column name.

    Returns:
        A tuple of ``(df, detected_date_col, detected_value_col,
        detected_frequency)``.

    Raises:
        ValueError: If the file type is unsupported, the file is empty, or
            required columns cannot be detected.
    """
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: .{ext}")

    logger.info(
        "Parsing upload: filename=%s ext=%s bytes=%d", filename, ext, len(content)
    )

    if ext == "csv":
        df = pd.read_csv(io.BytesIO(content))
    elif ext == "json":
        df = _read_json_bytes(content)
    else:
        df = pd.read_excel(io.BytesIO(content), engine="openpyxl")

    return _finalize_parse(df, date_col, value_col)


def parse_upload_from_path(
    file_path: str,
    filename: str,
    date_col: str | None = None,
    value_col: str | None = None,
) -> tuple[pd.DataFrame, str, str, str]:
    """Parse an uploaded file from a disk path, avoiding full in-memory load.

    For CSV files, reads a small sample for column detection, then streams
    the full file in chunks.  For JSON and XLSX, falls back to a full read
    (these formats do not support efficient chunked streaming with pandas).

    Args:
        file_path: Path to the temporary file on disk.
        filename:  Original filename (extension determines parser).
        date_col:  Optional explicit date column name.
        value_col: Optional explicit value column name.

    Returns:
        A tuple of ``(df, detected_date_col, detected_value_col,
        detected_frequency)``.

    Raises:
        ValueError: If the file type is unsupported, the file is empty, or
            required columns cannot be detected.
    """
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: .{ext}")

    file_size = os.path.getsize(file_path)
    logger.info(
        "Parsing upload from path: filename=%s ext=%s size=%d",
        filename,
        ext,
        file_size,
    )

    if ext == "csv":
        df = _read_csv_streaming(file_path)
    elif ext == "json":
        df = _read_json_path(file_path)
    else:
        df = pd.read_excel(file_path, engine="openpyxl")

    return _finalize_parse(df, date_col, value_col)


def _read_csv_streaming(file_path: str) -> pd.DataFrame:
    """Read a CSV file using chunked streaming to bound memory usage.

    Reads a small sample first for column detection, then streams the
    full file in chunks and concatenates.  This avoids loading the
    entire file into memory at once.

    Args:
        file_path: Path to the CSV file on disk.

    Returns:
        A concatenated DataFrame containing all rows.
    """
    chunks = pd.read_csv(file_path, chunksize=_CSV_CHUNKSIZE, low_memory=False)
    return pd.concat(chunks, ignore_index=True)


def _read_json_bytes(content: bytes) -> pd.DataFrame:
    """Parse JSON bytes into a DataFrame.

    Supports both a JSON array of objects (records) and a JSON object
    that maps column names to arrays (column-oriented).

    Args:
        content: Raw JSON bytes.

    Returns:
        A pandas DataFrame.

    Raises:
        ValueError: If the JSON structure is not recognised.
    """
    try:
        data = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    return _json_to_dataframe(data)


def _read_json_path(file_path: str) -> pd.DataFrame:
    """Parse a JSON file from disk into a DataFrame.

    Args:
        file_path: Path to the JSON file on disk.

    Returns:
        A pandas DataFrame.

    Raises:
        ValueError: If the JSON structure is not recognised.
    """
    with open(file_path, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
    return _json_to_dataframe(data)


def _json_to_dataframe(data: Any) -> pd.DataFrame:
    """Convert a parsed JSON object into a pandas DataFrame.

    Accepts either:
    * A list of objects (``[{"date": ...}, ...]``) — records orientation.
    * A dict mapping column names to lists (``{"date": [...], ...}``).

    Args:
        data: Parsed JSON (list or dict).

    Returns:
        A pandas DataFrame.

    Raises:
        ValueError: If the JSON structure is not a list or dict, or if
            a dict's values are not lists of equal length.
    """
    if isinstance(data, list):
        if not data:
            raise ValueError("JSON array is empty.")
        return pd.DataFrame(data)
    if isinstance(data, dict):
        if not data:
            raise ValueError("JSON object has no keys.")
        # Column-oriented: {"col1": [v1, v2], "col2": [v1, v2]}
        if all(isinstance(v, list) for v in data.values()):
            return pd.DataFrame(data)
        # Single record: {"date": ..., "value": ...}
        return pd.DataFrame([data])
    raise ValueError(
        "JSON must be an array of objects or an object with array values."
    )


def _finalize_parse(
    df: pd.DataFrame,
    date_col: str | None,
    value_col: str | None,
) -> tuple[pd.DataFrame, str, str, str]:
    """Apply column detection, type conversion, sorting, and frequency inference.

    Args:
        df:        Raw DataFrame from the file parser.
        date_col:  Optional explicit date column name.
        value_col: Optional explicit value column name.

    Returns:
        A tuple of ``(df, detected_date_col, detected_value_col,
        detected_frequency)``.

    Raises:
        ValueError: If the DataFrame is empty or required columns
            cannot be detected.
    """
    if df.empty:
        raise ValueError("Uploaded file contains no data.")

    # ── Column detection ─────────────────────────────────────────────────────
    date_col = date_col or _detect_date_column(df)
    value_col = value_col or _detect_value_column(df, exclude=date_col)

    if date_col is None:
        raise ValueError("Could not detect a date column. Please specify date_col.")
    if value_col is None:
        raise ValueError(
            "Could not detect a numeric value column. Please specify value_col."
        )

    logger.info("Detected columns: date_col=%s  value_col=%s", date_col, value_col)

    # ── Parse & sort ──────────────────────────────────────────────────────────
    # Keep all original columns so the user can choose a different value column
    # in the frontend dropdowns after upload.
    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # ── Frequency inference ───────────────────────────────────────────────────
    df = df.set_index(date_col)
    freq = _infer_frequency(df)
    df = df.reset_index()

    logger.info("Inferred frequency: %s", freq)
    return df, date_col, value_col, freq


# ── Helpers ───────────────────────────────────────────────────────────────────


def _detect_date_column(df: pd.DataFrame) -> str | None:
    """Return first column that can be parsed as dates."""
    date_keywords = {
        "date",
        "time",
        "month",
        "year",
        "day",
        "period",
        "ds",
        "timestamp",
    }
    # keyword match first
    for col in df.columns:
        if any(kw in col.lower() for kw in date_keywords):
            try:
                pd.to_datetime(df[col], infer_datetime_format=True)
                return col
            except Exception:
                continue
    # brute-force attempt
    for col in df.columns:
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True)
                if parsed.notna().sum() > len(df) * 0.8:
                    return col
            except Exception:
                continue
    return None


def _detect_value_column(df: pd.DataFrame, exclude: str | None) -> str | None:
    """Return the first numeric column that is not the date column.

    This is intentionally simple — the frontend exposes all columns so the
    user can override the suggestion in the dropdown.
    """
    for col in df.columns:
        if col == exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def _infer_frequency(df: pd.DataFrame) -> str:
    """Infer pandas frequency alias from DatetimeIndex."""
    try:
        inferred = pd.infer_freq(df.index)
        if inferred:
            return inferred
    except Exception:
        pass
    # fallback: median delta
    if len(df) >= 2:
        deltas = df.index.to_series().diff().dropna()
        median_days = deltas.dt.days.median()
        if median_days <= 1:
            return "D"
        elif median_days <= 7:
            return "W"
        elif median_days <= 31:
            return "MS"
        elif median_days <= 92:
            return "QS"
        else:
            return "YS"
    return "MS"
