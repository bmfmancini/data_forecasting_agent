from __future__ import annotations

import io
from typing import Optional

import pandas as pd

from core.logging_config import get_logger

logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {"csv", "xlsx"}


def parse_upload(
    content: bytes,
    filename: str,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
) -> tuple[pd.DataFrame, str, str, str]:
    """Parse uploaded CSV or XLSX bytes into a cleaned DataFrame.

    Returns:
        (df, detected_date_col, detected_value_col, detected_frequency)
    """
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: .{ext}")

    logger.info("Parsing upload: filename=%s ext=%s bytes=%d", filename, ext, len(content))

    if ext == "csv":
        df = pd.read_csv(io.BytesIO(content))
    else:
        df = pd.read_excel(io.BytesIO(content), engine="openpyxl")

    if df.empty:
        raise ValueError("Uploaded file contains no data.")

    # ── Column detection ─────────────────────────────────────────────────────
    date_col = date_col or _detect_date_column(df)
    value_col = value_col or _detect_value_column(df, exclude=date_col)

    if date_col is None:
        raise ValueError("Could not detect a date column. Please specify date_col.")
    if value_col is None:
        raise ValueError("Could not detect a numeric value column. Please specify value_col.")

    logger.info("Detected columns: date_col=%s  value_col=%s", date_col, value_col)

    # ── Parse & sort ──────────────────────────────────────────────────────────
    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
    df = df[[date_col, value_col]].dropna(subset=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # ── Frequency inference ───────────────────────────────────────────────────
    df = df.set_index(date_col)
    freq = _infer_frequency(df)
    df = df.reset_index()

    logger.info("Inferred frequency: %s", freq)
    return df, date_col, value_col, freq


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Return first column that can be parsed as dates."""
    date_keywords = {"date", "time", "month", "year", "day", "period", "ds", "timestamp"}
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


def _detect_value_column(df: pd.DataFrame, exclude: Optional[str]) -> Optional[str]:
    """Return first numeric column that is not the date column."""
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
