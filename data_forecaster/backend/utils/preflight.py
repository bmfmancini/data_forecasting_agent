from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from core.logging_config import get_logger
from schemas import PreflightDecision, PreflightResponse


logger = get_logger(__name__)

AI_DECIDE = "let AI decide"
AGGREGATION_OPTIONS = [AI_DECIDE, "sum", "mean", "latest"]
MISSING_OPTIONS = [AI_DECIDE, "interpolate", "forward-fill", "drop"]
FREQUENCY_OPTIONS = [AI_DECIDE, "D", "W", "MS", "QS", "YS"]
SHORT_SERIES_OPTIONS = [AI_DECIDE, "continue", "stop"]


def run_preflight_checks(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    forecast_horizon: int,
) -> PreflightResponse:
    """Inspect the selected series and return any user decisions needed."""
    selected = _selected_frame(df, date_col, value_col)
    series = selected.set_index(date_col)[value_col]
    diffs = series.index.to_series().diff().dropna()
    mode_diff = diffs.mode()[0] if len(diffs) > 0 else None

    duplicate_ts = int(selected[date_col].duplicated().sum())
    missing_values = int(series.isna().sum())
    missing_ts = int((diffs > mode_diff * 1.5).sum()) if mode_diff is not None else 0
    is_regular = bool(diffs.nunique() == 1) if len(diffs) > 0 else True
    detected_frequency = _infer_frequency(selected.set_index(date_col))
    usable_observations = int(series.dropna().shape[0])

    issues: list[str] = []
    warnings: list[str] = []
    decisions: list[PreflightDecision] = []
    defaults: dict[str, Any] = {
        "duplicate_strategy": AI_DECIDE,
        "missing_strategy": AI_DECIDE,
        "frequency": AI_DECIDE,
        "continue_short_series": AI_DECIDE,
    }

    if duplicate_ts:
        issues.append(f"{duplicate_ts} duplicate timestamp(s) found.")
        decisions.append(PreflightDecision(
            key="duplicate_strategy",
            label="Duplicate timestamp handling",
            message="Choose how repeated timestamps should be combined before forecasting, or let AI decide from the data.",
            options=AGGREGATION_OPTIONS,
            default=AI_DECIDE,
        ))

    if missing_values:
        issues.append(f"{missing_values} missing value(s) found in '{value_col}'.")
        decisions.append(PreflightDecision(
            key="missing_strategy",
            label="Missing value handling",
            message="Choose how missing values should be handled before modeling, or let AI decide from the data.",
            options=MISSING_OPTIONS,
            default=AI_DECIDE,
        ))

    if missing_ts or not is_regular:
        issues.append("Irregular intervals or timestamp gaps found.")
        decisions.append(PreflightDecision(
            key="frequency",
            label="Forecast frequency",
            message="Choose the regular frequency to use for the forecast output, or let AI decide from the timestamp pattern.",
            options=FREQUENCY_OPTIONS,
            default=AI_DECIDE,
        ))

    if usable_observations < 20:
        warnings.append(
            f"Only {usable_observations} usable observation(s) are available; forecast reliability may be low."
        )
        decisions.append(PreflightDecision(
            key="continue_short_series",
            label="Short series confirmation",
            message="Confirm whether to continue with a short time series, or let AI decide.",
            options=SHORT_SERIES_OPTIONS,
            default=AI_DECIDE,
        ))

    if forecast_horizon > max(usable_observations, 1):
        warnings.append(
            "The forecast horizon is longer than the usable historical series; uncertainty may be high."
        )

    if series.dropna().lt(0).any():
        warnings.append("The selected value column contains negative values.")

    if decisions:
        status = "needs_input" if any(d.required for d in decisions) else "warning"
    elif warnings:
        status = "warning"
    else:
        status = "ready"

    return PreflightResponse(
        status=status,
        detected_frequency=detected_frequency,
        row_count=len(selected),
        usable_observations=usable_observations,
        duplicate_timestamps=duplicate_ts,
        missing_values=missing_values,
        missing_timestamps=missing_ts,
        is_regular=is_regular,
        issues=issues,
        warnings=warnings,
        decisions=decisions,
        defaults=defaults,
    )


def prepare_series_frame(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    options: dict | None = None,
) -> tuple[pd.DataFrame, str]:
    """Apply preflight choices and return a clean two-column time series frame."""
    options = options or {}
    selected = _selected_frame(df, date_col, value_col)
    inferred_frequency = _infer_frequency(selected.set_index(date_col))
    resolved_options = _resolve_ai_options(
        selected, date_col, value_col, options, inferred_frequency
    )
    if resolved_options.get("continue_short_series") == "stop":
        raise ValueError("Analysis stopped because the selected series is too short.")

    frequency = _resolve_frequency(resolved_options.get("frequency"), inferred_frequency)
    duplicate_strategy = _resolve_duplicate_strategy(
        resolved_options.get("duplicate_strategy"), value_col
    )
    missing_strategy = _resolve_missing_strategy(resolved_options.get("missing_strategy"))

    series = selected.set_index(date_col)[value_col].sort_index()

    if series.index.has_duplicates:
        series = _aggregate_duplicates(series, duplicate_strategy)

    if frequency:
        series = series.asfreq(frequency)

    series = _handle_missing(series, missing_strategy)
    series = series.dropna()

    prepared = series.rename(value_col).reset_index()
    prepared.columns = [date_col, value_col]
    return prepared, frequency


def _selected_frame(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    if date_col == value_col:
        raise ValueError("Date column and value column must be different.")
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' was not found.")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' was not found.")

    selected = df[[date_col, value_col]].copy()
    selected[date_col] = pd.to_datetime(selected[date_col], errors="coerce")
    selected[value_col] = pd.to_numeric(selected[value_col], errors="coerce")
    selected = selected.dropna(subset=[date_col])
    selected = selected.sort_values(date_col).reset_index(drop=True)
    if selected.empty:
        raise ValueError("No usable timestamps were found for the selected date column.")
    return selected


def _aggregate_duplicates(series: pd.Series, strategy: str) -> pd.Series:
    if strategy == "sum":
        return series.groupby(level=0).sum(min_count=1)
    if strategy == "latest":
        return series.groupby(level=0).last()
    return series.groupby(level=0).mean()


def _handle_missing(series: pd.Series, strategy: str) -> pd.Series:
    if strategy == "drop":
        return series.dropna()
    if strategy == "forward-fill":
        return series.ffill().bfill()
    return series.interpolate(method="time").ffill().bfill()


def _resolve_frequency(choice: str | None, inferred_frequency: str) -> str:
    if not choice or choice == AI_DECIDE:
        return inferred_frequency
    return choice


def _resolve_missing_strategy(choice: str | None) -> str:
    if not choice or choice == AI_DECIDE:
        return "interpolate"
    return choice


def _resolve_duplicate_strategy(choice: str | None, value_col: str) -> str:
    if choice and choice != AI_DECIDE:
        return choice

    name = value_col.lower()
    additive_terms = (
        "sales", "revenue", "amount", "total", "count", "units", "quantity",
        "qty", "volume", "demand", "orders", "passengers",
    )
    point_in_time_terms = (
        "inventory", "stock", "balance", "level", "price", "rate", "ratio",
        "temperature", "temp", "score", "index",
    )

    if any(term in name for term in additive_terms):
        return "sum"
    if any(term in name for term in point_in_time_terms):
        return "latest"
    return "mean"


def _resolve_ai_options(
    selected: pd.DataFrame,
    date_col: str,
    value_col: str,
    options: dict,
    inferred_frequency: str,
) -> dict:
    if not any(value == AI_DECIDE for value in options.values()):
        return options

    heuristic_options = {
        "duplicate_strategy": _resolve_duplicate_strategy(
            options.get("duplicate_strategy"), value_col
        ),
        "missing_strategy": _resolve_missing_strategy(options.get("missing_strategy")),
        "frequency": _resolve_frequency(options.get("frequency"), inferred_frequency),
        "continue_short_series": "continue",
    }

    try:
        from langchain_groq import ChatGroq
    except ImportError:
        return {**options, **heuristic_options}

    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return {**options, **heuristic_options}

        series = selected.set_index(date_col)[value_col]
        payload = {
            "value_column": value_col,
            "row_count": len(selected),
            "usable_observations": int(series.dropna().shape[0]),
            "duplicate_timestamps": int(selected[date_col].duplicated().sum()),
            "missing_values": int(series.isna().sum()),
            "inferred_frequency": inferred_frequency,
            "requested_ai_decisions": [
                key for key, value in options.items() if value == AI_DECIDE
            ],
        }
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key,
            temperature=0,
        )
        response = llm.invoke(
            "Choose preprocessing options for a univariate time-series forecast. "
            "Return only valid JSON with keys duplicate_strategy, missing_strategy, "
            "frequency, and continue_short_series. Valid duplicate_strategy values: "
            "sum, mean, latest. Valid missing_strategy values: interpolate, "
            "forward-fill, drop. Valid frequency values: D, W, MS, QS, YS. "
            "Valid continue_short_series values: continue, stop. "
            f"Dataset summary: {json.dumps(payload)}"
        )
        content = getattr(response, "content", str(response))
        llm_options = json.loads(content)
        return {
            **options,
            "duplicate_strategy": _valid_or_default(
                llm_options.get("duplicate_strategy"),
                ["sum", "mean", "latest"],
                heuristic_options["duplicate_strategy"],
            ),
            "missing_strategy": _valid_or_default(
                llm_options.get("missing_strategy"),
                ["interpolate", "forward-fill", "drop"],
                heuristic_options["missing_strategy"],
            ),
            "frequency": _valid_or_default(
                llm_options.get("frequency"),
                ["D", "W", "MS", "QS", "YS"],
                heuristic_options["frequency"],
            ),
            "continue_short_series": _valid_or_default(
                llm_options.get("continue_short_series"),
                ["continue", "stop"],
                heuristic_options["continue_short_series"],
            ),
        }
    except Exception as exc:
        logger.warning("AI preflight option resolution failed; using heuristics: %s", exc)
        return {**options, **heuristic_options}


def _valid_or_default(value: Any, allowed: list[str], default: str) -> str:
    return value if value in allowed else default


def _infer_frequency(df: pd.DataFrame) -> str:
    try:
        inferred = pd.infer_freq(df.index)
        if inferred:
            return _normalize_frequency(inferred)
    except Exception:
        pass

    if len(df) >= 2:
        deltas = df.index.to_series().diff().dropna()
        median_days = deltas.dt.days.median()
        if median_days <= 1:
            return "D"
        if median_days <= 7:
            return "W"
        if median_days <= 31:
            return "MS"
        if median_days <= 92:
            return "QS"
        return "YS"
    return "MS"


def _normalize_frequency(freq: str) -> str:
    f = (freq or "").upper().lstrip("-")
    if f.startswith("MS") or f.startswith("M"):
        return "MS"
    if f.startswith("QS") or f.startswith("Q"):
        return "QS"
    if f.startswith("W"):
        return "W"
    if f.startswith("D"):
        return "D"
    if f.startswith("YS") or f.startswith("Y") or f.startswith("A"):
        return "YS"
    return f
