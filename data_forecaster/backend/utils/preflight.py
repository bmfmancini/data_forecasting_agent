"""Preflight checks and recommendations for uploaded time series data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from schemas import PreflightDecision, PreflightResponse
from utils.data_cleaning import (
    detect_outliers_iqr,
    reindex_series,
    resolve_duplicates,
)

AGGREGATION_OPTIONS = ["Let AI Decide", "sum", "mean", "latest"]
MISSING_OPTIONS = ["Let AI Decide", "interpolate", "forward-fill", "drop"]
FREQUENCY_OPTIONS = ["Let AI Decide", "D", "W", "MS", "QS", "YS"]
OUTLIER_OPTIONS = [
    "None",
    "Let AI Decide",
    "Clip (Winsorize)",
    "Remove",
    "Z-Score Clip",
]
DOMAIN_OPTIONS = [
    "Skip / Let AI Guess",
    "General / Unknown",
    "Retail / Sales",
    "Network Traffic / IoT",
    "Finance / Stock",
    "Weather / Climate",
    "Other (Custom)",
]


def run_preflight_checks(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    forecast_horizon: int,
) -> PreflightResponse:
    """Inspect the selected series and return any user decisions needed.

    Performs a structured audit (mirroring the freeCodeCamp cleaning
    checklist) covering missing timestamps, duplicate timestamps, null
    values, regularity, frequency and outlier counts.  It then surfaces a
    list of :class:`PreflightDecision` objects the user must resolve
    (e.g. duplicate handling, missing-value imputation, outlier treatment)
    before forecasting can begin.

    Args:
        df: Source DataFrame containing the time series.
        date_col: Name of the datetime column.
        value_col: Name of the numeric value column.
        forecast_horizon: Number of future steps the user wishes to
            forecast (used to surface length-vs-horizon warnings).

    Returns:
        A :class:`PreflightResponse` describing the detected issues,
        warnings and any required user decisions.
    """
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

    outlier_info = detect_outliers_iqr(series.dropna())

    issues: list[str] = []
    warnings: list[str] = []
    decisions: list[PreflightDecision] = []
    defaults: dict[str, Any] = {
        "duplicate_strategy": "Let AI Decide",
        "missing_strategy": "Let AI Decide",
        "frequency": "Let AI Decide",
        "data_domain": "Skip / Let AI Guess",
        "outlier_strategy": "Let AI Decide",
        "continue_short_series": "continue",
        "loss_metric": "mase",
        "units": "Unspecified",
        "interventions": "None known",
        "censoring_or_stockouts": "None known",
        "known_future_covariates": "None",
        "aggregation": "As provided",
        "minimum_value": None,
        "maximum_value": None,
    }

    if duplicate_ts:
        issues.append(f"{duplicate_ts} duplicate timestamp(s) found.")
        decisions.append(
            PreflightDecision(
                key="duplicate_strategy",
                label="Duplicate timestamp handling",
                message="Choose how repeated timestamps should be combined before forecasting.",
                options=AGGREGATION_OPTIONS,
                default="Let AI Decide",
            )
        )

    if missing_values:
        issues.append(f"{missing_values} missing value(s) found in '{value_col}'.")
        decisions.append(
            PreflightDecision(
                key="missing_strategy",
                label="Missing value handling",
                message="Choose how missing values should be handled before modeling.",
                options=MISSING_OPTIONS,
                default="Let AI Decide",
            )
        )

    if missing_ts or not is_regular:
        issues.append("Irregular intervals or timestamp gaps found.")
        decisions.append(
            PreflightDecision(
                key="frequency",
                label="Forecast frequency",
                message="Choose the regular frequency to use for the forecast output.",
                options=FREQUENCY_OPTIONS,
                default="Let AI Decide",
            )
        )

    decisions.append(
        PreflightDecision(
            key="data_domain",
            label="Data Domain Context",
            message="What is the domain of this data? This helps the AI choose the best cleaning strategy.",
            options=DOMAIN_OPTIONS,
            default="Skip / Let AI Guess",
            required=True,
            allow_custom=True,
        )
    )
    decisions.extend(
        [
            PreflightDecision(
                key="loss_metric",
                label="Decision loss",
                message="Which out-of-sample loss should control model ranking?",
                options=["mase", "rmse", "mae", "wape"],
                default="mase",
            ),
            PreflightDecision(
                key="units",
                label="Target units",
                message="What units does the forecast target use?",
                options=["Unspecified", "Count", "Currency", "Rate", "Other"],
                default="Unspecified",
                allow_custom=True,
            ),
            PreflightDecision(
                key="interventions",
                label="Known interventions",
                message="List promotions, outages, policy changes, or other interventions.",
                options=["None known", "Known events"],
                default="None known",
                allow_custom=True,
            ),
            PreflightDecision(
                key="censoring_or_stockouts",
                label="Censoring or stockouts",
                message="Can recorded values be capped, censored, or limited by stockouts?",
                options=["None known", "Possible", "Confirmed"],
                default="None known",
                allow_custom=True,
            ),
            PreflightDecision(
                key="known_future_covariates",
                label="Future information",
                message="Are future holidays, prices, schedules, or covariates known?",
                options=["None", "Available"],
                default="None",
                allow_custom=True,
            ),
        ]
    )

    if outlier_info["count"] > 0:
        warnings.append(f"Detected {outlier_info['count']} potential outliers.")
        decisions.append(
            PreflightDecision(
                key="outlier_strategy",
                label="Outlier Handling",
                message=f"Detected {outlier_info['count']} outliers. How should they be treated?",
                options=OUTLIER_OPTIONS,
                default="Let AI Decide",
            )
        )

    if usable_observations < 20:
        warnings.append(
            f"Only {usable_observations} usable observation(s) are available; forecast reliability may be low."
        )
        decisions.append(
            PreflightDecision(
                key="continue_short_series",
                label="Short series confirmation",
                message="Confirm that you want to continue with a short time series.",
                options=["continue", "stop"],
                default="continue",
            )
        )

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
    options: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Apply preflight choices and return a clean two-column time series frame.

    The pipeline executes the following stages in order, each backed by a
    helper in :mod:`utils.data_cleaning`:

    1. Resolve automatic (``"Let AI Decide"``) user selections to defaults.
    2. Aggregate or drop duplicate timestamps.
    3. Reindex the series onto a canonical frequency grid.
    4. Preserve missing values and model-affecting preprocessing choices for
       training-window fitting during rolling-origin evaluation.

    Args:
        df: Source DataFrame containing the time series.
        date_col: Name of the datetime column.
        value_col: Name of the numeric value column.
        options: Mapping of user-selected preflight decisions.

    Returns:
        A tuple of the prepared two-column DataFrame and the resolved
        frequency alias.
    """
    options = options or {}
    selected = _selected_frame(df, date_col, value_col)

    frequency = options.get("frequency")
    if not frequency or frequency == "Let AI Decide":
        frequency = _infer_frequency(selected.set_index(date_col))

    duplicate_strategy = options.get("duplicate_strategy", "mean")
    if duplicate_strategy == "Let AI Decide":
        duplicate_strategy = "mean"

    missing_strategy = options.get("missing_strategy", "interpolate")
    if missing_strategy == "Let AI Decide":
        missing_strategy = "interpolate"

    series = selected.set_index(date_col)[value_col].sort_index()

    if series.index.has_duplicates:
        series = resolve_duplicates(series, duplicate_strategy)

    if frequency:
        series = reindex_series(series, frequency)

    outlier_strategy = options.get("outlier_strategy", "None")
    if outlier_strategy == "Let AI Decide":
        # Diagnostics may flag anomalies, but automatic full-series clipping
        # would leak future distributional information into backtests.
        outlier_strategy = "None"
    # Model-affecting preprocessing is intentionally deferred. Rolling-origin
    # evaluation fits imputation, clipping, and smoothing independently within
    # each training window; the production refit applies them to full history
    # only after deterministic selection.
    if missing_strategy == "drop":
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
        raise ValueError(
            "No usable timestamps were found for the selected date column."
        )
    return selected


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
