"""Persist issued forecasts and score subsequently supplied observations."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from core.database import transaction
from forecasting.metrics import calculate_forecast_metrics
from forecasting.residual_diagnostics import analyze_backtest_errors


def save_snapshot(job_id: str, forecast: dict[str, Any]) -> None:
    """An issued forecast is immutable, even after actuals become available."""
    with transaction() as connection:
        connection.execute(
            "INSERT OR IGNORE INTO forecast_snapshots (job_id, forecast_json) VALUES (?, ?)",
            (job_id, json.dumps(forecast, allow_nan=False)),
        )


def _timestamp(value: str) -> str:
    return pd.Timestamp(value).isoformat()


def monitor_forecast(
    job_id: str, requester: dict, actuals: dict[str, float] | None = None
) -> dict:
    """Score one authorized forecast vintage, retaining its original baselines.

    Metrics are descriptive, not significance tests. A single vintage provides
    only one observation per horizon; sample sizes are always returned.
    """
    from services.job_service import get_job

    if get_job(job_id, requester=requester) is None:
        raise LookupError("Job not found.")
    with transaction() as connection:
        row = connection.execute(
            "SELECT forecast_json FROM forecast_snapshots WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise LookupError("No issued forecast snapshot is available for this job.")
        forecast = json.loads(row["forecast_json"])
        dates = [_timestamp(value) for value in forecast["forecast_dates"]]
        incoming = {
            _timestamp(date): float(value) for date, value in (actuals or {}).items()
        }
        if any(
            date not in dates or not math.isfinite(value)
            for date, value in incoming.items()
        ):
            raise ValueError(
                "Actuals must be finite and match issued forecast timestamps."
            )
        connection.executemany(
            "INSERT INTO forecast_actuals (job_id, timestamp, actual) VALUES (?, ?, ?) "
            "ON CONFLICT(job_id, timestamp) DO UPDATE SET actual=excluded.actual, recorded_at=datetime('now')",
            [(job_id, date, value) for date, value in incoming.items()],
        )
        recorded = {
            row["timestamp"]: row["actual"]
            for row in connection.execute(
                "SELECT timestamp, actual FROM forecast_actuals WHERE job_id = ?",
                (job_id,),
            )
        }
    actual = np.asarray([recorded.get(date, np.nan) for date in dates])
    point = np.asarray(forecast["forecast"])
    quantile = (
        forecast.get("validation_design", {}).get("decision_loss", {}).get("quantile")
        or 0.5
    )
    metrics = calculate_forecast_metrics(actual, point, quantile=quantile)
    observed = np.isfinite(actual)
    bias = (
        float(np.mean(actual[observed] - point[observed])) if observed.any() else None
    )
    baselines = forecast.get("validation_design", {}).get("monitoring_baselines", {})
    skill = {}
    for name, predictions in baselines.items():
        baseline = calculate_forecast_metrics(actual, predictions)
        if baseline.mae is not None and baseline.mae > 0 and metrics.mae is not None:
            skill[f"mae_skill_vs_{name}"] = 1 - metrics.mae / baseline.mae
    interval = analyze_backtest_errors(
        [(actual - point).tolist()],
        fold_actuals=[actual.tolist()],
        fold_lower=[forecast.get("lower_ci") or None],
        fold_upper=[forecast.get("upper_ci") or None],
    )
    return {
        "job_id": job_id,
        "model": forecast["model_used"],
        "n_observed": int(observed.sum()),
        "n_pending": int((~observed).sum()),
        "metrics": metrics.model_dump(),
        "bias_actual_minus_forecast": bias,
        "skill_scores": skill,
        "interval_coverage": interval.interval_coverage,
        "winkler_score": interval.winkler_score,
        "by_horizon": {
            str(i + 1): {
                "timestamp": date,
                "actual": recorded.get(date),
                "forecast": float(point[i]),
                "n_observed": int(observed[i]),
                "error": float(actual[i] - point[i]) if observed[i] else None,
                "interval_coverage": interval.interval_coverage_by_horizon.get(i + 1),
            }
            for i, date in enumerate(dates)
        },
        "interpretation": "Descriptive monitoring of one forecast vintage; no statistical drift claim.",
    }
