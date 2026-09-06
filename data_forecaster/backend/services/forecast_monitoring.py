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


def monitor_forecasts(job_ids: list[str], requester: dict) -> dict:
    """Compare authorized vintages of one declared series, in forecast-date order.

    Alerts are operational review rules, not statistical drift tests. Repeated
    submissions for the same first forecast timestamp count as one vintage.
    """
    from services.job_service import get_job

    if not job_ids or len(job_ids) > 100 or len(set(job_ids)) != len(job_ids):
        raise ValueError("Supply 1–100 distinct job IDs.")
    records = []
    identity = None
    frequency = None
    point_quantile = None
    for job_id in job_ids:
        job = get_job(job_id, requester=requester)
        if job is None:
            raise LookupError("Job not found.")
        options = json.loads(job["preflight_options"])
        key = (
            job["backend_owner_id"],
            job["application_user_id"],
            options.get("monitoring_series_id") or job["file_id"],
            job["date_col"],
            job["value_col"],
            *[
                json.dumps(options.get(field), sort_keys=True)
                for field in (
                    "frequency",
                    "aggregation",
                    "units",
                    "minimum_value",
                    "maximum_value",
                )
            ],
        )
        if identity is not None and key != identity:
            raise ValueError(
                "Jobs must describe the same series, owner, columns, units, frequency, aggregation, and bounds. Use monitoring_series_id to identify a series across uploads."
            )
        identity = key
        result = monitor_forecast(job_id, requester)
        with transaction() as connection:
            row = connection.execute(
                "SELECT forecast_json, issued_at FROM forecast_snapshots WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        snapshot = json.loads(row["forecast_json"])
        current_quantile = (
            snapshot.get("validation_design", {})
            .get("decision_loss", {})
            .get("quantile")
            or 0.5
        )
        if point_quantile is not None and current_quantile != point_quantile:
            raise ValueError("Jobs must use the same forecast quantile.")
        point_quantile = current_quantile
        actual_frequency = snapshot.get("validation_design", {}).get("frequency")
        dates = pd.DatetimeIndex(snapshot["forecast_dates"])
        if actual_frequency is None and len(dates) >= 3:
            actual_frequency = pd.infer_freq(dates)
        if frequency is not None and actual_frequency != frequency:
            raise ValueError("Jobs must use the same forecast frequency.")
        frequency = actual_frequency
        records.append(
            (snapshot["forecast_dates"][0], row["issued_at"], job_id, result, snapshot)
        )
    # Keep the latest issued revision of a forecast origin; input order is irrelevant.
    unique = {}
    for record in sorted(
        records, key=lambda item: (pd.Timestamp(item[0]), item[1], item[2])
    ):
        unique[_timestamp(record[0])] = record
    vintages = list(unique.values())

    def summarize(items):
        actual, predicted, errors = [], [], []
        covered = []
        baseline_errors: dict[str, list[tuple[float, float]]] = {}
        by_horizon: dict[str, list[dict]] = {}
        for _, _, _, result, snapshot in items:
            for h, point in result["by_horizon"].items():
                if point["actual"] is None:
                    continue
                i = int(h) - 1
                actual.append(point["actual"])
                predicted.append(point["forecast"])
                errors.append(point["error"])
                if point["interval_coverage"] is not None:
                    covered.append(point["interval_coverage"])
                by_horizon.setdefault(h, []).append(point)
                for name, values in (
                    snapshot.get("validation_design", {})
                    .get("monitoring_baselines", {})
                    .items()
                ):
                    if i < len(values) and math.isfinite(values[i]):
                        baseline_errors.setdefault(name, []).append(
                            (abs(point["error"]), abs(point["actual"] - values[i]))
                        )
        skill = {}
        for name, pairs in baseline_errors.items():
            model_error, baseline_error = np.asarray(pairs).sum(axis=0)
            skill[name] = {
                "mae_skill": (
                    float(1 - model_error / baseline_error)
                    if baseline_error > 0
                    else None
                ),
                "n_observed": len(pairs),
            }
        return {
            "n_vintages": len(items),
            "n_observed": len(actual),
            "n_unique_actual_dates": len(
                {p["timestamp"] for rows in by_horizon.values() for p in rows}
            ),
            "metrics": calculate_forecast_metrics(
                actual, predicted, quantile=point_quantile or 0.5
            ).model_dump(),
            "bias_actual_minus_forecast": float(np.mean(errors)) if errors else None,
            "interval_coverage": float(np.mean(covered)) if covered else None,
            "n_intervals": len(covered),
            "baseline_skill": skill,
            "by_horizon": {
                h: {
                    "n_observed": len(rows),
                    "mae": float(np.mean([abs(p["error"]) for p in rows])),
                    "bias_actual_minus_forecast": float(
                        np.mean([p["error"] for p in rows])
                    ),
                    "interval_coverage": (
                        float(
                            np.mean(
                                [
                                    p["interval_coverage"]
                                    for p in rows
                                    if p["interval_coverage"] is not None
                                ]
                            )
                        )
                        if any(p["interval_coverage"] is not None for p in rows)
                        else None
                    ),
                }
                for h, rows in by_horizon.items()
            },
        }

    # Compare only fully observed vintages with identical horizon lengths. This
    # avoids treating a recent run's still-missing distant horizons as improvement.
    complete = [r for r in vintages if r[3]["n_pending"] == 0]
    alerts = []
    earlier = recent = None
    ready = False
    recent_items = []
    if complete:
        length = len(complete[-1][4]["forecast_dates"])
        comparable = [r for r in complete if len(r[4]["forecast_dates"]) == length]

        def take_window(items):
            chosen = []
            dates = set()
            count = 0
            for item in reversed(items):
                chosen.append(item)
                dates.update(item[4]["forecast_dates"])
                count += item[3]["n_observed"]
                if len(chosen) >= 5 and count >= 20 and len(dates) >= 20:
                    return list(reversed(chosen))
            return []

        recent_items = take_window(comparable)
        earlier_items = (
            take_window(comparable[: -len(recent_items)]) if recent_items else []
        )
        if recent_items and earlier_items:
            earlier, recent = summarize(earlier_items), summarize(recent_items)
            ready = True
            if ready:
                biases = [r[3]["bias_actual_minus_forecast"] for r in recent_items]
                if all(b > 0 for b in biases) or all(b < 0 for b in biases):
                    alerts.append(
                        {
                            "kind": "persistent_bias",
                            "direction": (
                                "underforecast" if biases[0] > 0 else "overforecast"
                            ),
                        }
                    )
                for name, current in recent["baseline_skill"].items():
                    prior = earlier["baseline_skill"].get(name, {})
                    if (
                        current["n_observed"] >= 20
                        and prior.get("n_observed", 0) >= 20
                        and current["mae_skill"] is not None
                        and prior.get("mae_skill") is not None
                        and current["mae_skill"] < 0
                        and current["mae_skill"] < prior["mae_skill"] - 0.1
                    ):
                        alerts.append(
                            {"kind": "declining_baseline_skill", "baseline": name}
                        )
                if (
                    min(earlier["n_intervals"], recent["n_intervals"]) >= 20
                    and recent["interval_coverage"] < 0.9
                    and earlier["interval_coverage"] - recent["interval_coverage"]
                    >= 0.1
                ):
                    alerts.append({"kind": "declining_interval_coverage"})
    return {
        "job_ids": [r[2] for r in vintages],
        "duplicate_origins_excluded": len(records) - len(vintages),
        "summary": summarize(vintages),
        "earlier": earlier,
        "recent": recent,
        "alert_status": "evaluated" if ready else "insufficient_evidence",
        "alerts": alerts,
        "alert_policy": {
            "minimum_vintages_per_window": 5,
            "minimum_observations_per_window": 20,
            "minimum_unique_actual_dates_per_window": 20,
            "complete_same_horizon_vintages_only": True,
        },
        "interpretation": "Descriptive comparisons and review alerts; overlapping forecast errors are dependent. No statistical drift claim or automatic retraining.",
    }
