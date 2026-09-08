"""Empirical interval expansion using errors that precede the final audit.

This is a small-sample heuristic, not a distribution-free coverage guarantee.
The selected procedure shares selection data with calibration; only the final
holdout supplies an untouched coverage audit.
"""

from __future__ import annotations

import numpy as np

from forecasting.contracts import ForecastFitStatus


def fit_interval_calibration(evaluation, series, *, minimum_origins: int = 5) -> dict:
    """Estimate nonnegative 95th-percentile misses separately for each horizon.

    Use disjoint validation windows to avoid counting overlapping origins as
    independent observations. Missing actuals do not contribute scores.
    """
    scores: dict[int, list[float]] = {}
    previous_end = -1
    used = []
    for fold in sorted(evaluation.folds, key=lambda item: item.fold.test_start_index):
        if (
            fold.status != ForecastFitStatus.OK
            or fold.fold.test_start_index < previous_end
        ):
            continue
        actual = np.asarray(
            series.iloc[fold.fold.test_start_index : fold.fold.test_end_index],
            dtype=float,
        )
        lower, upper = np.asarray(fold.lower_ci), np.asarray(fold.upper_ci)
        if lower.shape != actual.shape or upper.shape != actual.shape:
            continue
        previous_end = fold.fold.test_end_index
        used.append(fold.fold.fold_index)
        for h, (y, lo, hi) in enumerate(zip(actual, lower, upper), 1):
            if np.isfinite([y, lo, hi]).all() and lo <= hi:
                scores.setdefault(h, []).append(float(max(lo - y, y - hi, 0)))
    horizons = {}
    for h in range(
        1, int(evaluation.validation_design.get("requested_horizon", 0)) + 1
    ):
        values = scores.get(h, [])
        horizons[str(h)] = {
            "n_origins": len(values),
            "status": (
                "estimated"
                if len(values) >= minimum_origins
                else "insufficient_evidence"
            ),
            "expansion": (
                float(np.quantile(values, 0.95, method="higher"))
                if len(values) >= minimum_origins
                else None
            ),
        }
    return {
        "method": "empirical_backtest_expansion",
        "nominal_coverage": 0.95,
        "minimum_origins": minimum_origins,
        "origin_indices": used,
        "by_horizon": horizons,
        "final_test_used_for_calibration": False,
        "interpretation": "Empirical expansion using selection folds; coverage is not guaranteed. Audit uses the untouched final test.",
    }


def apply_interval_calibration(
    lower, upper, calibration: dict, *, minimum=None, maximum=None
):
    """Widen available model intervals; leave unsupported horizons unchanged."""
    lo, hi = list(lower), list(upper)
    applied = []
    for i in range(min(len(lo), len(hi))):
        amount = calibration.get("by_horizon", {}).get(str(i + 1), {}).get("expansion")
        if amount is None or not np.isfinite([lo[i], hi[i]]).all() or lo[i] > hi[i]:
            continue
        lo[i] -= amount
        hi[i] += amount
        if minimum is not None:
            lo[i] = max(float(minimum), lo[i])
        if maximum is not None:
            hi[i] = min(float(maximum), hi[i])
        applied.append(i + 1)
    return lo, hi, applied


def audit_interval_calibration(fold, series, calibration, **bounds) -> dict:
    """Compare original and adjusted intervals on the same untouched actuals."""
    if fold is None or fold.status != ForecastFitStatus.OK:
        return {
            "status": "unavailable",
            "reason": "No successful untouched final test.",
        }
    lower, upper, applied = apply_interval_calibration(
        fold.lower_ci, fold.upper_ci, calibration, **bounds
    )
    actual = series.iloc[fold.fold.test_start_index : fold.fold.test_end_index]
    rows = {}
    for i, (y, lo, hi, old_lo, old_hi) in enumerate(
        zip(actual, lower, upper, fold.lower_ci, fold.upper_ci), 1
    ):
        if not np.isfinite([y, lo, hi, old_lo, old_hi]).all():
            continue
        rows[str(i)] = {
            "n_observed": 1,
            "adjustment_applied": i in applied,
            "original_coverage": float(old_lo <= y <= old_hi),
            "adjusted_coverage": float(lo <= y <= hi),
            "original_width": float(old_hi - old_lo),
            "adjusted_width": float(hi - lo),
        }
    return {
        "status": "audited" if rows else "unavailable",
        "by_horizon": rows,
        "n_observed": len(rows),
        "interpretation": "One final-test observation per horizon; insufficient to establish reliable coverage.",
    }
