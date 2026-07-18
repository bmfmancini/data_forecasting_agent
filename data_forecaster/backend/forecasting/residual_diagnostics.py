"""Residual diagnostics and uncertainty calibration.

This module implements the residual-diagnostics and interval-calibration
requirements:

* Return fitted innovations where supported and pooled backtest errors
  from rolling-origin folds. Never mix them under one ``residuals`` name.
* Apply diagnostics to appropriate error types: bias/mean error and
  confidence interval, residual/error ACF, Ljung-Box at relevant lags with
  fitted AR/MA degrees-of-freedom adjustment for ARIMA-family innovations,
  variance by horizon, and distribution/tail diagnostics as interval-
  assumption evidence.
* Calculate empirical coverage, average width, and interval/Winkler score by
  horizon.
* Suppress a nominal "95%" claim when coverage cannot be evaluated; label
  such output model-based or experimental.

The diagnostics are computed in Python and returned as typed
:class:`ResidualDiagnosticsResult` objects so the statistical review agent
and report builder consume real evidence rather than heuristic bands.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import shapiro, t

from core.logging_config import get_logger
from forecasting.contracts import ResidualDiagnosticsResult

logger = get_logger(__name__)

_ZERO_MEAN_P_THRESHOLD = 0.05
_AUTOCORRELATION_P_THRESHOLD = 0.05
_NORMALITY_P_THRESHOLD = 0.05
_NOMINAL_COVERAGE = 0.95


def _ljung_box(
    errors: np.ndarray,
    lags: int,
    df_adjust: int = 0,
) -> tuple[float | None, int]:
    """Compute the Ljung-Box statistic p-value with a df adjustment.

    Args:
        errors:     1-D array of residuals/errors.
        lags:       Number of lags to test.
        df_adjust:  Degrees-of-freedom adjustment (fitted AR+MA order for
                    ARIMA-family innovations).

    Returns:
        (p_value, lag_used). ``(None, lag)`` when the test cannot be computed.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox  # local import

    n = errors.size
    if n < 3:
        return None, lags
    lag = max(1, min(lags, n // 2))
    effective_df = max(1, lag - df_adjust)
    try:
        result = acorr_ljungbox(errors, lags=[lag], return_df=True)
        p_value = float(result["lb_pvalue"].iloc[0])
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Ljung-Box test failed: %s", exc)
        return None, lag
    # When df_adjust > 0 the nominal chi-square df is reduced. statsmodels
    # does not expose a df parameter, so we re-derive the p-value from the
    # statistic when an adjustment is requested.
    if df_adjust > 0 and "lb_stat" in result:
        stat = float(result["lb_stat"].iloc[0])
        from scipy.stats import chi2  # local import

        p_value = float(chi2.sf(stat, df=effective_df))
    return p_value, lag


def _mean_ci(errors: np.ndarray) -> tuple[float | None, float | None]:
    """Return the 95% confidence interval for the mean of ``errors``."""
    n = errors.size
    if n < 2:
        return None, None
    mean = float(np.mean(errors))
    se = float(np.std(errors, ddof=1) / math.sqrt(n))
    tcrit = float(t.ppf(0.975, df=n - 1))
    return mean - tcrit * se, mean + tcrit * se


def _variance_by_horizon(
    fold_residuals: Sequence[Sequence[float]],
) -> dict[int, float]:
    """Compute error variance keyed by horizon step across folds."""
    by_horizon: dict[int, list[float]] = {}
    for residuals in fold_residuals:
        for h, value in enumerate(residuals):
            by_horizon.setdefault(h, []).append(float(value))
    return {
        h: float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
        for h, values in sorted(by_horizon.items())
    }


def _interval_coverage(
    actuals: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float | None:
    """Empirical coverage: fraction of actuals inside the interval."""
    if (
        actuals.size == 0
        or lower.shape != actuals.shape
        or upper.shape != actuals.shape
    ):
        return None
    inside = (actuals >= lower) & (actuals <= upper)
    return float(np.mean(inside))


def _mean_width(lower: np.ndarray, upper: np.ndarray) -> float | None:
    """Average interval width."""
    if lower.size == 0 or upper.shape != lower.shape:
        return None
    return float(np.mean(upper - lower))


def _winkler_score(
    actuals: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_coverage: float = _NOMINAL_COVERAGE,
) -> float | None:
    """Mean Winkler interval score at the nominal coverage level.

    The Winkler score penalises both width and coverage failure. Lower is
    better. See Hyndman & Athanasopoulos, *Forecasting: principles and
    practice*, Section 5.8.
    """
    if (
        actuals.size == 0
        or lower.shape != actuals.shape
        or upper.shape != actuals.shape
    ):
        return None
    alpha = 1.0 - nominal_coverage
    width = upper - lower
    lower_penalty = 2.0 / alpha * (lower - actuals)
    upper_penalty = 2.0 / alpha * (actuals - upper)
    score = width.copy()
    below = actuals < lower
    above = actuals > upper
    score[below] += lower_penalty[below]
    score[above] += upper_penalty[above]
    return float(np.mean(score))


def analyze_innovations(
    innovations: np.ndarray | pd.Series,
    *,
    ar_ma_order: int = 0,
    disabled_tests: list[str] | None = None,
) -> ResidualDiagnosticsResult:
    """Run diagnostics on fitted innovations.

    Args:
        innovations: Fitted one-step-ahead innovations (residuals).
        ar_ma_order:  Sum of fitted AR and MA orders for the Ljung-Box
                      degrees-of-freedom adjustment (ARIMA-family only).
        disabled_tests: Tests to skip (``residual_zero_mean``,
                         ``residual_autocorrelation``, ``residual_normality``).

    Returns:
        :class:`ResidualDiagnosticsResult` with ``error_type="innovations"``.
    """
    disabled = set(disabled_tests or [])
    errors = np.asarray(innovations, dtype=float)
    errors = errors[np.isfinite(errors)]
    n = errors.size
    warnings: list[str] = []

    if n == 0:
        return ResidualDiagnosticsResult(
            error_type="innovations",
            warnings=["No finite innovations available for diagnostics."],
        )

    mean = float(np.mean(errors))
    ci_lower, ci_upper = _mean_ci(errors)

    is_zero_mean = None
    if "residual_zero_mean" not in disabled and n >= 2:
        ci_lower, ci_upper = _mean_ci(errors)
        is_zero_mean = (
            ci_lower is not None
            and ci_upper is not None
            and ci_lower <= 0.0 <= ci_upper
        )

    ljung_p: float | None = None
    lag_used: int | None = None
    is_uncorrelated = None
    if "residual_autocorrelation" not in disabled:
        lags = min(10, max(1, n // 5))
        ljung_p, lag_used = _ljung_box(errors, lags, df_adjust=ar_ma_order)
        if ljung_p is not None:
            is_uncorrelated = ljung_p >= _AUTOCORRELATION_P_THRESHOLD

    shapiro_p = None
    is_normal = None
    if "residual_normality" not in disabled and 3 <= n <= 5000:
        try:
            _, shapiro_p = shapiro(errors)
            shapiro_p = float(shapiro_p)
            is_normal = shapiro_p >= _NORMALITY_P_THRESHOLD
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Shapiro-Wilk test failed: %s", exc)
            warnings.append("Normality test could not be computed.")

    return ResidualDiagnosticsResult(
        error_type="innovations",
        n_errors=n,
        mean=mean,
        mean_ci_lower=ci_lower,
        mean_ci_upper=ci_upper,
        is_zero_mean=is_zero_mean,
        ljung_box_p_value=ljung_p,
        ljung_box_lag=lag_used,
        ljung_box_df_adjust=ar_ma_order,
        is_uncorrelated=is_uncorrelated,
        shapiro_p_value=shapiro_p,
        is_normal=is_normal,
        nominal_coverage=_NOMINAL_COVERAGE,
        coverage_estimable=False,
        warnings=warnings,
    )


def _compute_interval_metrics(
    fold_actuals: Sequence[Sequence[float]],
    fold_lower: Sequence[Sequence[float] | None],
    fold_upper: Sequence[Sequence[float] | None],
    nominal_coverage: float,
) -> tuple[float | None, float | None, float | None, bool]:
    """Compute empirical coverage, mean width, and Winkler score.

    Returns:
        (coverage, width, winkler_score, coverage_estimable). All ``None``
        when no aligned interval bounds are available.
    """
    actuals_list: list[float] = []
    lower_list: list[float] = []
    upper_list: list[float] = []
    for a, lo, hi in zip(fold_actuals, fold_lower, fold_upper):
        if lo is None or hi is None:
            continue
        a_arr = np.asarray(a, dtype=float)
        lo_arr = np.asarray(lo, dtype=float)
        hi_arr = np.asarray(hi, dtype=float)
        min_len = min(a_arr.size, lo_arr.size, hi_arr.size)
        actuals_list.extend(a_arr[:min_len].tolist())
        lower_list.extend(lo_arr[:min_len].tolist())
        upper_list.extend(hi_arr[:min_len].tolist())
    if not actuals_list:
        return None, None, None, False
    actuals_arr = np.asarray(actuals_list, dtype=float)
    lower_arr = np.asarray(lower_list, dtype=float)
    upper_arr = np.asarray(upper_list, dtype=float)
    coverage = _interval_coverage(actuals_arr, lower_arr, upper_arr)
    width = _mean_width(lower_arr, upper_arr)
    winkler = _winkler_score(actuals_arr, lower_arr, upper_arr, nominal_coverage)
    return coverage, width, winkler, coverage is not None


def analyze_backtest_errors(
    fold_residuals: Sequence[Sequence[float]],
    *,
    fold_actuals: Sequence[Sequence[float]] | None = None,
    fold_lower: Sequence[Sequence[float] | None] | None = None,
    fold_upper: Sequence[Sequence[float] | None] | None = None,
    disabled_tests: list[str] | None = None,
    nominal_coverage: float = _NOMINAL_COVERAGE,
) -> ResidualDiagnosticsResult:
    """Run diagnostics on pooled backtest errors from rolling-origin folds.

    Args:
        fold_residuals: Per-fold residuals (actuals - predictions).
        fold_actuals:   Per-fold actuals (required for interval coverage).
        fold_lower:     Per-fold lower prediction-interval bounds (or ``None``).
        fold_upper:     Per-fold upper prediction-interval bounds (or ``None``).
        disabled_tests: Tests to skip.
        nominal_coverage: Nominal coverage level for interval scoring.

    Returns:
        :class:`ResidualDiagnosticsResult` with ``error_type="backtest_errors"``
        and interval coverage/width/Winkler score when interval bounds are
        supplied.
    """
    disabled = set(disabled_tests or [])
    pooled = np.asarray(
        [float(v) for fold in fold_residuals for v in fold], dtype=float
    )
    pooled = pooled[np.isfinite(pooled)]
    n = pooled.size
    warnings: list[str] = []

    if n == 0:
        return ResidualDiagnosticsResult(
            error_type="backtest_errors",
            warnings=["No finite backtest errors available for diagnostics."],
        )

    mean = float(np.mean(pooled))
    ci_lower, ci_upper = _mean_ci(pooled)

    is_zero_mean = None
    if "residual_zero_mean" not in disabled and n >= 2:
        ci_lower, ci_upper = _mean_ci(pooled)
        is_zero_mean = (
            ci_lower is not None
            and ci_upper is not None
            and ci_lower <= 0.0 <= ci_upper
        )

    ljung_p: float | None = None
    lag_used: int | None = None
    is_uncorrelated = None
    if "residual_autocorrelation" not in disabled:
        lags = min(10, max(1, n // 5))
        ljung_p, lag_used = _ljung_box(pooled, lags, df_adjust=0)
        if ljung_p is not None:
            is_uncorrelated = ljung_p >= _AUTOCORRELATION_P_THRESHOLD

    shapiro_p = None
    is_normal = None
    if "residual_normality" not in disabled and 3 <= n <= 5000:
        try:
            _, shapiro_p = shapiro(pooled)
            shapiro_p = float(shapiro_p)
            is_normal = shapiro_p >= _NORMALITY_P_THRESHOLD
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Shapiro-Wilk test failed: %s", exc)
            warnings.append("Normality test could not be computed.")

    variance_by_horizon = _variance_by_horizon(fold_residuals)

    # Interval coverage / width / Winkler score.
    coverage: float | None = None
    width: float | None = None
    winkler: float | None = None
    coverage_estimable = False
    coverage_by_horizon: dict[int, float] = {}
    width_by_horizon: dict[int, float] = {}
    winkler_by_horizon: dict[int, float] = {}
    if fold_actuals is not None and fold_lower is not None and fold_upper is not None:
        coverage, width, winkler, coverage_estimable = _compute_interval_metrics(
            fold_actuals, fold_lower, fold_upper, nominal_coverage
        )
        max_horizon = max((len(values) for values in fold_actuals), default=0)
        for horizon in range(max_horizon):
            aligned = [
                (actual[horizon], lower[horizon], upper[horizon])
                for actual, lower, upper in zip(fold_actuals, fold_lower, fold_upper)
                if lower is not None
                and upper is not None
                and len(actual) > horizon
                and len(lower) > horizon
                and len(upper) > horizon
            ]
            if not aligned:
                continue
            actual_arr = np.asarray([values[0] for values in aligned], dtype=float)
            lower_arr = np.asarray([values[1] for values in aligned], dtype=float)
            upper_arr = np.asarray([values[2] for values in aligned], dtype=float)
            step_coverage = _interval_coverage(actual_arr, lower_arr, upper_arr)
            step_width = _mean_width(lower_arr, upper_arr)
            step_winkler = _winkler_score(
                actual_arr, lower_arr, upper_arr, nominal_coverage
            )
            step = horizon + 1
            if step_coverage is not None:
                coverage_by_horizon[step] = step_coverage
            if step_width is not None:
                width_by_horizon[step] = step_width
            if step_winkler is not None:
                winkler_by_horizon[step] = step_winkler

    return ResidualDiagnosticsResult(
        error_type="backtest_errors",
        n_errors=n,
        mean=mean,
        mean_ci_lower=ci_lower,
        mean_ci_upper=ci_upper,
        is_zero_mean=is_zero_mean,
        ljung_box_p_value=ljung_p,
        ljung_box_lag=lag_used,
        ljung_box_df_adjust=0,
        is_uncorrelated=is_uncorrelated,
        shapiro_p_value=shapiro_p,
        is_normal=is_normal,
        variance_by_horizon=variance_by_horizon,
        interval_coverage=coverage,
        interval_mean_width=width,
        winkler_score=winkler,
        interval_coverage_by_horizon=coverage_by_horizon,
        interval_width_by_horizon=width_by_horizon,
        winkler_score_by_horizon=winkler_by_horizon,
        nominal_coverage=nominal_coverage,
        weighted_interval_score=(
            winkler * (1.0 - nominal_coverage) / 2.0 if winkler is not None else None
        ),
        coverage_estimable=coverage_estimable,
        warnings=warnings,
    )


def interval_nonconformity_scores(
    actual: np.ndarray | list[float],
    lower: np.ndarray | list[float],
    upper: np.ndarray | list[float],
) -> list[float]:
    """Calculate two-sided conformal nonconformity scores.

    A score is zero for an observation inside its interval and otherwise is
    the distance to the nearest violated bound. Non-finite or misaligned
    inputs return no calibration evidence.

    Args:
        actual: Observed validation values.
        lower: Validation lower interval bounds.
        upper: Validation upper interval bounds.

    Returns:
        Non-negative scores aligned to the validation observations.
    """
    observed = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if observed.shape != lo.shape or observed.shape != hi.shape:
        return []
    finite = np.isfinite(observed) & np.isfinite(lo) & np.isfinite(hi)
    if not np.all(finite) or np.any(lo > hi):
        return []
    scores = np.maximum.reduce((lo - observed, observed - hi, np.zeros_like(observed)))
    return scores.astype(float).tolist()


def calibrate_interval_width(
    lower: np.ndarray | list[float],
    upper: np.ndarray | list[float],
    *,
    calibration_scores: np.ndarray | list[float],
    nominal_coverage: float = _NOMINAL_COVERAGE,
) -> tuple[list[float], list[float]]:
    """Conformalize future intervals using held-out nonconformity scores.

    This uses the finite-sample split-conformal ``higher`` quantile. Aggregate
    empirical coverage alone is deliberately insufficient: calibration needs
    the actual distances by which validation observations missed their bounds.

    Args:
        lower: Future lower prediction-interval bounds.
        upper: Future upper prediction-interval bounds.
        calibration_scores: Held-out nonconformity scores.
        nominal_coverage: Target marginal coverage in ``(0, 1)``.

    Returns:
        Conformalized lower and upper bounds. With fewer than ten usable
        scores, returns the original bounds rather than claiming calibration.
    """
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    scores = np.asarray(calibration_scores, dtype=float)
    scores = scores[np.isfinite(scores) & (scores >= 0.0)]
    if (
        lo.shape != hi.shape
        or np.any(~np.isfinite(lo))
        or np.any(~np.isfinite(hi))
        or np.any(lo > hi)
        or not 0.0 < nominal_coverage < 1.0
        or scores.size < 10
    ):
        return lo.tolist(), hi.tolist()
    rank = min(scores.size, math.ceil((scores.size + 1) * nominal_coverage))
    adjustment = float(np.sort(scores)[rank - 1])
    return (lo - adjustment).tolist(), (hi + adjustment).tolist()
