"""Exponentially weighted moving average (simple exponential smoothing) adapter.

The adapter estimates ``alpha`` by minimizing one-step-ahead squared error on
the training split, evaluates holdout metrics centrally, then refits on the
full series for the production forecast. The multi-step point forecast is
flat (the final estimated level), which is the correct SES behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from core.logging_config import get_logger
from forecasting.contracts import (
    ForecastAdapterResult,
    ForecastFitStatus,
    ForecastMetrics,
)
from forecasting.evaluation import evaluate_predictions, make_terminal_holdout

logger = get_logger(__name__)

# Grid of candidate alpha values for SSE-based estimation.
_ALPHA_GRID = np.linspace(0.01, 0.99, 99)


def _estimate_alpha(train: pd.Series) -> float:
    """Estimate the SES smoothing parameter by minimizing one-step SSE.

    Args:
        train: Training observations.

    Returns:
        The alpha value from a fixed grid that minimizes in-sample SSE.
        Falls back to ``0.3`` when estimation is not possible.
    """
    if len(train) < 3:
        return 0.3

    best_alpha = 0.3
    best_sse = float("inf")
    for alpha in _ALPHA_GRID:
        # Compare y[t] with the level available at t-1. Comparing with the
        # contemporaneous smoothed value leaks y[t] into its own prediction
        # and degenerately favors alpha values near one.
        levels = train.ewm(alpha=float(alpha), adjust=False).mean()
        one_step_forecast = levels.shift(1)
        errors = train.iloc[1:] - one_step_forecast.iloc[1:]
        sse = float(np.sum(errors**2))
        if sse < best_sse:
            best_sse = sse
            best_alpha = float(alpha)
    return best_alpha


def fit_ewma(
    series: pd.Series,
    forecast_horizon: int,
    alpha: float | None = None,
    mase_period: int = 1,
) -> ForecastAdapterResult:
    """Fit SES/EWMA and return a typed adapter result.

    When ``alpha`` is ``None`` the adapter estimates it from the training
    split. The multi-step forecast is flat at the final smoothed level,
    which is the correct simple-exponential-smoothing point forecast.

    Args:
        series: Time series data.
        forecast_horizon: Number of periods to forecast.
        alpha: Optional fixed smoothing parameter. When ``None``, alpha is
            estimated by minimizing one-step SSE on the training split.

    Returns:
        :class:`ForecastAdapterResult` with status, forecast, intervals,
        nullable metrics, and fitted configuration provenance.
    """
    series = series.dropna().astype(float)

    if len(series) < 3:
        logger.warning(
            "Series too short for EWMA (%d points). Returning persistence forecast.",
            len(series),
        )
        last_val = float(series.iloc[-1]) if not series.empty else 0.0
        return ForecastAdapterResult(
            status=ForecastFitStatus.NOT_ESTIMABLE,
            failure_reason="EWMA requires at least three observations.",
            is_fallback=True,
            forecast=[last_val] * forecast_horizon,
            lower_ci=[last_val] * forecast_horizon,
            upper_ci=[last_val] * forecast_horizon,
            fitted_configuration={
                "model": "EWMA",
                "alpha": None,
                "initialization": None,
                "fallback": "persistence",
            },
        )

    # Split data into train and test sets for metrics calculation.
    holdout = make_terminal_holdout(series, forecast_horizon)
    train, test = holdout.train, holdout.test

    estimated_alpha = alpha if alpha is not None else _estimate_alpha(train)

    # ── Evaluate holdout metrics on the training split ──────────────────────
    try:
        train_fit = SimpleExpSmoothing(
            train, initialization_method="estimated"
        ).fit(smoothing_level=alpha, optimized=alpha is None)
        estimated_alpha = float(train_fit.params["smoothing_level"])
        test_fc = np.asarray(train_fit.forecast(len(test)), dtype=float)
        metrics = evaluate_predictions(
            holdout,
            test_fc,
            mase_period=mase_period,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("EWMA metrics calculation failed: %s", exc)
        metrics = ForecastMetrics(unavailable_reasons={"all": str(exc)})

    # ── Full-series fit for forecast ─────────────────────────────────────────
    full_fit = SimpleExpSmoothing(
        series, initialization_method="estimated"
    ).fit(smoothing_level=estimated_alpha, optimized=False)
    forecast_values = np.asarray(full_fit.forecast(forecast_horizon), dtype=float)
    residuals = pd.Series(np.asarray(full_fit.resid, dtype=float)).dropna()
    rng = np.random.default_rng(42)
    sampled = rng.choice(
        residuals.to_numpy(dtype=float),
        size=(1000, forecast_horizon),
        replace=True,
    )
    simulated = forecast_values[None, :] + sampled
    lower_ci = np.quantile(simulated, 0.025, axis=0).tolist()
    upper_ci = np.quantile(simulated, 0.975, axis=0).tolist()

    # Expose fitted innovations (one-step smoothing errors).
    innovations: list[float] = []
    try:
        resid_arr = np.asarray(residuals.dropna(), dtype=float)
        innovations = resid_arr[np.isfinite(resid_arr)].tolist()
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("EWMA innovations unavailable: %s", exc)

    logger.info("EWMA model fitted with alpha=%.4f", estimated_alpha)

    status = (
        ForecastFitStatus.OK if metrics.rmse is not None else ForecastFitStatus.DEGRADED
    )
    failure_reason = (
        None if metrics.rmse is not None else metrics.unavailable_reasons.get("all")
    )

    return ForecastAdapterResult(
        status=status,
        failure_reason=failure_reason,
        is_fallback=False,
        forecast=forecast_values.tolist(),
        lower_ci=lower_ci,
        upper_ci=upper_ci,
        metrics=metrics,
        fitted_configuration={
            "model": "EWMA",
            "alpha": estimated_alpha,
            "initialization": "level",
            "estimated": alpha is None,
        },
        innovations=innovations,
        interval_label="bootstrap_prediction_interval",
    )
