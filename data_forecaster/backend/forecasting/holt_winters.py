"""Holt-Winters exponential smoothing forecasting implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from core.logging_config import get_logger
from forecasting.contracts import (
    ForecastAdapterResult,
    ForecastFitStatus,
    ForecastMetrics,
)
from forecasting.evaluation import evaluate_predictions, make_terminal_holdout

logger = get_logger(__name__)


def fit_holt_winters(
    series: pd.Series, forecast_horizon: int, mase_period: int = 1
) -> ForecastAdapterResult:
    """Fit Holt-Winters Triple Exponential Smoothing and return a typed result.

    The adapter selects additive versus multiplicative seasonality on the
    training split (not the full series) to avoid leaking test observations
    into model-form selection. It then refits the chosen configuration on
    the full series for the production forecast.

    Args:
        series: A pandas Series containing the time series data.
        forecast_horizon: The number of periods to forecast.

    Returns:
        :class:`ForecastAdapterResult` with status, forecast, intervals,
        nullable metrics, and fitted configuration provenance.
    """
    series = series.dropna().astype(float)
    seasonal_period = _infer_seasonal_period(series)

    trend = "add"
    seasonal: str | None = None

    # Split data into train and test sets for metrics calculation and
    # model-form selection (additive vs multiplicative seasonal).
    holdout = make_terminal_holdout(series, forecast_horizon)
    train, test = holdout.train, holdout.test
    # Seasonal model-form selection is valid only when the training sample,
    # not merely the full series, contains enough cycles.
    use_seasonal = len(train) >= 2 * seasonal_period

    # ── Select seasonal type on the *training* split only ────────────────────
    if use_seasonal:
        if (train > 0).all():
            try:
                m_fit = ExponentialSmoothing(
                    train,
                    trend="add",
                    seasonal="mul",
                    seasonal_periods=seasonal_period,
                ).fit(optimized=True)
                a_fit = ExponentialSmoothing(
                    train,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=seasonal_period,
                ).fit(optimized=True)
                seasonal = "mul" if m_fit.aic < a_fit.aic else "add"
            except Exception:  # pylint: disable=broad-except
                seasonal = "add"
        else:
            seasonal = "add"

    logger.info(
        "Holt-Winters config: seasonal=%s seasonal_period=%d series_len=%d",
        seasonal,
        seasonal_period,
        len(series),
    )

    # ── Evaluate holdout metrics on the training split ──────────────────────
    try:
        train_fit = ExponentialSmoothing(
            train,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_period if use_seasonal else None,
        ).fit(optimized=True)
        test_fc = train_fit.forecast(len(test))
        metrics = evaluate_predictions(
            holdout,
            test_fc.values,
            mase_period=mase_period,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Holt-Winters metrics failed: %s", exc)
        metrics = ForecastMetrics(unavailable_reasons={"all": str(exc)})

    # ── Fit the model on the full series for final forecasting ───────────────
    full_fit = ExponentialSmoothing(
        series,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_period if use_seasonal else None,
    ).fit(optimized=True)

    forecast_values = full_fit.forecast(forecast_horizon)
    resid_std = float(np.std(full_fit.resid))
    h = np.arange(1, forecast_horizon + 1)
    lower_ci = (forecast_values.values - 1.96 * resid_std * np.sqrt(h)).tolist()
    upper_ci = (forecast_values.values + 1.96 * resid_std * np.sqrt(h)).tolist()

    # Expose fitted innovations (level residuals) for diagnostics.
    innovations: list[float] = []
    try:
        resid = np.asarray(full_fit.resid, dtype=float)
        innovations = resid[np.isfinite(resid)].tolist()
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Holt-Winters innovations unavailable: %s", exc)

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
            "model": "Holt-Winters",
            "trend": trend,
            "damped_trend": False,
            "seasonal": seasonal,
            "seasonal_period": seasonal_period if use_seasonal else None,
            "initialization_method": getattr(
                full_fit, "initialization_method", "estimated"
            ),
        },
        innovations=innovations,
        # Holt-Winters intervals are residual-std heuristic bands, not
        # calibrated prediction intervals. Label them as experimental until
        # simulation/bootstrap intervals are implemented.
        interval_label="experimental",
    )


def _infer_seasonal_period(series: pd.Series) -> int:
    """Infer the seasonal period based on the series frequency.

    Args:
        series: A pandas Series with DatetimeIndex.

    Returns:
        int: The inferred seasonal period.
    """
    if hasattr(series.index, "freq") and series.index.freq is not None:
        freq_str = str(series.index.freq).upper()
        if "MS" in freq_str or freq_str.startswith("M"):
            return 12
        if "QS" in freq_str or freq_str.startswith("Q"):
            return 4
        if "W" in freq_str:
            return 52
        if freq_str.startswith("D"):
            return 7
    return 12
