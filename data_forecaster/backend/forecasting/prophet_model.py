"""Prophet (Meta Prophet) forecasting adapter.

Prophet is an additive regression model that decomposes a series into trend,
seasonality (daily/weekly/yearly), and holidays.  It is robust to missing
values, outliers, and irregular sampling, and produces uncertainty intervals
natively.

Prophet is imported lazily via :func:`forecasting.prophet_compat.import_prophet`
so this module remains importable when the optional ``prophet`` dependency is
not installed.  The forecasting agent wraps ``fit_prophet`` in try/except and
simply skips the model when Prophet is unavailable.
"""

from __future__ import annotations

from threading import Lock

import numpy as np
import pandas as pd

from core.logging_config import get_logger
from forecasting.contracts import (
    ForecastAdapterResult,
    ForecastFitStatus,
    ForecastMetrics,
)

logger = get_logger(__name__)
_PREDICTION_LOCK = Lock()


def _to_history_frame(series: pd.Series) -> pd.DataFrame:
    """Convert a Series to the Prophet ``ds``/``y`` history frame.

    Prophet requires a ``ds`` column of datetimes.  When the input index is
    not datetime-valued, a synthetic regular date range is generated so the
    model can still fit.

    Args:
        series: A pandas Series (any index).

    Returns:
        A two-column DataFrame with ``ds`` (datetime) and ``y`` (float).
    """
    if isinstance(series.index, pd.DatetimeIndex):
        ds = series.index
    else:
        ds = pd.date_range(start="2000-01-01", periods=len(series), freq="D")
    return pd.DataFrame({"ds": ds, "y": series.values})


def _future_frame(
    history: pd.DataFrame, periods: int, freq: str | None
) -> pd.DataFrame:
    """Build a Prophet future frame of ``periods`` timestamps after history.

    Args:
        history: The history frame (used to anchor the continuation).
        periods: Number of future periods to generate.
        freq: Optional pandas frequency string.  Inferred from the history
            dates when not supplied.

    Returns:
        A one-column DataFrame (``ds``) covering the future periods.
    """
    if freq is None:
        freq = pd.infer_freq(history["ds"]) or "D"
    last_ds = history["ds"].iloc[-1]
    future_dates = pd.date_range(start=last_ds, periods=periods + 1, freq=freq)[1:]
    return pd.DataFrame({"ds": future_dates})


def _align_regressor(values: dict, ds: pd.Series) -> np.ndarray:
    """Align a ``{date: value}`` regressor dict onto the ``ds`` timestamps.

    Prophet requires a finite regressor value at every timestamp in both the
    history and future frames.  Event-type indicator covariates produced by
    :mod:`forecasting.known_context` are already 0.0-filled; user-supplied
    covariates may have gaps, which are forward- then back-filled (and finally
    zero-filled) so Prophet never sees a missing regressor value.
    """
    ser = pd.Series(values, dtype=float)
    ser.index = pd.to_datetime(ser.index)
    aligned = ser.reindex(pd.DatetimeIndex(ds))
    aligned = aligned.ffill().bfill().fillna(0.0)
    return aligned.to_numpy()


def prophet_predictive_samples(model, future: pd.DataFrame) -> np.ndarray:
    """Reproducible Prophet marginal draws without leaking random state."""
    with _PREDICTION_LOCK:
        state = np.random.get_state()
        try:
            np.random.seed(42)
            model.uncertainty_samples = 2000
            return np.asarray(model.predictive_samples(future)["yhat"], dtype=float).T
        finally:
            np.random.set_state(state)
            model.uncertainty_samples = 0


def fit_prophet(
    series: pd.Series,
    forecast_horizon: int,
    freq: str | None = None,
) -> ForecastAdapterResult:
    """Compatibility adapter with nullable evaluation and the shared fitter."""
    from forecasting.window_models import fit_prophet_window
    from forecasting.evaluation import make_terminal_holdout, evaluate_predictions

    if series.notna().sum() < 3:
        return ForecastAdapterResult(
            status=ForecastFitStatus.NOT_ESTIMABLE,
            failure_reason="Prophet requires at least three observations.",
            fitted_configuration={"model": "Prophet"},
        )
    holdout = make_terminal_holdout(series, forecast_horizon)
    metrics = ForecastMetrics()
    try:
        fit = fit_prophet_window(holdout.train, len(holdout.test), freq=freq)
        metrics = evaluate_predictions(holdout, np.asarray(fit.forecast), mase_period=1)
    except Exception as exc:
        metrics = ForecastMetrics(unavailable_reasons={"all": str(exc)})
    try:
        result = fit_prophet_window(series, forecast_horizon, freq=freq)
    except Exception as exc:
        return ForecastAdapterResult(
            status=ForecastFitStatus.FAILED,
            failure_reason=str(exc),
            metrics=metrics,
            fitted_configuration={"model": "Prophet"},
        )
    result.metrics = metrics
    if metrics.rmse is None:
        result.status = ForecastFitStatus.DEGRADED
        result.failure_reason = metrics.unavailable_reasons.get("all")
    return result
