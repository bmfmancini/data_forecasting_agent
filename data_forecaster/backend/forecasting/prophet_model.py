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

import numpy as np
import pandas as pd

from core.logging_config import get_logger
from forecasting.prophet_compat import import_prophet

logger = get_logger(__name__)


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


def _metrics_from_holdout(
    train: pd.Series,
    test: pd.Series,
    prophet_module,
    freq: str | None,
) -> tuple[float, float, float]:
    """Fit Prophet on ``train`` and score RMSE/MAE/MAPE against ``test``.

    Args:
        train: Training observations.
        test: Holdout observations.
        prophet_module: The imported ``prophet`` module.
        freq: Optional pandas frequency string for the future frame.

    Returns:
        ``(rmse, mae, mape)`` — zeroed on any fit/predict failure so the
        caller can still produce a full-series forecast.
    """
    try:
        train_history = _to_history_frame(train)
        m = prophet_module.Prophet()
        m.fit(train_history)
        future = _future_frame(train_history, periods=len(test), freq=freq)
        fc = m.predict(future)
        pred = fc["yhat"].to_numpy()
        actual = test.to_numpy()
        residuals = actual - pred
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))
        mape = float(np.mean(np.abs(residuals / (actual + 1e-8))) * 100)
        return rmse, mae, mape
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Prophet holdout metrics failed: %s", exc)
        return 0.0, 0.0, 0.0


def fit_prophet(
    series: pd.Series,
    forecast_horizon: int,
    freq: str | None = None,
) -> dict:
    """Fit Meta Prophet and return forecast + metrics.

    Args:
        series: A pandas Series containing the time series data.
        forecast_horizon: The number of periods to forecast.
        freq: Optional pandas frequency string for building the future frame.
            Inferred from the series index when not supplied.

    Returns:
        dict with keys: forecast, lower_ci, upper_ci, rmse, mae, mape
    """
    series = series.dropna().astype(float)

    if len(series) < 2:
        logger.warning(
            "Series too short for Prophet (%d points). Returning persistence "
            "forecast.",
            len(series),
        )
        last_val = series.iloc[-1] if not series.empty else 0.0
        return {
            "forecast": [last_val] * forecast_horizon,
            "lower_ci": [last_val] * forecast_horizon,
            "upper_ci": [last_val] * forecast_horizon,
            "rmse": 0.0,
            "mae": 0.0,
            "mape": 0.0,
        }

    prophet_module = import_prophet()

    # Split data into train and test sets for metrics calculation
    split = max(
        int(len(series) * 0.8),
        len(series) - forecast_horizon,
    )
    split = min(split, len(series) - 1)
    train, test = series.iloc[:split], series.iloc[split:]

    rmse, mae, mape = 0.0, 0.0, 0.0
    if len(train) >= 2 and len(test) >= 1:
        rmse, mae, mape = _metrics_from_holdout(train, test, prophet_module, freq)

    # Fit the model on the full series for the final forecast
    history = _to_history_frame(series)
    m = prophet_module.Prophet()
    m.fit(history)
    future = _future_frame(history, periods=forecast_horizon, freq=freq)
    fc = m.predict(future)

    forecast_values = fc["yhat"].to_numpy()
    lower_ci = fc["yhat_lower"].to_numpy()
    upper_ci = fc["yhat_upper"].to_numpy()

    logger.info(
        "Prophet fitted: series_len=%d horizon=%d freq=%s",
        len(series),
        forecast_horizon,
        freq,
    )

    return {
        "forecast": forecast_values.tolist(),
        "lower_ci": lower_ci.tolist(),
        "upper_ci": upper_ci.tolist(),
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }
