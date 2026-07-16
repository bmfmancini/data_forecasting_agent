"""Terminal-holdout validation helper for forecast model evaluation.

This module performs a single terminal holdout split — not rolling-origin
validation. The rolling-origin backtesting service will replace this with a
proper expanding-window approach that generates identical folds for every
candidate model.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from forecasting.metrics import calculate_forecast_metrics

ForecastFunction = Callable[[pd.Series, int], pd.Series]


def terminal_holdout_validation(
    series: pd.Series,
    forecast_horizon: int,
    forecast_fn: ForecastFunction,
) -> dict[str, float]:
    """Evaluate a forecast function against a single terminal holdout split.

    This is a simple train/test evaluation — not rolling-origin validation.
    It creates one split, fits on the training portion, and scores the
    forecast against the holdout. The rolling-origin backtesting service
    will replace this with a proper expanding-window approach that generates
    identical folds for every candidate model.

    Args:
        series: Historical observations ordered by time.
        forecast_horizon: Number of periods the model forecasts.
        forecast_fn: Function that accepts training data and horizon and
            returns forecast values for the holdout period.

    Returns:
        Mapping with metric keys from :class:`ForecastMetrics`. Returns an
        empty mapping when the series is too short for a holdout split.
    """
    clean_series = series.dropna().astype(float)
    if forecast_horizon < 1 or len(clean_series) <= forecast_horizon:
        return {}

    split = max(1, len(clean_series) - forecast_horizon)
    train = clean_series.iloc[:split]
    test = clean_series.iloc[split:]
    if train.empty or test.empty:
        return {}

    forecast = forecast_fn(train, len(test)).astype(float)
    forecast_values = forecast.to_numpy()[: len(test)]
    test_values = test.to_numpy()
    metrics = calculate_forecast_metrics(
        test_values,
        forecast_values,
        training=train.values,
        mase_period=1,
    )
    return metrics.model_dump()
