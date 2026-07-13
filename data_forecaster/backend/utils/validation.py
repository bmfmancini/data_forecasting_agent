"""Rolling-origin validation helpers for forecast model evaluation."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from forecasting.metrics import calculate_forecast_metrics

ForecastFunction = Callable[[pd.Series, int], pd.Series]


def perform_rolling_origin_validation(
    series: pd.Series,
    forecast_horizon: int,
    forecast_fn: ForecastFunction,
) -> dict[str, float]:
    """Evaluate a forecast function against a simple holdout split.

    Args:
        series: Historical observations ordered by time.
        forecast_horizon: Number of periods the model forecasts.
        forecast_fn: Function that accepts training data and horizon and returns
            forecast values for the holdout period.

    Returns:
        Mapping with ``rmse``, ``mae``, and ``mape``. Returns an empty mapping
        when the series is too short for a holdout validation split.
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
