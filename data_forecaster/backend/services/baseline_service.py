"""Service to compute metrics for simple baseline forecasting models.

Provides a function to generate forecasts and calculate error metrics for
several heuristic models:
- Naive
- Seasonal Naive
- Mean Forecast
- Drift

These metrics are used in the final report's model comparison table to
demonstrate that the selected sophisticated model provides a tangible
improvement over simple approaches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from core.logging_config import get_logger

logger = get_logger(__name__)


def _calculate_metrics(
    y_true: pd.Series, y_pred: pd.Series
) -> dict[str, float]:
    """Calculate standard forecast error metrics.

    Args:
        y_true: Ground truth (actual) values.
        y_pred: Predicted values.

    Returns:
        A dict with RMSE, MAE, and MAPE.
    """
    y_true = y_true.values
    y_pred = y_pred.values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Avoid division by zero for MAPE
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def run_baseline_models(
    series: pd.Series,
    forecast_horizon: int,
    seasonal_period: int,
) -> dict[str, dict[str, float]]:
    """Compute metrics for all baseline models.

    Args:
        series: The full time series data.
        forecast_horizon: The number of steps to forecast.
        seasonal_period: The seasonal period for the Seasonal Naive model.

    Returns:
        A dictionary mapping baseline model names to their error metrics.
    """
    # Use an 80/20 split, ensuring test set is at least the horizon length
    split_point = max(
        int(len(series) * 0.8), len(series) - forecast_horizon
    )
    train, test = series[:split_point], series[split_point:]

    # Ensure test set matches horizon if it's longer
    if len(test) > forecast_horizon:
        test = test[:forecast_horizon]

    # Adjust horizon if test set is shorter
    h = len(test)
    if h == 0:
        logger.warning("Cannot run baseline models on empty test set.")
        return {}

    metrics = {}

    # 1. Naive Forecast
    last_val = train.iloc[-1]
    naive_pred = pd.Series(np.repeat(last_val, h), index=test.index)
    metrics["Naive"] = _calculate_metrics(test, naive_pred)

    # 2. Seasonal Naive Forecast
    if len(train) >= seasonal_period:
        final_season = train.iloc[-seasonal_period:]
        snaive_forecast = [
            final_season.iloc[i % seasonal_period] for i in range(h)
        ]
        snaive_pred = pd.Series(snaive_forecast, index=test.index)
        metrics["Seasonal Naive"] = _calculate_metrics(test, snaive_pred)

    # 3. Mean Forecast
    mean_val = train.mean()
    mean_pred = pd.Series(np.repeat(mean_val, h), index=test.index)
    metrics["Mean Forecast"] = _calculate_metrics(test, mean_pred)

    # 4. Drift Forecast
    if len(train) > 1:
        drift = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
        drift_pred_values = [
            train.iloc[-1] + i * drift for i in range(1, h + 1)
        ]
        drift_pred = pd.Series(drift_pred_values, index=test.index)
        metrics["Drift"] = _calculate_metrics(test, drift_pred)

    # Log the results for traceability
    for model, m in metrics.items():
        logger.info(
            "Baseline model %s -> MAE: %.4f, RMSE: %.4f, MAPE: %.2f%%",
            model,
            m["MAE"],
            m["RMSE"],
            m["MAPE"],
        )

    return metrics
