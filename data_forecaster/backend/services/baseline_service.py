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

Baselines share the common terminal-holdout fold from
:mod:`forecasting.backtesting` so that all candidates use the same
evaluation boundary. Baselines label their intervals as experimental
because they do not produce model-based prediction intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.logging_config import get_logger
from forecasting.contracts import ForecastAdapterResult, ForecastFitStatus
from forecasting.evaluation import (
    TerminalHoldout,
    evaluate_predictions,
    make_terminal_holdout,
)

logger = get_logger(__name__)


def _evaluate_baseline(
    name: str,
    y_pred: pd.Series,
    split: TerminalHoldout,
    mase_period: int,
) -> ForecastAdapterResult:
    """Calculate standard forecast error metrics.

    Args:
        y_true: Ground truth (actual) values.
        y_pred: Predicted values.

    Returns:
        A dict with RMSE, MAE, and MAPE.
    """
    result = evaluate_predictions(
        split,
        y_pred,
        mase_period=mase_period,
    )
    status = (
        ForecastFitStatus.OK
        if result.rmse is not None and result.mae is not None
        else ForecastFitStatus.NOT_ESTIMABLE
    )
    return ForecastAdapterResult(
        status=status,
        forecast=y_pred.astype(float).tolist(),
        metrics=result,
        failure_reason=(
            None if status == ForecastFitStatus.OK else "Baseline metrics unavailable."
        ),
        fitted_configuration={"model": name, "mase_period": mase_period},
        # Baselines do not produce model-based prediction intervals.
        interval_label="experimental",
    )


def run_baseline_models(
    series: pd.Series,
    forecast_horizon: int,
    seasonal_period: int,
) -> dict[str, ForecastAdapterResult]:
    """Compute metrics for all baseline models.

    Args:
        series: The full time series data.
        forecast_horizon: The number of steps to forecast.
        seasonal_period: The seasonal period for the Seasonal Naive model.

    Returns:
        A dictionary mapping baseline model names to their error metrics.
    """
    # Use an 80/20 split, ensuring test set is at least the horizon length
    holdout = make_terminal_holdout(series, forecast_horizon)
    train, test = holdout.train, holdout.test

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
    metrics["Naive"] = _evaluate_baseline("Naive", naive_pred, holdout, seasonal_period)

    # 2. Seasonal Naive Forecast
    if len(train) >= seasonal_period:
        final_season = train.iloc[-seasonal_period:]
        snaive_forecast = [final_season.iloc[i % seasonal_period] for i in range(h)]
        snaive_pred = pd.Series(snaive_forecast, index=test.index)
        metrics["Seasonal Naive"] = _evaluate_baseline(
            "Seasonal Naive", snaive_pred, holdout, seasonal_period
        )

    # 3. Mean Forecast
    mean_val = train.mean()
    mean_pred = pd.Series(np.repeat(mean_val, h), index=test.index)
    metrics["Mean Forecast"] = _evaluate_baseline(
        "Mean Forecast", mean_pred, holdout, seasonal_period
    )

    # 4. Drift Forecast
    if len(train) > 1:
        drift = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
        drift_pred_values = [train.iloc[-1] + i * drift for i in range(1, h + 1)]
        drift_pred = pd.Series(drift_pred_values, index=test.index)
        metrics["Drift"] = _evaluate_baseline(
            "Drift", drift_pred, holdout, seasonal_period
        )

    # Log the results for traceability
    for model, result in metrics.items():
        logger.info("Baseline model %s -> metrics=%s", model, result.metrics)

    return metrics
