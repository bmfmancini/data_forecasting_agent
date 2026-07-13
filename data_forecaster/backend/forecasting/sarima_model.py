"""Seasonal ARIMA forecasting adapter backed by pmdarima auto_arima."""

from __future__ import annotations

import pandas as pd

from core.logging_config import get_logger
from forecasting.metrics import calculate_holdout_metrics
from forecasting.contracts import ForecastFitStatus, ForecastMetrics
from forecasting.pmdarima_compat import import_pmdarima

logger = get_logger(__name__)
pm = import_pmdarima()


def _calculate_metrics(
    train: pd.Series, test: pd.Series, model, seasonal_period: int
) -> ForecastMetrics:
    """Calculate RMSE, MAE, and MAPE for the given model and test data.

    Args:
        test: Test data.
        model: Trained SARIMA model.

    Returns:
        tuple[float, float, float]: RMSE, MAE, and MAPE metrics.
    """
    try:
        return calculate_holdout_metrics(
            test,
            model,
            training=train,
            mase_period=seasonal_period if seasonal_period > 1 else 1,
        )
    except Exception as exc:
        logger.warning("SARIMA metrics calculation failed: %s", exc)
        return ForecastMetrics(unavailable_reasons={"all": str(exc)})


def fit_sarima(
    series: pd.Series,
    forecast_horizon: int,
    seasonal_period: int = 12,
) -> dict:
    """Fit SARIMA via pmdarima auto_arima (seasonal=True) and return forecast + metrics.

    Args:
        series: A pandas Series containing the time series data.
        forecast_horizon: The number of periods to forecast.
        seasonal_period: The seasonal period of the time series.

    Returns:
        dict with keys: forecast, lower_ci, upper_ci, rmse, mae, mape
    """
    series = series.dropna().astype(float)

    # Check if we have enough data for seasonal modeling
    if len(series) < 2 * seasonal_period:
        logger.warning(
            "Series too short (%d obs) for seasonal period %d. Fitting non-seasonal ARIMA.",
            len(series),
            seasonal_period,
        )
        seasonal_period = 1

    use_seasonal = seasonal_period > 1

    # Split data into train and test sets for metrics calculation
    split = max(int(len(series) * 0.8), len(series) - forecast_horizon)
    train, test = series.iloc[:split], series.iloc[split:]

    train_model = None
    metrics = ForecastMetrics(
        unavailable_reasons={"all": "Training model unavailable."}
    )

    try:
        train_model = pm.auto_arima(
            train,
            seasonal=use_seasonal,
            m=seasonal_period,
            stepwise=True,
            max_p=3,
            max_q=3,
            max_P=2,
            max_Q=2,
            max_order=10,
            error_action="ignore",
            suppress_warnings=True,
            information_criterion="aic",
        )
        metrics = _calculate_metrics(train, test, train_model, seasonal_period)
    except Exception as exc:
        logger.warning("SARIMA training failed: %s", exc)

    # Fit the model on the full series using parameters from training.
    # Fall back to default orders when auto_arima failed (train_model is None),
    # matching the pattern used in arima_model.py.
    order = train_model.order if train_model is not None else (1, 1, 1)
    seasonal_order = (
        train_model.seasonal_order
        if train_model is not None
        else (0, 0, 0, seasonal_period)
    )
    full_model = pm.ARIMA(
        order=order,
        seasonal_order=seasonal_order,
        suppress_warnings=True,
    ).fit(series)

    logger.info(
        "SARIMA selected order: %s seasonal_order: %s",
        full_model.order,
        full_model.seasonal_order,
    )

    forecast_values, conf_int = full_model.predict(
        n_periods=forecast_horizon, return_conf_int=True
    )

    return {
        "status": (
            ForecastFitStatus.OK.value
            if metrics.rmse is not None
            else ForecastFitStatus.DEGRADED.value
        ),
        "failure_reason": (
            None if metrics.rmse is not None else metrics.unavailable_reasons.get("all")
        ),
        "is_fallback": train_model is None or not use_seasonal,
        "forecast": forecast_values.tolist(),
        "lower_ci": conf_int[:, 0].tolist(),
        "upper_ci": conf_int[:, 1].tolist(),
        "rmse": metrics.rmse,
        "mae": metrics.mae,
        "mape": metrics.mape,
        "wape": metrics.wape,
        "mase": metrics.mase,
    }
