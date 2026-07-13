"""ARIMA forecasting adapter backed by pmdarima auto_arima."""

from __future__ import annotations

import pandas as pd

from core.logging_config import get_logger
from forecasting.metrics import calculate_holdout_metrics
from forecasting.contracts import ForecastFitStatus, ForecastMetrics
from forecasting.pmdarima_compat import import_pmdarima

logger = get_logger(__name__)
pm = import_pmdarima()


def _calculate_metrics(train: pd.Series, test: pd.Series, model) -> ForecastMetrics:
    """Calculate RMSE, MAE, and MAPE for the given model and test data.

    Args:
        test: Test data.
        model: Trained ARIMA model.

    Returns:
        tuple[float, float, float]: RMSE, MAE, and MAPE metrics.
    """
    try:
        return calculate_holdout_metrics(test, model, training=train, mase_period=1)
    except Exception as exc:
        logger.warning("ARIMA metrics calculation failed: %s", exc)
        return ForecastMetrics(unavailable_reasons={"all": str(exc)})


def fit_arima(series: pd.Series, forecast_horizon: int) -> dict:
    """Fit ARIMA via pmdarima auto_arima and return forecast + metrics.

    Args:
        series: A pandas Series containing the time series data.
        forecast_horizon: The number of periods to forecast.

    Returns:
        dict with keys: forecast, lower_ci, upper_ci, rmse, mae, mape
    """
    series = series.dropna().astype(float)

    if len(series) < 3:
        logger.warning(
            "Series too short for ARIMA (%d points). Returning persistence forecast.",
            len(series),
        )
        last_val = series.iloc[-1] if not series.empty else 0.0
        return {
            "status": ForecastFitStatus.NOT_ESTIMABLE.value,
            "failure_reason": "ARIMA requires at least three observations.",
            "is_fallback": True,
            "forecast": [last_val] * forecast_horizon,
            "lower_ci": [last_val] * forecast_horizon,
            "upper_ci": [last_val] * forecast_horizon,
            "rmse": None,
            "mae": None,
            "mape": None,
        }

    # Split data into train and test sets for metrics calculation
    split = max(
        1,
        min(
            len(series) - 1, max(int(len(series) * 0.8), len(series) - forecast_horizon)
        ),
    )
    train, test = series.iloc[:split], series.iloc[split:]

    train_model = None
    metrics = ForecastMetrics(
        unavailable_reasons={"all": "Training model unavailable."}
    )

    if len(train) >= 2:
        try:
            train_model = pm.auto_arima(
                train,
                seasonal=False,
                stepwise=True,
                max_p=5,
                max_q=5,
                error_action="ignore",
                suppress_warnings=True,
                information_criterion="aic",
            )
            metrics = _calculate_metrics(train, test, train_model)
        except Exception as exc:
            logger.warning("ARIMA training failed: %s", exc)

    # Fit the model on the full series using the order from training
    order = train_model.order if train_model is not None else (1, 1, 1)
    full_model = pm.ARIMA(order=order, suppress_warnings=True).fit(series)

    logger.info("ARIMA selected order: %s", full_model.order)

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
        "is_fallback": train_model is None,
        "forecast": forecast_values.tolist(),
        "lower_ci": conf_int[:, 0].tolist(),
        "upper_ci": conf_int[:, 1].tolist(),
        "rmse": metrics.rmse,
        "mae": metrics.mae,
        "mape": metrics.mape,
        "wape": metrics.wape,
        "mase": metrics.mase,
    }
