from __future__ import annotations

import numpy as np
import pandas as pd

# Compatibility shim: pmdarima 2.0.x uses sklearn's force_all_finite which was
# removed in scikit-learn 1.6. Translate it to ensure_all_finite.
import sklearn.utils.validation as _skval  # noqa: E402

if not hasattr(_skval, "_patched_for_pmdarima"):
    _orig = _skval.check_array

    def _patched(*args, **kwargs):  # noqa: E306
        if "force_all_finite" in kwargs:
            kwargs.setdefault("ensure_all_finite", kwargs.pop("force_all_finite"))
        return _orig(*args, **kwargs)

    _skval.check_array = _patched
    _skval._patched_for_pmdarima = True

import pmdarima as pm

from core.logging_config import get_logger

logger = get_logger(__name__)


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
            "forecast": [last_val] * forecast_horizon,
            "lower_ci": [last_val] * forecast_horizon,
            "upper_ci": [last_val] * forecast_horizon,
            "rmse": 0.0,
            "mae": 0.0,
            "mape": 0.0,
        }

    # Split data into train and test sets for metrics calculation
    split = max(
        1,
        min(
            len(series) - 1, max(int(len(series) * 0.8), len(series) - forecast_horizon)
        ),
    )
    train, test = series.iloc[:split], series.iloc[split:]

    rmse = mae = mape = 0.0
    train_model = None

    try:
        if len(train) >= 2:
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
            if len(test) > 0:
                test_fc, _ = train_model.predict(
                    n_periods=len(test), return_conf_int=True
                )
                rmse = float(np.sqrt(np.mean((test.values - test_fc) ** 2)))
                mae = float(np.mean(np.abs(test.values - test_fc)))
                mape = float(
                    np.mean(np.abs((test.values - test_fc) / (test.values + 1e-8)))
                    * 100
                )
    except Exception as exc:
        logger.warning("ARIMA metrics failed: %s", exc)

    # Fit the model on the full series using the order from training
    order = train_model.order if train_model is not None else (1, 1, 1)

    full_model = pm.ARIMA(order=order, suppress_warnings=True).fit(series)

    logger.info("ARIMA selected order: %s", full_model.order)

    forecast_values, conf_int = full_model.predict(
        n_periods=forecast_horizon, return_conf_int=True
    )

    return {
        "forecast": forecast_values.tolist(),
        "lower_ci": conf_int[:, 0].tolist(),
        "upper_ci": conf_int[:, 1].tolist(),
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }
