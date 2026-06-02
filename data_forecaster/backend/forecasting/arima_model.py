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

    Returns:
        dict with keys: forecast, lower_ci, upper_ci, rmse, mae, mape
    """
    series = series.dropna().astype(float)

    # ── Metrics via train/test split ─────────────────────────────────────────
    split = max(int(len(series) * 0.8), len(series) - forecast_horizon)
    train, test = series.iloc[:split], series.iloc[split:]

    try:
        train_model = pm.auto_arima(
            train,
            seasonal=False,
            stepwise=True,
            error_action="ignore",
            suppress_warnings=True,
            information_criterion="aic",
        )
        test_fc, _ = train_model.predict(n_periods=len(test), return_conf_int=True)
        rmse = float(np.sqrt(np.mean((test.values - test_fc) ** 2)))
        mae = float(np.mean(np.abs(test.values - test_fc)))
        mape = float(np.mean(np.abs((test.values - test_fc) / (test.values + 1e-8))) * 100)
    except Exception as exc:
        logger.warning("ARIMA metrics failed: %s", exc)
        rmse = mae = mape = float("nan")

    # ── Full-series fit for forecast ─────────────────────────────────────────
    full_model = pm.auto_arima(
        series,
        seasonal=False,
        stepwise=True,
        error_action="ignore",
        suppress_warnings=True,
        information_criterion="aic",
    )
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
