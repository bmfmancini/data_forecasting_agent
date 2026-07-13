"""Shared metric calculations for forecast model adapters."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from forecasting.contracts import ForecastMetrics


class PredictsIntervals(Protocol):
    """Protocol for fitted models that can forecast with confidence intervals."""

    def predict(
        self,
        n_periods: int,
        return_conf_int: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return forecast values and confidence intervals."""


def calculate_holdout_metrics(
    test: pd.Series,
    model: PredictsIntervals | None,
    *,
    training: pd.Series | None = None,
    mase_period: int = 1,
) -> ForecastMetrics:
    """Calculate RMSE, MAE, and MAPE for a fitted forecast model.

    Args:
        test: Holdout observations to compare against model predictions.
        model: Fitted model exposing a pmdarima-style ``predict`` method.

    Returns:
        Typed metrics. Empty holdout data or a missing model returns unavailable
        metrics with a reason; unavailable evidence is never encoded as zero.
    """
    if len(test) == 0 or model is None:
        return ForecastMetrics(
            unavailable_reasons={"all": "Holdout data or fitted model unavailable."}
        )

    test_fc, _ = model.predict(n_periods=len(test), return_conf_int=True)
    residuals = test.values - test_fc
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    return calculate_forecast_metrics(
        test.values,
        test_fc,
        training=training,
        mase_period=mase_period,
    )


def calculate_forecast_metrics(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    *,
    training: np.ndarray | pd.Series | None = None,
    mase_period: int = 1,
) -> ForecastMetrics:
    """Calculate point metrics under one documented set of conventions.

    MAPE is unavailable when actuals contain zeros. MASE uses a fixed naive
    lag supplied by the caller and is unavailable when its scale cannot be
    estimated. WAPE uses the sum of absolute actuals as its denominator.
    """
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        return ForecastMetrics(
            unavailable_reasons={
                "all": "Actual and predicted values must be non-empty and aligned."
            }
        )
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    n_missing = int(y_true.size - np.count_nonzero(finite))
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    if y_true.size == 0:
        return ForecastMetrics(
            unavailable_reasons={"all": "No finite aligned observations."}
        )

    errors = y_true - y_pred
    absolute_errors = np.abs(errors)
    reasons: dict[str, str] = {}
    mape = None
    if np.any(y_true == 0):
        reasons["mape"] = "MAPE is undefined when any actual value is zero."
    else:
        mape = float(np.mean(np.abs(errors / y_true)) * 100)

    denominator = float(np.sum(np.abs(y_true)))
    wape = None
    if denominator == 0:
        reasons["wape"] = (
            "WAPE is undefined when the absolute-actual denominator is zero."
        )
    else:
        wape = float(np.sum(absolute_errors) / denominator)

    mase = None
    if training is None:
        reasons["mase"] = "Training data is required for MASE."
    else:
        train = np.asarray(training, dtype=float)
        train = train[np.isfinite(train)]
        if mase_period < 1 or train.size <= mase_period:
            reasons["mase"] = "Training data is too short for the configured naive lag."
        else:
            scale = float(np.mean(np.abs(train[mase_period:] - train[:-mase_period])))
            if scale == 0:
                reasons["mase"] = (
                    "MASE is undefined because the naive error scale is zero."
                )
            else:
                mase = float(np.mean(absolute_errors) / scale)

    return ForecastMetrics(
        rmse=float(np.sqrt(np.mean(errors**2))),
        mae=float(np.mean(absolute_errors)),
        mape=mape,
        wape=wape,
        mase=mase,
        n_evaluated=int(y_true.size),
        n_missing=n_missing,
        unavailable_reasons=reasons,
    )
