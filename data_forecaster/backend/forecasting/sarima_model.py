"""Seasonal ARIMA forecasting adapter backed by pmdarima auto_arima."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.logging_config import get_logger
from forecasting.contracts import (
    ForecastAdapterResult,
    ForecastFitStatus,
    ForecastMetrics,
)
from forecasting.evaluation import evaluate_predictions, make_terminal_holdout
from forecasting.pmdarima_compat import import_pmdarima

logger = get_logger(__name__)
pm = import_pmdarima()


def _calculate_metrics(holdout, model, mase_period: int) -> ForecastMetrics:
    """Calculate holdout metrics for the given SARIMA model.

    Args:
        train: Training data used for MASE scale.
        test: Holdout observations.
        model: Trained SARIMA model with a ``predict`` method.
        seasonal_period: Seasonal period used for the MASE naive lag.

    Returns:
        Typed metrics. Unavailable evidence is never encoded as zero.
    """
    try:
        predictions, _ = model.predict(
            n_periods=len(holdout.test), return_conf_int=True
        )
        return evaluate_predictions(
            holdout,
            predictions,
            mase_period=mase_period,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("SARIMA metrics calculation failed: %s", exc)
        return ForecastMetrics(unavailable_reasons={"all": str(exc)})


def fit_sarima(
    series: pd.Series,
    forecast_horizon: int,
    seasonal_period: int = 12,
    mase_period: int = 1,
) -> ForecastAdapterResult:
    """Fit SARIMA via pmdarima auto_arima and return a typed adapter result.

    When the series is too short for the requested seasonal period, the
    adapter falls back to a non-seasonal ARIMA and marks the result as a
    fallback. The adapter discovers orders on a training split, evaluates
    holdout metrics, then refits the *same* orders (including intercept/trend
    configuration) on the full series for the production forecast.

    Args:
        series: A pandas Series containing the time series data.
        forecast_horizon: The number of periods to forecast.
        seasonal_period: The seasonal period of the time series.

    Returns:
        :class:`ForecastAdapterResult` with status, forecast, intervals,
        nullable metrics, and fitted configuration provenance.
    """
    series = series.dropna().astype(float)

    # Check if we have enough data for seasonal modeling
    if len(series) < 2 * seasonal_period:
        logger.warning(
            "Series too short (%d obs) for seasonal period %d. "
            "Fitting non-seasonal ARIMA.",
            len(series),
            seasonal_period,
        )
        seasonal_period = 1

    use_seasonal = seasonal_period > 1

    # Split data into train and test sets for metrics calculation
    holdout = make_terminal_holdout(series, forecast_horizon)
    train = holdout.train

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
            information_criterion="aicc",
            test="kpss",
            seasonal_test="ocsb",
            max_d=2,
            max_D=1,
        )
        metrics = _calculate_metrics(holdout, train_model, mase_period)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("SARIMA training failed: %s", exc)

    # Determine the order and trend configuration from the training fit.
    order = train_model.order if train_model is not None else (1, 1, 1)
    seasonal_order = (
        train_model.seasonal_order
        if train_model is not None
        else (0, 0, 0, seasonal_period)
    )
    with_intercept = (
        getattr(train_model, "with_intercept", None)
        if train_model is not None
        else None
    )

    # Refit on the full series using the exact selected orders and intercept
    # configuration so the production forecast reflects the chosen model.
    full_model = pm.ARIMA(
        order=order,
        seasonal_order=seasonal_order,
        with_intercept=with_intercept,
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
    converged = bool(
        getattr(getattr(full_model, "arima_res_", None), "mle_retvals", {}).get(
            "converged", True
        )
    )
    roots_estimable = True
    try:
        ar_roots = np.asarray(full_model.arroots(), dtype=complex)
        ma_roots = np.asarray(full_model.maroots(), dtype=complex)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("SARIMA root diagnostics unavailable: %s", exc)
        roots_estimable = False
        ar_roots = np.asarray([], dtype=complex)
        ma_roots = np.asarray([], dtype=complex)
    stationary = bool(ar_roots.size == 0 or np.all(np.abs(ar_roots) > 1.0))
    invertible = bool(ma_roots.size == 0 or np.all(np.abs(ma_roots) > 1.0))
    fit_warnings: list[str] = []
    if not roots_estimable:
        fit_warnings.append("AR/MA root diagnostics were not estimable.")
    if use_seasonal and len(train) < 3 * seasonal_period:
        fit_warnings.append(
            "Fewer than three seasonal cycles are available; seasonal estimates are uncertain."
        )
    if not converged:
        fit_warnings.append("Maximum-likelihood optimization did not converge.")
    if not stationary:
        fit_warnings.append("Fitted AR roots do not satisfy stationarity.")
    if not invertible:
        fit_warnings.append("Fitted MA roots do not satisfy invertibility.")

    # Expose fitted innovations for residual diagnostics.
    innovations: list[float] = []
    try:
        resid = np.asarray(full_model.resid(), dtype=float)
        innovations = resid[np.isfinite(resid)].tolist()
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("SARIMA innovations unavailable: %s", exc)

    # AR+MA order sum (non-seasonal + seasonal) for the Ljung-Box df adjustment.
    ar_ma_order = (
        int(order[0]) + int(order[2]) + int(seasonal_order[0]) + int(seasonal_order[2])
    )

    status = (
        ForecastFitStatus.OK if metrics.rmse is not None else ForecastFitStatus.DEGRADED
    )
    failure_reason = (
        None if metrics.rmse is not None else metrics.unavailable_reasons.get("all")
    )

    return ForecastAdapterResult(
        status=(
            status
            if converged and stationary and invertible
            else ForecastFitStatus.DEGRADED
        ),
        failure_reason=failure_reason,
        is_fallback=train_model is None or not use_seasonal,
        forecast=forecast_values.tolist(),
        lower_ci=conf_int[:, 0].tolist(),
        upper_ci=conf_int[:, 1].tolist(),
        metrics=metrics,
        fitted_configuration={
            "model": "SARIMA",
            "order": list(full_model.order),
            "seasonal_order": list(full_model.seasonal_order),
            "trend": "c" if with_intercept else "n",
            "with_intercept": with_intercept,
            "seasonal_period": seasonal_period,
            "used_seasonal": use_seasonal,
            "ar_ma_order": ar_ma_order,
            "differencing_test": "kpss",
            "seasonal_differencing_test": "ocsb",
            "max_d": 2,
            "max_D": 1,
            "information_criterion": "aicc",
            "converged": converged,
            "stationary_roots": stationary,
            "invertible_roots": invertible,
            "root_diagnostics_estimable": roots_estimable,
        },
        warnings=fit_warnings,
        innovations=innovations,
        interval_label="prediction_interval",
    )
