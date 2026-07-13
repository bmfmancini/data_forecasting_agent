"""ARIMA forecasting adapter backed by pmdarima auto_arima."""

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
    """Calculate RMSE, MAE, and MAPE for the given model and test data.

    Args:
        train: Training data used for MASE scale.
        test: Holdout observations.
        model: Trained ARIMA model with a ``predict`` method.

    Returns:
        Typed metrics. Unavailable evidence is never encoded as zero.
    """
    try:
        predictions, _ = model.predict(
            n_periods=len(holdout.test), return_conf_int=True
        )
        return evaluate_predictions(holdout, predictions, mase_period=mase_period)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("ARIMA metrics calculation failed: %s", exc)
        return ForecastMetrics(unavailable_reasons={"all": str(exc)})


def fit_arima(
    series: pd.Series, forecast_horizon: int, mase_period: int = 1
) -> ForecastAdapterResult:
    """Fit ARIMA via pmdarima auto_arima and return a typed adapter result.

    The adapter discovers an order on a training split, evaluates holdout
    metrics, then refits the *same* order (including trend/intercept
    configuration) on the full series for the production forecast.

    Args:
        series: A pandas Series containing the time series data.
        forecast_horizon: The number of periods to forecast.

    Returns:
        :class:`ForecastAdapterResult` with status, forecast, intervals,
        nullable metrics, and fitted configuration provenance.
    """
    series = series.dropna().astype(float)

    if len(series) < 3:
        logger.warning(
            "Series too short for ARIMA (%d points). Returning persistence forecast.",
            len(series),
        )
        last_val = float(series.iloc[-1]) if not series.empty else 0.0
        return ForecastAdapterResult(
            status=ForecastFitStatus.NOT_ESTIMABLE,
            failure_reason="ARIMA requires at least three observations.",
            is_fallback=True,
            forecast=[last_val] * forecast_horizon,
            lower_ci=[last_val] * forecast_horizon,
            upper_ci=[last_val] * forecast_horizon,
            fitted_configuration={
                "model": "ARIMA",
                "order": None,
                "trend": None,
                "with_intercept": None,
                "fallback": "persistence",
            },
        )

    # Split data into train and test sets for metrics calculation
    holdout = make_terminal_holdout(series, forecast_horizon)
    train = holdout.train

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
                information_criterion="aicc",
                test="kpss",
                max_d=2,
            )
            metrics = _calculate_metrics(holdout, train_model, mase_period)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("ARIMA training failed: %s", exc)

    # Determine the order and trend configuration from the training fit.
    order = train_model.order if train_model is not None else (1, 1, 1)
    # pmdarima exposes ``with_intercept`` on the fitted model; preserve it
    # so the full-series refit matches the selected specification.
    with_intercept = (
        getattr(train_model, "with_intercept", None)
        if train_model is not None
        else None
    )

    # Refit on the full series using the exact selected order and intercept
    # configuration so the production forecast reflects the chosen model.
    full_model = pm.ARIMA(
        order=order,
        with_intercept=with_intercept,
        suppress_warnings=True,
    ).fit(series)

    logger.info("ARIMA selected order: %s", full_model.order)

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
        logger.warning("ARIMA root diagnostics unavailable: %s", exc)
        roots_estimable = False
        ar_roots = np.asarray([], dtype=complex)
        ma_roots = np.asarray([], dtype=complex)
    stationary = bool(ar_roots.size == 0 or np.all(np.abs(ar_roots) > 1.0))
    invertible = bool(ma_roots.size == 0 or np.all(np.abs(ma_roots) > 1.0))
    fit_warnings: list[str] = []
    if not roots_estimable:
        fit_warnings.append("AR/MA root diagnostics were not estimable.")
    if not converged:
        fit_warnings.append("Maximum-likelihood optimization did not converge.")
    if not stationary:
        fit_warnings.append("Fitted AR roots do not satisfy stationarity.")
    if not invertible:
        fit_warnings.append("Fitted MA roots do not satisfy invertibility.")

    forecast_values, conf_int = full_model.predict(
        n_periods=forecast_horizon, return_conf_int=True
    )

    # Expose fitted innovations for residual diagnostics.
    innovations: list[float] = []
    try:
        resid = np.asarray(full_model.resid(), dtype=float)
        innovations = resid[np.isfinite(resid)].tolist()
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("ARIMA innovations unavailable: %s", exc)

    # AR+MA order sum for the Ljung-Box degrees-of-freedom adjustment.
    ar_ma_order = int(order[0]) + int(order[2])

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
        is_fallback=train_model is None,
        forecast=forecast_values.tolist(),
        lower_ci=conf_int[:, 0].tolist(),
        upper_ci=conf_int[:, 1].tolist(),
        metrics=metrics,
        fitted_configuration={
            "model": "ARIMA",
            "order": list(full_model.order),
            "trend": "c" if with_intercept else "n",
            "with_intercept": with_intercept,
            "refit_order": list(order),
            "ar_ma_order": ar_ma_order,
            "differencing_test": "kpss",
            "max_d": 2,
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
