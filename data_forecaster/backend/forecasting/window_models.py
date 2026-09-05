"""Training-window fitters shared by validation and production.

No fitter reads validation targets or creates another holdout. Model choices
and parameter estimation use all and only the supplied training observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from forecasting.contracts import ForecastAdapterResult, ForecastFitStatus
from forecasting.holt_winters import select_holt_winters_fit
from forecasting.intervals import path_intervals, smoothing_paths


def _result(name, forecast, lower, upper, residuals, configuration, paths=None):
    forecast = np.asarray(forecast, dtype=float)
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (
        forecast.ndim != 1
        or not np.isfinite(forecast).all()
        or lo.shape != forecast.shape
        or hi.shape != forecast.shape
        or not np.isfinite(lo).all()
        or not np.isfinite(hi).all()
        or np.any(lo > hi)
    ):
        raise ValueError(f"{name} produced invalid forecasts or intervals.")
    innovations = np.asarray(residuals, dtype=float)
    return ForecastAdapterResult(
        status=ForecastFitStatus.OK,
        forecast=forecast.tolist(),
        lower_ci=lo.tolist(),
        upper_ci=hi.tolist(),
        innovations=innovations[np.isfinite(innovations)].tolist(),
        fitted_configuration={
            "model": name,
            "selection_scope": "supplied_training_window",
            "parameter_uncertainty_included": False,
            **configuration,
        },
        prediction_samples=np.asarray(paths).tolist() if paths is not None else [],
        interval_label="model_based_prediction_interval",
    )


def fit_arima_window(train: pd.Series, horizon: int, *, seasonal_period=1, freq=None):
    return _fit_arima(train, horizon, seasonal_period=1, name="ARIMA")


def fit_sarima_window(train: pd.Series, horizon: int, *, seasonal_period=1, freq=None):
    return _fit_arima(train, horizon, seasonal_period=seasonal_period, name="SARIMA")


def _fit_arima(train, horizon, *, seasonal_period, name):
    from forecasting.pmdarima_compat import import_pmdarima

    seasonal = seasonal_period > 1 and len(train) >= 2 * seasonal_period
    if len(train) < 3:
        raise ValueError("ARIMA requires at least three training observations.")
    model = import_pmdarima().auto_arima(
        train,
        seasonal=seasonal,
        m=seasonal_period if seasonal else 1,
        stepwise=True,
        max_p=3 if seasonal else 5,
        max_q=3 if seasonal else 5,
        max_P=2,
        max_Q=2,
        max_order=10,
        test="kpss",
        seasonal_test="ocsb",
        max_d=2,
        max_D=1,
        error_action="ignore",
        suppress_warnings=True,
        information_criterion="aicc",
    )
    if not model.arima_res_.mle_retvals.get("converged", True):
        raise ValueError("ARIMA optimization did not converge.")
    point, bounds = model.predict(n_periods=horizon, return_conf_int=True, alpha=0.05)
    point, bounds = np.asarray(point), np.asarray(bounds)
    # Marginal Gaussian draws preserve the fitted h-step forecast variance.
    # They are used only for marginal intervals/retransformation, not path events.
    sigma = (bounds[:, 1] - bounds[:, 0]) / (2 * norm.ppf(0.975))
    paths = (
        point[None, :] + np.random.default_rng(42).normal(size=(2000, horizon)) * sigma
    )
    return _result(
        name,
        point,
        bounds[:, 0],
        bounds[:, 1],
        model.resid(),
        {
            "order": list(model.order),
            "seasonal_order": list(model.seasonal_order),
            "seasonal_period": seasonal_period if seasonal else 1,
            "with_intercept": model.with_intercept,
            "ar_ma_order": sum(model.order[::2]) + sum(model.seasonal_order[:3:2]),
            "sample_method": "marginal_gaussian_forecast_distribution",
        },
        paths,
    )


def fit_holt_winters_window(train, horizon, *, seasonal_period=1, freq=None):
    fit, spec = select_holt_winters_fit(train, seasonal_period)
    if not getattr(fit, "mle_retvals", {}).get("success", True):
        raise ValueError("Holt-Winters optimization did not converge.")
    paths = smoothing_paths(fit, horizon)
    lo, hi = path_intervals(paths)
    return _result(
        "Holt-Winters",
        fit.forecast(horizon),
        lo,
        hi,
        fit.resid,
        {
            **spec.__dict__,
            "selection_criterion": "aicc",
            "sample_method": "state_recursion",
        },
        paths,
    )


def fit_ewma_window(train, horizon, *, seasonal_period=1, freq=None):
    if len(train) < 3:
        raise ValueError("EWMA requires at least three training observations.")
    fit = SimpleExpSmoothing(train, initialization_method="estimated").fit(
        optimized=True
    )
    if not getattr(fit, "mle_retvals", {}).get("success", True):
        raise ValueError("EWMA optimization did not converge.")
    paths = smoothing_paths(fit, horizon)
    lo, hi = path_intervals(paths)
    return _result(
        "EWMA",
        fit.forecast(horizon),
        lo,
        hi,
        fit.resid,
        {
            "alpha": float(fit.params["smoothing_level"]),
            "sample_method": "state_recursion",
        },
        paths,
    )


def fit_prophet_window(train, horizon, *, seasonal_period=1, freq=None):
    from forecasting.prophet_compat import import_prophet
    from forecasting.prophet_model import _to_history_frame, _future_frame

    if len(train) < 3:
        raise ValueError("Prophet requires at least three training observations.")
    history = _to_history_frame(train)
    model = import_prophet().Prophet(interval_width=0.95, uncertainty_samples=0)
    model.fit(history)
    future = _future_frame(history, horizon, freq)
    prediction = model.predict(future)
    # Prophet's uncertainty sampler uses global numpy state. Run a deterministic
    # sampler under a lock and restore that state for concurrent forecast jobs.
    from forecasting.prophet_model import prophet_predictive_samples

    paths = prophet_predictive_samples(model, future)
    lo, hi = path_intervals(paths)
    fitted = model.predict(history)["yhat"].to_numpy()
    return _result(
        "Prophet",
        prediction["yhat"],
        lo,
        hi,
        train.to_numpy() - fitted,
        {
            "interval_width": 0.95,
            "sample_method": "prophet_predictive_samples",
        },
        paths,
    )


def fit_baseline_window(name, train, horizon, *, seasonal_period=1):
    values = train.to_numpy(dtype=float)
    if len(values) < 2:
        raise ValueError("Baselines require at least two training observations.")
    rng = np.random.default_rng(42)
    steps = np.arange(1, horizon + 1)
    period = seasonal_period if name == "Seasonal Naive" else 1
    if name == "Seasonal Naive" and (period < 2 or len(values) < 2 * period):
        raise ValueError("Seasonal naive needs two complete training cycles.")
    if name == "Mean Forecast":
        point = np.repeat(values.mean(), horizon)
        errors = values - values.mean()
        noise = rng.choice(errors, (2000, horizon))
        # Include sampling uncertainty in the estimated level.
        noise += rng.normal(0, errors.std(ddof=1) / np.sqrt(len(values)), (2000, 1))
        paths = point + noise
    else:
        drift = (values[-1] - values[0]) / (len(values) - 1) if name == "Drift" else 0.0
        point = np.resize(values[-period:], horizon) + steps * drift
        errors = values[period:] - values[:-period] - period * drift
        errors -= errors.mean()
        noise = rng.choice(errors, (2000, horizon))
        # Seasonal errors accumulate only along the same seasonal position.
        for h in range(period, horizon):
            noise[:, h] += noise[:, h - period]
        if name == "Drift":
            noise += (
                rng.normal(0, errors.std(ddof=1) / np.sqrt(len(errors)), (2000, 1))
                * steps
            )
        paths = point + noise
    lo, hi = path_intervals(paths)
    return _result(
        name,
        point,
        lo,
        hi,
        errors,
        {
            "seasonal_period": period,
            "sample_method": "baseline_recursion",
        },
        paths,
    )
