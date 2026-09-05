"""Calendar/Fourier regression with ARIMA errors and known-ahead predictors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from forecasting.window_models import _result


def calendar_periods(freq: str) -> tuple[float, ...]:
    """Multiple calendar cycles in observation units, including subdaily data."""
    offset = pd.tseries.frequencies.to_offset(freq)
    name = offset.name.upper().split("-")[0]
    multiple = offset.n
    if name in {"MS", "ME", "M"}:
        return (12 / multiple,)
    if name in {"QS", "QE", "Q"}:
        return (4 / multiple,)
    if name == "B":
        return (5 / multiple, 260 / multiple)
    if name == "W":
        return (365.25 / (7 * multiple),)
    try:
        days = offset.nanos / pd.Timedelta(days=1).value
    except ValueError:
        return ()
    candidates = (1 / days, 7 / days, 365.25 / days)
    return tuple(period for period in candidates if period >= 2)


def design_matrix(
    train: pd.Series, horizon: int, freq: str, harmonics: int, options: dict
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build deterministic calendar features and explicitly known-ahead inputs."""
    n = len(train)
    steps = np.arange(n + horizon, dtype=float)
    columns = {"trend": steps / max(1, n)}
    for period in calendar_periods(freq):
        if n < 2 * period:
            continue
        for k in range(1, min(harmonics, int(period // 2)) + 1):
            phase = 2 * np.pi * k * steps / period
            columns[f"sin_{period:g}_{k}"] = np.sin(phase)
            columns[f"cos_{period:g}_{k}"] = np.cos(phase)
    covariates = options.get("known_covariates") or {}
    if covariates:
        if options.get("covariates_known_in_advance") is not True:
            raise ValueError(
                "Predictors must be declared known at each historical forecast origin."
            )
        if not isinstance(train.index, pd.DatetimeIndex):
            raise ValueError("Known-ahead predictors require a datetime index.")
        future = pd.date_range(train.index[-1], periods=horizon + 1, freq=freq)[1:]
        dates = train.index.append(future)
        for name, observations in covariates.items():
            values = pd.Series(observations, dtype=float)
            values.index = pd.to_datetime(values.index)
            if not values.index.is_unique:
                raise ValueError(f"Predictor '{name}' has duplicate timestamps.")
            aligned = values.reindex(dates).to_numpy(dtype=float)
            if not np.isfinite(aligned).all():
                raise ValueError(
                    f"Predictor '{name}' needs finite values at every training and forecast timestamp."
                )
            columns[f"input_{name}"] = aligned
    names, features = [], []
    for name, values in columns.items():
        scale = values[:n].std()
        if scale < 1e-8:
            continue  # a constant training predictor has no estimable effect
        names.append(name)
        features.append((values - values[:n].mean()) / scale)
    matrix = np.column_stack(features)
    return matrix[:n], matrix[n:], names


def fit_dynamic_window(train, horizon, *, seasonal_period=1, freq=None, options=None):
    """Choose Fourier complexity using training AICc; never inspect future y."""
    from forecasting.pmdarima_compat import import_pmdarima

    if freq is None or len(train) < 12:
        raise ValueError(
            "Dynamic regression needs a calendar frequency and at least 12 observations."
        )
    options = options or {}
    fitted = []
    seen = set()
    for harmonics in (1, 2):
        x_train, x_future, names = design_matrix(
            train, horizon, freq, harmonics, options
        )
        if tuple(names) in seen or len(train) < 3 * (len(names) + 2):
            continue
        seen.add(tuple(names))
        try:
            model = import_pmdarima().auto_arima(
                train,
                X=x_train,
                seasonal=False,
                stepwise=True,
                start_p=0,
                start_q=0,
                max_p=2,
                max_q=2,
                max_d=1,
                information_criterion="aicc",
                error_action="ignore",
                suppress_warnings=True,
            )
            if not model.arima_res_.mle_retvals.get("converged", True):
                continue
            if np.isfinite(model.aicc()):
                fitted.append((model.aicc(), model, x_future, names, harmonics))
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not fitted:
        raise ValueError("No converged dynamic regression specification was estimable.")
    _, model, x_future, names, harmonics = min(fitted, key=lambda item: item[0])
    point, bounds = model.predict(
        n_periods=horizon, X=x_future, return_conf_int=True, alpha=0.05
    )
    point, bounds = np.asarray(point), np.asarray(bounds)
    sigma = (bounds[:, 1] - bounds[:, 0]) / (2 * norm.ppf(0.975))
    samples = point + np.random.default_rng(42).normal(size=(2000, horizon)) * sigma
    return _result(
        "Dynamic Regression",
        point,
        bounds[:, 0],
        bounds[:, 1],
        model.resid(),
        {
            "features": names,
            "harmonics": harmonics,
            "order": list(model.order),
            "ar_ma_order": model.order[0] + model.order[2],
            "selection_criterion": "aicc",
            "predictor_uncertainty_included": False,
        },
        samples,
    )


def fit_dynamic_regression(series, forecast_horizon, freq=None):
    """Registry compatibility entry point; validation belongs to the engine."""
    return fit_dynamic_window(series, forecast_horizon, freq=freq)
