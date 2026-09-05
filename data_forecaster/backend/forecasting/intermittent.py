"""Opt-in TSB forecasting for nonnegative intermittent demand.

The demand probability is updated every period; positive demand size is
updated only on an occurrence. See Teunter, Syntetos & Babai (2011),
doi:10.1016/j.ejor.2011.05.018. Interval simulation is an explicit additional
Bernoulli/empirical-size assumption, not an analytical TSB guarantee.
"""

import numpy as np

from forecasting.intervals import path_intervals
from forecasting.window_models import _result


def tsb_states(values, alpha: float, beta: float):
    """Return final probability/size and genuinely one-step training errors."""
    first = int(np.flatnonzero(values > 0)[0])
    size = float(values[first])
    probability = 1.0 / (first + 1)
    errors = []
    for value in values[first + 1 :]:
        errors.append(float(value - probability * size))
        occurred = float(value > 0)
        probability += beta * (occurred - probability)
        if occurred:
            size += alpha * (value - size)
    return probability, size, np.asarray(errors)


def fit_intermittent_window(
    train, horizon, *, seasonal_period=1, freq=None, options=None
):
    values = train.to_numpy(dtype=float)
    positive = values[values > 0]
    if (
        len(values) < 10
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or positive.size < 2
        or not np.any(values == 0)
    ):
        raise ValueError(
            "TSB requires nonnegative demand, zero-demand periods, and at least two positive occurrences."
        )
    candidates = []
    for alpha in (0.05, 0.1, 0.2, 0.4):
        for beta in (0.05, 0.1, 0.2, 0.4):
            probability, size, errors = tsb_states(values, alpha, beta)
            if errors.size:
                candidates.append(
                    (np.mean(errors**2), alpha, beta, probability, size, errors)
                )
    _, alpha, beta, probability, size, errors = min(
        candidates, key=lambda item: item[0]
    )
    rng = np.random.default_rng(42)
    samples = rng.choice(positive, (2000, horizon)) * (size / positive.mean())
    if (options or {}).get("units") == "Count":
        # Stochastic rounding preserves the expected value and count support.
        floor = np.floor(samples)
        samples = floor + (rng.random(samples.shape) < samples - floor)
    samples *= rng.random(samples.shape) < probability
    lower, upper = path_intervals(samples)
    result = _result(
        "Intermittent Demand",
        np.repeat(probability * size, horizon),
        lower,
        upper,
        errors,
        {
            "method": "TSB",
            "alpha": alpha,
            "beta": beta,
            "demand_probability": probability,
            "positive_demand_size": size,
            "selection_criterion": "training_one_step_squared_error",
            "interval_assumption": "independent Bernoulli occurrences and resampled positive sizes",
        },
        samples,
    )
    result.interval_label = "experimental_bernoulli_size_interval"
    result.warnings.append(
        "TSB intervals use an additional simulation assumption; validate observed coverage before operational use."
    )
    return result


def fit_intermittent(series, forecast_horizon):
    return fit_intermittent_window(series, forecast_horizon)
