"""One auditable forecasting procedure for folds, final tests, and production."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forecasting import registry
from forecasting.backtesting import BacktestConfig, FoldPrediction
from forecasting.contracts import BacktestFold, ForecastAdapterResult
from forecasting.diagnostics import detect_seasonality, _freq_to_period
from forecasting.intervals import path_intervals
from forecasting.preprocessing import (
    BoxCoxTransform,
    YeoJohnsonTransform,
    prepare_training_series,
)
from forecasting.window_models import fit_baseline_window

BASELINES = ("Naive", "Seasonal Naive", "Mean Forecast", "Drift")


def calendar_period(freq: str) -> int:
    """A fixed calendar prior and common MASE lag, never learned from targets."""
    from forecasting.dynamic_regression import calendar_periods

    periods = calendar_periods(freq)
    return max(1, _freq_to_period(freq) or (round(periods[0]) if periods else 1))


def backtest_config(
    length: int, horizon: int, freq: str, options: dict[str, Any]
) -> BacktestConfig:
    """Bound runtime while retaining at least two full-horizon selection origins."""
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive.")
    tuning = options.get("backtesting") or {}
    period = calendar_period(freq)
    minimum_train = max(10, 2 * period)
    reserve = horizon if length >= minimum_train + 3 * horizon else 0
    selection_length = length - reserve
    # Preserve the requested horizon unless there is insufficient history.
    feasible_horizon = max(
        1, (selection_length - min(minimum_train, max(2, selection_length // 2))) // 2
    )
    evaluated = min(horizon, feasible_horizon)
    initial = max(minimum_train, selection_length // 2)
    initial = min(initial, selection_length - 2 * evaluated)
    initial = max(2, initial)
    max_origins = int(tuning.get("max_origins", 8))
    step = int(tuning.get("step_size", evaluated))
    if not 2 <= max_origins <= 30 or step < 1:
        raise ValueError(
            "Backtesting requires 2–30 maximum origins and a positive step size."
        )
    return BacktestConfig(
        initial_train_size=initial,
        horizon=evaluated,
        requested_horizon=horizon,
        step_size=step,
        max_origins=max_origins,
        mase_period=period,
        final_test_size=reserve,
        quantile=float(options.get("point_quantile", 0.5)),
    )


@dataclass
class ForecastProcedure:
    """A fixed candidate definition with training-local adaptive choices."""

    name: str
    engine: ForecastEngine
    base_name: str
    transform: bool = False
    window: int | None = None
    members: tuple[str, ...] = ()
    handles_preprocessing: bool = True

    def fit(self, raw: pd.Series, horizon: int) -> ForecastAdapterResult:
        if self.members:
            fitted = [
                self.engine.candidates[name].fit(raw, horizon) for name in self.members
            ]
            # A mixture distribution is explicitly reported. Averaging interval
            # endpoints would pretend independent models have independent errors.
            samples = np.concatenate(
                [np.asarray(item.prediction_samples) for item in fitted]
            )
            lower, upper = path_intervals(samples)
            result = fitted[0].model_copy(deep=True)
            result.forecast = np.mean(
                [item.forecast for item in fitted], axis=0
            ).tolist()
            result.lower_ci, result.upper_ci = lower, upper
            result.innovations = []
            result.prediction_samples = samples.tolist()
            result.interval_label = "predictive_mixture_interval"
            result.fitted_configuration = {
                "model": self.name,
                "members": list(self.members),
                "weights": [1 / len(fitted)] * len(fitted),
            }
            if "point_quantile" in self.engine.options:
                q = self.engine.options["point_quantile"]
                result.forecast = np.quantile(samples, q, axis=0).tolist()
                result.fitted_configuration["point_quantile"] = q
            return result
        raw = raw.iloc[-self.window :] if self.window else raw
        train = prepare_training_series(raw, **self.engine.preprocessing)
        if not np.isfinite(train.to_numpy()).all():
            raise ValueError(
                "Training window contains no imputable finite observations."
            )
        transform = None
        if self.transform and abs(float(train.skew())) > 1.0:
            transform = (
                BoxCoxTransform() if (train > 0).all() else YeoJohnsonTransform()
            ).fit(train)
            if not transform.transform.is_fitted:
                raise ValueError("Training-window transformation was not estimable.")
        cache_key = (
            self.base_name,
            type(transform).__name__,
            horizon,
            len(raw),
            str(raw.index[0]),
            str(raw.index[-1]),
            hash(raw.to_numpy().tobytes()),
        )
        if cache_key in self.engine.cache:
            result = self.engine.cache[cache_key].model_copy(deep=True)
            result.fitted_configuration.update(
                procedure=self.name, training_window=self.window
            )
            return result
        disabled = (self.engine.options.get("statistical_tuning") or {}).get(
            "disabled_tests", []
        )
        evidence = detect_seasonality(
            train,
            metadata_period=calendar_period(self.engine.freq),
            disabled="periodogram" in disabled or "stl" in disabled,
        )
        period = evidence.selected_period
        model_train = transform.transform_series(train) if transform else train
        if self.base_name in BASELINES:
            result = fit_baseline_window(
                self.base_name,
                model_train,
                horizon,
                seasonal_period=calendar_period(self.engine.freq),
            )
        else:
            fitter = registry.MODELS[self.base_name]["window_fn"]
            extra = (
                {"options": self.engine.options}
                if self.base_name
                in {"Dynamic Regression", "Intermittent Demand", "Prophet"}
                else {}
            )
            result = fitter(
                model_train,
                horizon,
                seasonal_period=period,
                freq=self.engine.freq,
                **extra,
            )
        if transform:
            samples = transform.inverse_transform(np.asarray(result.prediction_samples))
            if not np.isfinite(samples).all():
                raise ValueError(
                    "Transformed predictive distribution leaves the inverse-transform domain."
                )
            result.forecast = np.mean(samples, axis=0).tolist()
            result.lower_ci, result.upper_ci = path_intervals(samples)
            result.prediction_samples = samples.tolist()
            # Transformed innovations are not original-unit errors.
            result.innovations = []
        lower_bound = self.engine.options.get("minimum_value")
        upper_bound = self.engine.options.get("maximum_value")
        if lower_bound is not None or upper_bound is not None:
            samples = np.clip(
                np.asarray(result.prediction_samples), lower_bound, upper_bound
            )
            result.prediction_samples = samples.tolist()
            result.forecast = np.mean(samples, axis=0).tolist()
            result.lower_ci, result.upper_ci = path_intervals(samples)
            result.fitted_configuration["target_bounds"] = [lower_bound, upper_bound]
        if "point_quantile" in self.engine.options:
            q = self.engine.options["point_quantile"]
            result.forecast = np.quantile(result.prediction_samples, q, axis=0).tolist()
            result.fitted_configuration["point_quantile"] = q
        result.fitted_configuration.update(
            {
                "procedure": self.name,
                "training_window": self.window,
                "training_observations": len(train),
                "seasonality": evidence.model_dump(),
                "preprocessing": (
                    transform.transform.model_dump() if transform else {"name": "none"}
                ),
                "retransformation_bias": (
                    "predictive_distribution_mean" if transform else "not_applicable"
                ),
            }
        )
        # Bound cached simulation storage to roughly one million floats.
        capacity = max(1, min(64, 1_000_000 // (2000 * max(1, horizon))))
        while len(self.engine.cache) >= capacity:
            self.engine.cache.pop(next(iter(self.engine.cache)))
        self.engine.cache[cache_key] = result
        return result.model_copy(deep=True)

    def __call__(self, raw: pd.Series, fold: BacktestFold) -> FoldPrediction:
        gap = fold.test_start_index - fold.train_end_index
        result = self.fit(raw, fold.horizon + gap)
        return FoldPrediction(
            predictions=np.asarray(result.forecast)[gap:],
            lower_ci=np.asarray(result.lower_ci)[gap:],
            upper_ci=np.asarray(result.upper_ci)[gap:],
            status=result.status,
            warnings=result.warnings,
            fitted_configuration=result.fitted_configuration,
        )


@dataclass
class ForecastEngine:
    freq: str
    options: dict[str, Any] = field(default_factory=dict)
    candidates: dict[str, ForecastProcedure] = field(init=False, default_factory=dict)
    cache: dict = field(init=False, default_factory=dict)

    @property
    def preprocessing(self) -> dict[str, str]:
        strategy = {
            "Clip (Winsorize)": "clip",
            "Remove": "remove",
            "Z-Score Clip": "zscore_clip",
            "clip": "clip",
            "remove": "remove",
            "zscore_clip": "zscore_clip",
        }.get(self.options.get("outlier_strategy"), "none")
        method = self.options.get("missing_strategy", "interpolate")
        return {
            "outlier_strategy": strategy,
            "imputation_method": "interpolate" if method == "Let AI Decide" else method,
            "smoothing_method": self.options.get("smoothing", "none"),
        }

    def __post_init__(self):
        if "point_quantile" in self.options:
            q = float(self.options["point_quantile"])
            if not 0 < q < 1:
                raise ValueError(
                    "The forecast quantile must be strictly between zero and one."
                )
            self.options["point_quantile"] = q
        for key in ("minimum_value", "maximum_value"):
            if self.options.get(key) in (None, ""):
                self.options.pop(key, None)
            else:
                value = float(self.options[key])
                if not np.isfinite(value):
                    raise ValueError("Target bounds must be finite.")
                self.options[key] = value
        if self.options.get("minimum_value", -np.inf) > self.options.get(
            "maximum_value", np.inf
        ):
            raise ValueError("The minimum target value cannot exceed the maximum.")
        enabled = registry.get_enabled_models()
        if self.options.get("demand_pattern") != "intermittent":
            enabled = tuple(name for name in enabled if name != "Intermittent Demand")
        for name in (*enabled, *BASELINES):
            self.candidates[name] = ForecastProcedure(name, self, name)
        for name in enabled:
            disabled = (self.options.get("statistical_tuning") or {}).get(
                "disabled_tests", []
            )
            if (
                name in {"Prophet", "Dynamic Regression", "Intermittent Demand"}
                or "box_cox" in disabled
            ):
                continue
            transformed = f"{name} + Auto transform"
            self.candidates[transformed] = ForecastProcedure(
                transformed, self, name, transform=True
            )
        # A fixed calendar-based window, not one selected from future breaks.
        window = int(
            self.options.get("recent_window", max(24, 4 * calendar_period(self.freq)))
        )
        if window < 10:
            raise ValueError(
                "The recent training window must contain at least 10 periods."
            )
        for name in enabled:
            recent = f"{name} + Recent window"
            self.candidates[recent] = ForecastProcedure(
                recent, self, name, window=window
            )
        members = tuple(
            name for name in ("Holt-Winters", "ARIMA", "EWMA") if name in enabled
        )
        if len(members) >= 2:
            self.candidates["Simple Ensemble"] = ForecastProcedure(
                "Simple Ensemble",
                self,
                "Simple Ensemble",
                members=members,
            )
