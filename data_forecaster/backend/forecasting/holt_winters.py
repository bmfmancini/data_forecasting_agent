"""Holt-Winters forecasting with training-only model-form selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from core.logging_config import get_logger
from forecasting.contracts import (
    ForecastAdapterResult,
    ForecastFitStatus,
    ForecastMetrics,
)
from forecasting.evaluation import evaluate_predictions, make_terminal_holdout
from forecasting.intervals import ets_prediction_interval

logger = get_logger(__name__)


@dataclass(frozen=True)
class HoltWintersSpec:
    """One admissible Holt-Winters configuration."""

    trend: str | None
    damped_trend: bool
    seasonal: str | None
    seasonal_period: int | None


def _candidate_specs(train: pd.Series, seasonal_period: int) -> list[HoltWintersSpec]:
    """Build admissible model forms for the training window."""
    trend_specs = [(None, False), ("add", False), ("add", True)]
    seasonal_specs: list[tuple[str | None, int | None]] = [(None, None)]
    if seasonal_period > 1 and len(train) >= 2 * seasonal_period:
        seasonal_specs.append(("add", seasonal_period))
        if (train > 0).all():
            seasonal_specs.append(("mul", seasonal_period))
    return [
        HoltWintersSpec(trend, damped, seasonal, period)
        for trend, damped in trend_specs
        for seasonal, period in seasonal_specs
    ]


def select_holt_winters_fit(
    train: pd.Series,
    seasonal_period: int,
) -> tuple[object, HoltWintersSpec]:
    """Fit candidate forms on training data and return the lowest-AICc fit."""
    fitted: list[tuple[float, object, HoltWintersSpec]] = []
    failures: list[str] = []
    for spec in _candidate_specs(train, seasonal_period):
        try:
            result = ExponentialSmoothing(
                train,
                trend=spec.trend,
                damped_trend=spec.damped_trend,
                seasonal=spec.seasonal,
                seasonal_periods=spec.seasonal_period,
                initialization_method="estimated",
            ).fit(optimized=True)
            criterion = float(getattr(result, "aicc", result.aic))
            if not np.isfinite(criterion):
                criterion = float(result.aic)
            if np.isfinite(criterion):
                fitted.append((criterion, result, spec))
        except Exception as exc:  # pylint: disable=broad-except
            failures.append(f"{spec}: {exc}")
    if not fitted:
        raise ValueError("No Holt-Winters form was estimable: " + "; ".join(failures))
    _, best_fit, best_spec = min(fitted, key=lambda item: item[0])
    return best_fit, best_spec


def bootstrap_holt_winters_interval(
    fitted: object,
    point_forecast: np.ndarray,
    *,
    seed: int = 42,
    repetitions: int = 1000,
) -> tuple[list[float], list[float]]:
    """Bootstrap paths through the selected Holt-Winters state equations."""
    return ets_prediction_interval(
        fitted,
        int(point_forecast.size),
        repetitions=repetitions,
        seed=seed,
    )


def fit_holt_winters(
    series: pd.Series,
    forecast_horizon: int,
    seasonal_period: int = 1,
    mase_period: int = 1,
    fitted_configuration: dict[str, object] | None = None,
    evaluation_metrics: ForecastMetrics | None = None,
) -> ForecastAdapterResult:
    """Select the Holt-Winters form on training data and refit it on all data."""
    series = series.dropna().astype(float)
    seasonal_period = max(1, int(seasonal_period))
    selected = _spec_from_configuration(fitted_configuration)
    if selected is not None:
        metrics = evaluation_metrics or ForecastMetrics()
    else:
        holdout = make_terminal_holdout(series, forecast_horizon)
        train, test = holdout.train, holdout.test

        try:
            train_fit, selected = select_holt_winters_fit(train, seasonal_period)
            metrics = evaluate_predictions(
                holdout,
                np.asarray(train_fit.forecast(len(test)), dtype=float),
                mase_period=mase_period,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Holt-Winters model selection failed: %s", exc)
            last_val = float(series.iloc[-1]) if not series.empty else 0.0
            return ForecastAdapterResult(
                status=ForecastFitStatus.NOT_ESTIMABLE,
                failure_reason=str(exc),
                is_fallback=True,
                forecast=[last_val] * forecast_horizon,
                lower_ci=[last_val] * forecast_horizon,
                upper_ci=[last_val] * forecast_horizon,
                metrics=ForecastMetrics(unavailable_reasons={"all": str(exc)}),
                fitted_configuration={
                    "model": "Holt-Winters",
                    "requested_seasonal_period": seasonal_period,
                    "fallback": "persistence",
                },
            )

    try:
        full_fit = ExponentialSmoothing(
            series,
            trend=selected.trend,
            damped_trend=selected.damped_trend,
            seasonal=selected.seasonal,
            seasonal_periods=selected.seasonal_period,
            initialization_method="estimated",
        ).fit(optimized=True)
        forecast = np.asarray(full_fit.forecast(forecast_horizon), dtype=float)
        lower, upper = bootstrap_holt_winters_interval(full_fit, forecast)
        residuals = np.asarray(full_fit.resid, dtype=float)
        innovations = residuals[np.isfinite(residuals)].tolist()
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Holt-Winters full refit failed: %s", exc)
        return ForecastAdapterResult(
            status=ForecastFitStatus.FAILED,
            failure_reason=str(exc),
            metrics=metrics,
            fitted_configuration={"model": "Holt-Winters", **selected.__dict__},
        )

    return ForecastAdapterResult(
        status=ForecastFitStatus.OK,
        forecast=forecast.tolist(),
        lower_ci=lower,
        upper_ci=upper,
        metrics=metrics,
        fitted_configuration={
            "model": "Holt-Winters",
            "trend": selected.trend,
            "damped_trend": selected.damped_trend,
            "seasonal": selected.seasonal,
            "seasonal_period": selected.seasonal_period,
            "requested_seasonal_period": seasonal_period,
            "selection_criterion": "aicc",
            "selection_scope": "training_window",
            "initialization_method": "estimated",
            "parameter_uncertainty_included": False,
        },
        innovations=innovations,
        interval_label="bootstrap_prediction_interval",
    )


def _spec_from_configuration(
    fitted_configuration: dict[str, object] | None,
) -> HoltWintersSpec | None:
    """Build a validated Holt-Winters specification from backtest evidence."""
    if not fitted_configuration or "damped_trend" not in fitted_configuration:
        return None
    trend = fitted_configuration.get("trend")
    seasonal = fitted_configuration.get("seasonal")
    period = fitted_configuration.get("seasonal_period")
    if trend not in {None, "add"} or seasonal not in {None, "add", "mul"}:
        return None
    if period is not None:
        try:
            period = int(period)
        except (TypeError, ValueError):
            return None
    return HoltWintersSpec(
        trend=trend,
        damped_trend=bool(fitted_configuration.get("damped_trend")),
        seasonal=seasonal,
        seasonal_period=period,
    )


def refit_holt_winters_from_configuration(
    series: pd.Series,
    forecast_horizon: int,
    seasonal_period: int,
    fitted_configuration: dict[str, object],
    metrics: ForecastMetrics,
) -> ForecastAdapterResult:
    """Refit a backtest-selected Holt-Winters form on full history."""
    if _spec_from_configuration(fitted_configuration) is None:
        return ForecastAdapterResult(
            status=ForecastFitStatus.NOT_ESTIMABLE,
            failure_reason="Reusable Holt-Winters configuration is unavailable.",
            metrics=metrics,
            fitted_configuration={"model": "Holt-Winters"},
        )
    return fit_holt_winters(
        series,
        forecast_horizon,
        seasonal_period=seasonal_period,
        mase_period=seasonal_period,
        fitted_configuration=fitted_configuration,
        evaluation_metrics=metrics,
    )
