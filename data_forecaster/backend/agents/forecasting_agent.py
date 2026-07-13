"""Forecasting agent that selects and runs statistical model implementations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.llm_factory import get_llm
from core.logging_config import get_logger
from forecasting.arima_model import fit_arima
from forecasting.backtesting import BacktestConfig, evaluate_candidates
from forecasting.contracts import (
    BacktestEvaluation,
    ForecastAdapterResult,
    ForecastFitStatus,
    ForecastMetrics,
)
from forecasting.ewma_model import fit_ewma
from forecasting.holt_winters import fit_holt_winters
from forecasting.residual_diagnostics import (
    analyze_backtest_errors,
    analyze_innovations,
    calibrate_interval_width,
)
from forecasting.selection_policy import CandidateEvidence, select_model_deterministic
from forecasting.sarima_model import fit_sarima
from forecasting.preprocessing import (
    BoxCoxTransform,
    FoldSafeImputer,
    FoldSafeOutlierTreatment,
    YeoJohnsonTransform,
    bias_adjusted_inverse,
    smooth_training_series,
)
from prompts.forecasting_prompt import FORECASTING_PROMPT
from schemas import (
    ForecastCandidateResult,
    ForecastResult,
    ModelSelectionResult,
    ResidualDiagnostics,
    StatisticalResult,
)
from utils.token_tracking import estimate_input_text, extract_token_usage

logger = get_logger(__name__)


def _has_required_metrics(result: ForecastAdapterResult) -> bool:
    """Return whether required comparison metrics are present and finite.

    A model is rankable only when ``status == ok`` and RMSE/MAE are present
    and finite. MAPE is deliberately optional because it is undefined for
    holdouts containing zero actual values.
    """
    if result.status != ForecastFitStatus.OK:
        return False
    for metric in (result.metrics.rmse, result.metrics.mae):
        if metric is None or not np.isfinite(metric):
            return False
    return True


def _format_metric(value: float | None, fmt: str) -> str:
    """Format a nullable metric, returning 'not available' when ``None``."""
    if value is None or not np.isfinite(value):
        return "not available"
    return format(value, fmt)


def run_forecasting_agent(
    series: pd.Series,
    model_selection: ModelSelectionResult,
    stat_result: StatisticalResult,
    forecast_horizon: int,
    freq: str,
    existing_metrics: dict[str, dict[str, float]] | None = None,
    disabled_tests: list[str] | None = None,
    loss_preference: str = "mase",
    preprocessing_options: dict[str, Any] | None = None,
) -> tuple[ForecastResult, dict[str, dict[str, float]]]:
    """Run all forecasting models, return ForecastResult for the selected model
    and an all-metrics dict for the comparison chart.

    Args:
        series: Historical time series data.
        model_selection: Output of the model selection agent.
        stat_result: Output of the statistical analysis agent.
        forecast_horizon: Number of periods to forecast.
        freq: Frequency string for generating forecast dates.
        existing_metrics: Optional pre-existing metrics dict (e.g. from a prior
            run or baseline models) to merge into the returned dict so that
            re-runs preserve previously computed metrics.
        disabled_tests: Optional list of residual diagnostic tests to skip.

    Returns:
        (ForecastResult, all_metrics_dict) where all_metrics_dict maps model
        names to ``{"RMSE": x, "MAE": y, "MAPE": z, "WAPE": w, "MASE": m}``.

    Raises:
        RuntimeError: If no forecasting model produces valid evaluation metrics.
    """
    seasonal_period = max(1, stat_result.seasonal_period or 1)
    preprocessing_options = preprocessing_options or {}
    outlier_strategy = {
        "Clip (Winsorize)": "clip",
        "clip": "clip",
        "Remove": "remove",
        "remove": "remove",
        "Z-Score Clip": "zscore_clip",
        "zscore_clip": "zscore_clip",
    }.get(preprocessing_options.get("outlier_strategy"), "none")
    imputation_method = preprocessing_options.get("missing_strategy", "interpolate")
    if imputation_method == "Let AI Decide":
        imputation_method = "interpolate"
    smoothing_method = preprocessing_options.get("smoothing", "none")
    production_series = FoldSafeOutlierTreatment(outlier_strategy).fit(
        series
    ).transform_training(series)
    production_series = FoldSafeImputer(imputation_method).fit(
        production_series
    ).transform_training(production_series)
    production_series = smooth_training_series(production_series, smoothing_method)
    results_store: dict[str, ForecastAdapterResult] = {}

    # ── Fit all models directly in Python ─────────────────────────────────────
    for name, fn, kwargs in [
        (
            "Holt-Winters",
            fit_holt_winters,
            {
                "seasonal_period": seasonal_period,
                "mase_period": seasonal_period,
            },
        ),
        ("ARIMA", fit_arima, {"mase_period": seasonal_period}),
        (
            "SARIMA",
            fit_sarima,
            {"seasonal_period": seasonal_period, "mase_period": seasonal_period},
        ),
        ("EWMA", fit_ewma, {"mase_period": seasonal_period}),
    ]:
        try:
            results_store[name] = fn(production_series, forecast_horizon, **kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("%s fitting failed: %s", name, exc)
            results_store[name] = ForecastAdapterResult(
                status=ForecastFitStatus.FAILED,
                failure_reason=str(exc),
                fitted_configuration={"model": name},
            )

    # ── Common rolling-origin backtesting ───────────────────────────────────
    # Run all candidates on identical expanding-window folds so the reported
    # metrics are apples-to-apples. The terminal-holdout metrics produced by
    # each adapter remain on the result; the backtest evaluation supplements
    # them with pooled rolling-origin evidence.
    backtest_evals = _run_backtest_evaluation(
        series,
        forecast_horizon,
        seasonal_period,
        apply_iqr_clip=False,
        imputation_method=imputation_method,
        smoothing_method=smoothing_method,
        outlier_strategy=outlier_strategy,
    )

    comparison_summary = "Model comparison metrics (lower is better):\n"
    for name, res in results_store.items():
        # Include status, warnings, and provenance in the evidence passed
        # to the LLM so it has versioned typed evidence.
        status_text = f" [status={res.status.value}]"
        if res.is_fallback:
            status_text += " [fallback]"
        if res.failure_reason:
            status_text += f" [failure={res.failure_reason}]"
        warnings_text = ""
        if res.warnings:
            warnings_text = f" [warnings: {'; '.join(res.warnings)}]"
        if not _has_required_metrics(res):
            comparison_summary += (
                f"- {name}:{status_text}{warnings_text} required metrics unavailable\n"
            )
            continue
        wape_text = (
            f", WAPE={_format_metric(res.metrics.wape, '.2%')}"
            if res.metrics.wape is not None
            else ""
        )
        mase_text = (
            f", MASE={_format_metric(res.metrics.mase, '.4f')}"
            if res.metrics.mase is not None
            else ""
        )
        backtest_text = ""
        bt = backtest_evals.get(name)
        if bt is not None and bt.pooled_metrics.rmse is not None:
            backtest_text = (
                f", backtest RMSE={bt.pooled_metrics.rmse:.4f} "
                f"(n_origins={bt.n_origins})"
            )
        interval_text = ""
        if res.interval_label:
            interval_text = f" [interval={res.interval_label}]"
        comparison_summary += (
            f"- {name}:{status_text}{warnings_text}{interval_text} "
            f"RMSE={res.metrics.rmse:.4f}, MAE={res.metrics.mae:.4f}, "
            f"MAPE={_format_metric(res.metrics.mape, '.2f')}%"
            f"{wape_text}{mase_text}{backtest_text}\n"
        )

    # ── LLM Setup ────────────────────────────────────────────────────────────
    llm = get_llm(temperature=0)

    prompt = FORECASTING_PROMPT
    token_usage: dict[str, int] = {}

    try:
        chain = prompt | llm
        inputs = {
            "selected": model_selection.selected_model,
            "summary": comparison_summary,
        }
        response = chain.invoke(inputs)
        token_usage = extract_token_usage(
            response, input_text=estimate_input_text(prompt, inputs)
        )
        reasoning_steps = [
            {
                "thought": "Fitting Holt-Winters, ARIMA, and SARIMA in Python...",
                "observation": comparison_summary,
            },
            {
                "thought": "Analyzing metrics for performance comparison...",
                "observation": response.content,
            },
        ]
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Forecasting agent LLM call failed: %s", exc)
        reasoning_steps = [
            {
                "thought": "LLM analysis failed, relying on direct Python metrics.",
                "observation": comparison_summary,
            }
        ]

    # ── Select from common rolling-origin evidence ───────────────────────────
    selected = model_selection.selected_model
    if model_selection.selection_method != "forced":
        rankable = {
            name: evaluation
            for name, evaluation in backtest_evals.items()
            if evaluation.is_rankable
            and (
                name not in results_store
                or results_store[name].status == ForecastFitStatus.OK
            )
        }
        if rankable:
            outcome = select_model_deterministic(
                [
                    CandidateEvidence(
                        name=name,
                        adapter_result=results_store.get(name),
                        backtest=evaluation,
                        is_baseline=name in _BASELINE_NAMES,
                    )
                    for name, evaluation in rankable.items()
                ],
                user_loss_preference=loss_preference,
            )
            if outcome.selected_model:
                selected = outcome.selected_model
    if selected not in results_store and selected in _BASELINE_NAMES:
        results_store[selected] = _fit_baseline_production(
            selected,
            production_series,
            forecast_horizon,
            seasonal_period,
            backtest_evals.get(selected),
        )
    if " + " in selected and selected not in results_store:
        results_store[selected] = _fit_transformed_production(
            selected,
            production_series,
            forecast_horizon,
            seasonal_period,
            backtest_evals.get(selected),
        )
    if selected not in results_store:
        # Try to fit the selected model directly
        try:
            if selected == "Holt-Winters":
                results_store[selected] = fit_holt_winters(
                    series,
                    forecast_horizon,
                    seasonal_period=seasonal_period,
                    mase_period=seasonal_period,
                )
            elif selected == "ARIMA":
                results_store[selected] = fit_arima(
                    series, forecast_horizon, mase_period=seasonal_period
                )
            elif selected == "EWMA":
                results_store[selected] = fit_ewma(
                    series, forecast_horizon, mase_period=seasonal_period
                )
            else:
                results_store[selected] = fit_sarima(
                    series,
                    forecast_horizon,
                    seasonal_period,
                    mase_period=seasonal_period,
                )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Could not fit selected model %s: %s", selected, exc)
            # Fall back to any available result
            if results_store:
                selected = next(iter(results_store))
                logger.warning("Falling back to %s", selected)
            else:
                raise RuntimeError("All forecasting models failed.") from exc

    res = results_store[selected]
    if not _has_required_metrics(res):
        rankable = {
            name: candidate
            for name, candidate in results_store.items()
            if _has_required_metrics(candidate)
        }
        if not rankable:
            raise RuntimeError(
                "No forecasting model produced valid evaluation metrics."
            )
        # Deterministic policy: lowest RMSE wins. The LLM never decides
        # model rankings.
        selected = min(
            rankable, key=lambda name: rankable[name].metrics.rmse or float("inf")
        )
        res = rankable[selected]
        res = res.model_copy(update={"is_fallback": True})
        logger.warning(
            "Selected model lacked valid evaluation evidence; falling back to %s",
            selected,
        )

    # ── Generate forecast dates ───────────────────────────────────────────────
    last_date = series.index[-1] if hasattr(series.index, "max") else None
    forecast_dates: list[str] = []
    if last_date is not None:
        try:
            date_range = pd.date_range(
                start=last_date, periods=forecast_horizon + 1, freq=freq
            )[1:]
            forecast_dates = date_range.strftime("%Y-%m-%d").tolist()
        except Exception:  # pylint: disable=broad-except
            forecast_dates = [str(i + 1) for i in range(forecast_horizon)]
    else:
        forecast_dates = [str(i + 1) for i in range(forecast_horizon)]

    # ── Build all_metrics dict for comparison chart ───────────────────────────
    all_metrics: dict[str, dict[str, float]] = {}
    for name, evaluation in backtest_evals.items():
        if not evaluation.is_rankable:
            continue
        metrics = evaluation.pooled_metrics
        all_metrics[name] = {
            "RMSE": metrics.rmse if metrics.rmse is not None else float("nan"),
            "MAE": metrics.mae if metrics.mae is not None else float("nan"),
            "MAPE": metrics.mape if metrics.mape is not None else float("nan"),
            "WAPE": metrics.wape if metrics.wape is not None else float("nan"),
            "MASE": metrics.mase if metrics.mase is not None else float("nan"),
            "sMAPE": metrics.smape if metrics.smape is not None else float("nan"),
            "RMSSE": metrics.rmsse if metrics.rmsse is not None else float("nan"),
        }
    # Merge any pre-existing metrics (e.g. baselines) passed in by the caller
    # so re-runs preserve previously computed results.
    if existing_metrics is not None:
        for name, metrics in existing_metrics.items():
            all_metrics.setdefault(name, metrics)

    # ── Residual Analysis ───────────────────────────────────────────────────
    residual_diagnostics = _run_residual_diagnostics(
        res, backtest_evals.get(selected), series, disabled_tests
    )
    lower_ci = res.lower_ci
    upper_ci = res.upper_ci
    interval_label = res.interval_label
    if (
        residual_diagnostics is not None
        and residual_diagnostics.coverage_estimable
        and lower_ci
        and upper_ci
    ):
        lower_ci, upper_ci = calibrate_interval_width(
            lower_ci,
            upper_ci,
            empirical_coverage=residual_diagnostics.interval_coverage,
            nominal_coverage=residual_diagnostics.nominal_coverage,
        )
        interval_label = "calibrated_prediction_interval"

    logger.info("Forecasting complete. Selected: %s", selected)

    selected_evaluation = backtest_evals.get(selected)
    reported_metrics = (
        selected_evaluation.pooled_metrics
        if selected_evaluation is not None and selected_evaluation.is_rankable
        else res.metrics
    )
    forecast_result = ForecastResult(
        model_used=selected,
        status=res.status,
        failure_reason=res.failure_reason,
        is_fallback=res.is_fallback,
        forecast=res.forecast,
        lower_ci=lower_ci,
        upper_ci=upper_ci,
        forecast_dates=forecast_dates,
        rmse=reported_metrics.rmse,
        mae=reported_metrics.mae,
        mape=reported_metrics.mape,
        wape=reported_metrics.wape,
        mase=reported_metrics.mase,
        smape=reported_metrics.smape,
        rmsse=reported_metrics.rmsse,
        residual_diagnostics=residual_diagnostics,
        candidate_results=[
            *[
                ForecastCandidateResult(
                    model=name,
                    status=candidate.status,
                    failure_reason=candidate.failure_reason,
                    is_fallback=candidate.is_fallback,
                    rmse=(
                        backtest_evals[name].pooled_metrics.rmse
                        if name in backtest_evals
                        else None
                    ),
                    mae=(
                        backtest_evals[name].pooled_metrics.mae
                        if name in backtest_evals
                        else None
                    ),
                    mape=(
                        backtest_evals[name].pooled_metrics.mape
                        if name in backtest_evals
                        else None
                    ),
                    wape=(
                        backtest_evals[name].pooled_metrics.wape
                        if name in backtest_evals
                        else None
                    ),
                    mase=(
                        backtest_evals[name].pooled_metrics.mase
                        if name in backtest_evals
                        else None
                    ),
                    smape=(
                        backtest_evals[name].pooled_metrics.smape
                        if name in backtest_evals
                        else None
                    ),
                    rmsse=(
                        backtest_evals[name].pooled_metrics.rmsse
                        if name in backtest_evals
                        else None
                    ),
                    n_evaluated=(
                        backtest_evals[name].n_evaluated
                        if name in backtest_evals
                        else 0
                    ),
                    n_missing=candidate.metrics.n_missing,
                    fitted_configuration=candidate.fitted_configuration,
                    warnings=candidate.warnings,
                    interval_label=candidate.interval_label,
                    validation_design=(
                        backtest_evals[name].validation_design
                        if name in backtest_evals
                        else {}
                    ),
                    metric_intervals=(
                        backtest_evals[name].metric_intervals
                        if name in backtest_evals
                        else {}
                    ),
                    skill_scores=(
                        backtest_evals[name].skill_scores
                        if name in backtest_evals
                        else {}
                    ),
                    final_test_metrics=(
                        backtest_evals[name].final_test_metrics.model_dump()
                        if name in backtest_evals
                        else {}
                    ),
                )
                for name, candidate in results_store.items()
            ],
            *[
                ForecastCandidateResult(
                    model=name,
                    status=(
                        ForecastFitStatus.OK
                        if evaluation.is_rankable
                        else ForecastFitStatus.NOT_ESTIMABLE
                    ),
                    rmse=evaluation.pooled_metrics.rmse,
                    mae=evaluation.pooled_metrics.mae,
                    mape=evaluation.pooled_metrics.mape,
                    wape=evaluation.pooled_metrics.wape,
                    mase=evaluation.pooled_metrics.mase,
                    smape=evaluation.pooled_metrics.smape,
                    rmsse=evaluation.pooled_metrics.rmsse,
                    n_evaluated=evaluation.n_evaluated,
                    warnings=evaluation.warnings,
                    interval_label="backtest_only",
                    validation_design=evaluation.validation_design,
                    metric_intervals=evaluation.metric_intervals,
                    skill_scores=evaluation.skill_scores,
                    final_test_metrics=evaluation.final_test_metrics.model_dump(),
                )
                for name, evaluation in backtest_evals.items()
                if name not in results_store
            ],
        ],
        reasoning_steps=reasoning_steps,
        token_usage=token_usage,
        interval_label=interval_label,
        validation_design=(
            selected_evaluation.validation_design if selected_evaluation else {}
        ),
        selection_metrics=reported_metrics.model_dump(
            include={"rmse", "mae", "mape", "wape", "mase", "smape", "rmsse"}
        ),
        final_test_metrics=(
            selected_evaluation.final_test_metrics.model_dump()
            if selected_evaluation is not None
            else {}
        ),
    )
    return forecast_result, all_metrics


_BASELINE_NAMES = {"Constant", "Naive", "Seasonal Naive", "Mean Forecast", "Drift"}


def _fit_baseline_production(
    name: str,
    series: pd.Series,
    horizon: int,
    seasonal_period: int,
    evaluation: BacktestEvaluation | None,
) -> ForecastAdapterResult:
    """Generate a full-history baseline forecast after common evaluation."""
    if name == "Constant":
        predictions = np.repeat(float(series.iloc[-1]), horizon)
    elif name == "Naive":
        predictions = np.repeat(float(series.iloc[-1]), horizon)
    elif name == "Seasonal Naive":
        season = series.iloc[-seasonal_period:].to_numpy(dtype=float)
        predictions = np.resize(season, horizon)
    elif name == "Mean Forecast":
        predictions = np.repeat(float(series.mean()), horizon)
    else:
        drift = float(series.iloc[-1] - series.iloc[0]) / max(1, len(series) - 1)
        predictions = np.asarray(
            [float(series.iloc[-1]) + step * drift for step in range(1, horizon + 1)]
        )
    return ForecastAdapterResult(
        status=ForecastFitStatus.OK,
        forecast=predictions.tolist(),
        metrics=(evaluation.pooled_metrics if evaluation else ForecastMetrics()),
        fitted_configuration={"model": name, "seasonal_period": seasonal_period},
        interval_label="unavailable",
    )


def _fit_transformed_production(
    name: str,
    series: pd.Series,
    horizon: int,
    mase_period: int,
    evaluation: BacktestEvaluation | None,
) -> ForecastAdapterResult:
    """Refit a fold-selected model/transform pipeline on the full history."""
    base_name, transform_name = name.split(" + ", maxsplit=1)
    transform = (
        BoxCoxTransform() if transform_name == "Box-Cox" else YeoJohnsonTransform()
    ).fit(series)
    transformed = transform.transform_series(series)
    fitters = {
        "ARIMA": lambda: fit_arima(transformed, horizon, mase_period=mase_period),
        "SARIMA": lambda: fit_sarima(
            transformed, horizon, mase_period, mase_period=mase_period
        ),
        "Holt-Winters": lambda: fit_holt_winters(
            transformed, horizon, seasonal_period=mase_period, mase_period=mase_period
        ),
        "EWMA": lambda: fit_ewma(transformed, horizon, mase_period=mase_period),
    }
    result = fitters[base_name]()
    configuration = dict(result.fitted_configuration)
    configuration["preprocessing"] = transform.transform.model_dump()
    configuration["retransformation_bias"] = "residual_smearing"
    residuals = np.asarray(result.innovations, dtype=float)
    return result.model_copy(
        update={
            "forecast": bias_adjusted_inverse(
                transform, result.forecast, residuals
            ).tolist(),
            "lower_ci": transform.inverse_transform(result.lower_ci).tolist(),
            "upper_ci": transform.inverse_transform(result.upper_ci).tolist(),
            "metrics": evaluation.pooled_metrics if evaluation else result.metrics,
            "fitted_configuration": configuration,
        }
    )


def _run_backtest_evaluation(
    series: pd.Series,
    forecast_horizon: int,
    seasonal_period: int,
    *,
    apply_iqr_clip: bool = False,
    imputation_method: str = "interpolate",
    smoothing_method: str = "none",
    outlier_strategy: str = "none",
) -> dict[str, Any]:
    """Run common rolling-origin backtesting for all four adapters.

    Every candidate is evaluated on identical expanding-window folds so the
    reported metrics are apples-to-apples. The backtest evaluation
    supplements (does not replace) the terminal-holdout metrics each adapter
    computes internally.

    Args:
        series:           Cleaned historical series.
        forecast_horizon:  Production forecast horizon.
        seasonal_period:   Seasonal period for MASE scale.

    Returns:
        Mapping of model name to :class:`BacktestEvaluation`.
    """
    from forecasting.backtesting import BacktestFold, FoldPrediction  # local

    config = BacktestConfig(
        horizon=min(forecast_horizon, max(1, len(series) // 5)),
        requested_horizon=forecast_horizon,
        max_origins=5,
        mase_period=seasonal_period,
        apply_iqr_clip=apply_iqr_clip,
        imputation_method=imputation_method,
        smoothing_method=smoothing_method,
        outlier_strategy=outlier_strategy,
        final_test_size=(forecast_horizon if len(series) >= 3 * forecast_horizon else 0),
    )

    def _arima_fn(train: pd.Series, fold: BacktestFold) -> FoldPrediction | None:
        from forecasting.pmdarima_compat import import_pmdarima  # local

        pm = import_pmdarima()
        try:
            model = pm.auto_arima(
                train,
                seasonal=False,
                stepwise=True,
                max_p=5,
                max_q=5,
                test="kpss",
                max_d=2,
                error_action="ignore",
                suppress_warnings=True,
                information_criterion="aicc",
            )
            preds, bounds = model.predict(n_periods=fold.horizon, return_conf_int=True)
            return FoldPrediction(
                predictions=np.asarray(preds, dtype=float),
                lower_ci=np.asarray(bounds[:, 0], dtype=float),
                upper_ci=np.asarray(bounds[:, 1], dtype=float),
                fitted_configuration={
                    "order": model.order,
                    "with_intercept": getattr(model, "with_intercept", None),
                    "_transformed_residuals": np.asarray(
                        model.resid(), dtype=float
                    ).tolist(),
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Backtest ARIMA fold %d failed: %s", fold.fold_index, exc)
            return None

    def _sarima_fn(train: pd.Series, fold: BacktestFold) -> FoldPrediction | None:
        from forecasting.pmdarima_compat import import_pmdarima  # local

        pm = import_pmdarima()
        use_seasonal = len(train) >= 2 * seasonal_period
        try:
            model = pm.auto_arima(
                train,
                seasonal=use_seasonal,
                m=seasonal_period if use_seasonal else 1,
                stepwise=True,
                max_p=3,
                max_q=3,
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
            preds, bounds = model.predict(n_periods=fold.horizon, return_conf_int=True)
            return FoldPrediction(
                predictions=np.asarray(preds, dtype=float),
                lower_ci=np.asarray(bounds[:, 0], dtype=float),
                upper_ci=np.asarray(bounds[:, 1], dtype=float),
                fitted_configuration={
                    "order": model.order,
                    "seasonal_order": model.seasonal_order,
                    "with_intercept": getattr(model, "with_intercept", None),
                    "_transformed_residuals": np.asarray(
                        model.resid(), dtype=float
                    ).tolist(),
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Backtest SARIMA fold %d failed: %s", fold.fold_index, exc)
            return None

    def _hw_fn(train: pd.Series, fold: BacktestFold) -> FoldPrediction | None:
        from forecasting.holt_winters import (  # local
            bootstrap_holt_winters_interval,
            select_holt_winters_fit,
        )

        try:
            fit, spec = select_holt_winters_fit(train, seasonal_period)
            pred_values = np.asarray(fit.forecast(fold.horizon), dtype=float)
            lower, upper = bootstrap_holt_winters_interval(
                fit,
                pred_values,
                seed=42 + fold.fold_index,
            )
            return FoldPrediction(
                predictions=pred_values,
                lower_ci=np.asarray(lower, dtype=float),
                upper_ci=np.asarray(upper, dtype=float),
                fitted_configuration={
                    "trend": spec.trend,
                    "damped_trend": spec.damped_trend,
                    "seasonal": spec.seasonal,
                    "seasonal_period": spec.seasonal_period,
                    "selection_criterion": "aicc",
                    "_transformed_residuals": np.asarray(
                        fit.resid, dtype=float
                    ).tolist(),
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Backtest Holt-Winters fold %d failed: %s", fold.fold_index, exc
            )
            return None

    def _ewma_fn(train: pd.Series, fold: BacktestFold) -> FoldPrediction | None:
        try:
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing  # local

            fit = SimpleExpSmoothing(train, initialization_method="estimated").fit(
                optimized=True
            )
            alpha = float(fit.params["smoothing_level"])
            preds = np.asarray(fit.forecast(fold.horizon), dtype=float)
            residuals = np.asarray(fit.resid, dtype=float)
            residuals = residuals[np.isfinite(residuals)]
            rng = np.random.default_rng(142 + fold.fold_index)
            simulated = preds[None, :] + rng.choice(
                residuals, size=(1000, fold.horizon), replace=True
            )
            return FoldPrediction(
                predictions=preds,
                lower_ci=np.quantile(simulated, 0.025, axis=0),
                upper_ci=np.quantile(simulated, 0.975, axis=0),
                fitted_configuration={
                    "alpha": alpha,
                    "_transformed_residuals": residuals.tolist(),
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Backtest EWMA fold %d failed: %s", fold.fold_index, exc)
            return None

    def _naive_fn(train: pd.Series, fold: BacktestFold) -> FoldPrediction:
        return FoldPrediction(
            predictions=np.repeat(float(train.iloc[-1]), fold.horizon),
            fitted_configuration={"model": "Naive"},
        )

    def _seasonal_naive_fn(
        train: pd.Series, fold: BacktestFold
    ) -> FoldPrediction | None:
        if seasonal_period <= 1 or len(train) < seasonal_period:
            return None
        return FoldPrediction(
            predictions=np.resize(
                train.iloc[-seasonal_period:].to_numpy(dtype=float), fold.horizon
            ),
            fitted_configuration={
                "model": "Seasonal Naive",
                "seasonal_period": seasonal_period,
            },
        )

    def _mean_fn(train: pd.Series, fold: BacktestFold) -> FoldPrediction:
        return FoldPrediction(
            predictions=np.repeat(float(train.mean()), fold.horizon),
            fitted_configuration={"model": "Mean Forecast"},
        )

    def _drift_fn(train: pd.Series, fold: BacktestFold) -> FoldPrediction | None:
        if len(train) < 2:
            return None
        drift = float(train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
        return FoldPrediction(
            predictions=np.asarray(
                [
                    float(train.iloc[-1]) + step * drift
                    for step in range(1, fold.horizon + 1)
                ]
            ),
            fitted_configuration={"model": "Drift"},
        )

    def _transformed_candidate(base_fn: Any, transform_type: type[Any]) -> Any:
        def candidate(train: pd.Series, fold: BacktestFold) -> FoldPrediction | None:
            transform = transform_type().fit(train)
            if not transform.transform.is_fitted:
                return None
            transformed = transform.transform_series(train)
            raw = base_fn(transformed, fold)
            if raw is None:
                return None
            configuration = dict(raw.fitted_configuration or {})
            residuals = np.asarray(
                configuration.pop("_transformed_residuals", []), dtype=float
            )
            return FoldPrediction(
                predictions=bias_adjusted_inverse(
                    transform, raw.predictions, residuals
                ),
                lower_ci=(
                    transform.inverse_transform(raw.lower_ci)
                    if raw.lower_ci is not None
                    else None
                ),
                upper_ci=(
                    transform.inverse_transform(raw.upper_ci)
                    if raw.upper_ci is not None
                    else None
                ),
                status=raw.status,
                warnings=raw.warnings,
                fitted_configuration={
                    **configuration,
                    "preprocessing": transform.transform.model_dump(),
                    "retransformation_bias": "residual_smearing",
                },
            )

        return candidate

    candidates = {
        "ARIMA": _arima_fn,
        "SARIMA": _sarima_fn,
        "Holt-Winters": _hw_fn,
        "EWMA": _ewma_fn,
        "Naive": _naive_fn,
        "Seasonal Naive": _seasonal_naive_fn,
        "Mean Forecast": _mean_fn,
        "Drift": _drift_fn,
    }
    finite = series.dropna().astype(float)
    if not finite.empty and bool(np.isclose(finite.std(ddof=0), 0.0)):
        candidates = {
            "Constant": lambda train, fold: FoldPrediction(
                predictions=np.repeat(float(train.iloc[-1]), fold.horizon),
                fitted_configuration={
                    "model": "Constant",
                    "reason": "constant_series",
                },
            )
        }
    if abs(float(series.skew())) > 1.0:
        transform_name = "Box-Cox" if bool((series > 0).all()) else "Yeo-Johnson"
        transform_type = (
            BoxCoxTransform if transform_name == "Box-Cox" else YeoJohnsonTransform
        )
        for base_name, base_fn in (
            ("ARIMA", _arima_fn),
            ("SARIMA", _sarima_fn),
            ("Holt-Winters", _hw_fn),
            ("EWMA", _ewma_fn),
        ):
            candidates[f"{base_name} + {transform_name}"] = _transformed_candidate(
                base_fn, transform_type
            )
    try:
        return evaluate_candidates(series, candidates, config=config)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Backtest evaluation failed: %s", exc)
        return {}


def _run_residual_diagnostics(
    result: ForecastAdapterResult,
    backtest: BacktestEvaluation | None,
    series: pd.Series,
    disabled_tests: list[str] | None,
) -> ResidualDiagnostics | None:
    """Prefer pooled out-of-sample errors; fall back to innovations."""
    try:
        successful = (
            [fold for fold in backtest.folds if fold.status == ForecastFitStatus.OK]
            if backtest
            else []
        )
        if successful:
            fold_errors: list[list[float]] = []
            fold_actuals: list[list[float]] = []
            fold_lower: list[list[float] | None] = []
            fold_upper: list[list[float] | None] = []
            for fold_result in successful:
                fold = fold_result.fold
                actuals = (
                    series.iloc[fold.test_start_index : fold.test_end_index]
                    .astype(float)
                    .tolist()
                )
                fold_errors.append(fold_result.residuals)
                fold_actuals.append(actuals)
                complete = len(fold_result.lower_ci) == len(actuals) and len(
                    fold_result.upper_ci
                ) == len(actuals)
                fold_lower.append(fold_result.lower_ci if complete else None)
                fold_upper.append(fold_result.upper_ci if complete else None)
            diag = analyze_backtest_errors(
                fold_errors,
                fold_actuals=fold_actuals,
                fold_lower=fold_lower,
                fold_upper=fold_upper,
                disabled_tests=disabled_tests or [],
            )
        elif result.innovations:
            diag = analyze_innovations(
                np.asarray(result.innovations, dtype=float),
                ar_ma_order=int(result.fitted_configuration.get("ar_ma_order", 0)),
                disabled_tests=disabled_tests or [],
            )
        else:
            return None
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Residual diagnostics failed: %s", exc)
        return None

    return ResidualDiagnostics(
        mean=diag.mean,
        is_zero_mean=diag.is_zero_mean,
        ljung_box_p_value=diag.ljung_box_p_value,
        is_uncorrelated=diag.is_uncorrelated,
        shapiro_wilk_p_value=diag.shapiro_p_value,
        is_normal=diag.is_normal,
        disabled_tests=sorted(set(disabled_tests or [])),
        error_type=diag.error_type,
        n_errors=diag.n_errors,
        mean_ci_lower=diag.mean_ci_lower,
        mean_ci_upper=diag.mean_ci_upper,
        ljung_box_lag=diag.ljung_box_lag,
        ljung_box_df_adjust=diag.ljung_box_df_adjust,
        variance_by_horizon=diag.variance_by_horizon,
        interval_coverage=diag.interval_coverage,
        interval_mean_width=diag.interval_mean_width,
        winkler_score=diag.winkler_score,
        weighted_interval_score=diag.weighted_interval_score,
        interval_coverage_by_horizon=diag.interval_coverage_by_horizon,
        interval_width_by_horizon=diag.interval_width_by_horizon,
        winkler_score_by_horizon=diag.winkler_score_by_horizon,
        nominal_coverage=diag.nominal_coverage,
        coverage_estimable=diag.coverage_estimable,
        warnings=diag.warnings,
    )
