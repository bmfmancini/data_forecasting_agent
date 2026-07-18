"""Forecasting agent that selects and runs statistical model implementations."""

from __future__ import annotations

from dataclasses import dataclass
import re
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from core.llm_factory import get_llm
from core.logging_config import get_logger
from utils.memory import memory_snapshot
from forecasting.arima_model import fit_arima, refit_arima_from_configuration
from forecasting.backtesting import BacktestConfig, evaluate_candidates
from forecasting.contracts import (
    BacktestEvaluation,
    ForecastAdapterResult,
    ForecastFitStatus,
    ForecastMetrics,
)
from forecasting.ewma_model import fit_ewma, refit_ewma_from_configuration
from forecasting.holt_winters import (
    fit_holt_winters,
    refit_holt_winters_from_configuration,
)
from forecasting.residual_diagnostics import (
    analyze_backtest_errors,
    analyze_innovations,
    calibrate_interval_width,
    interval_nonconformity_scores,
)
from forecasting.selection_policy import CandidateEvidence, select_model_deterministic
from forecasting.sarima_model import fit_sarima, refit_sarima_from_configuration
from forecasting.preprocessing import (
    BoxCoxTransform,
    YeoJohnsonTransform,
    bias_adjusted_inverse,
    prepare_training_series,
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

_SUPPORTED_LOSS_METRICS = ("mase", "wape", "rmse", "mae")
_AUTO_LOSS_VALUES = {"auto", "ai", "recommended", "let ai decide"}


@dataclass(frozen=True)
class ForecastEvaluationEvidence:
    """Reusable comparison evidence for one unchanged forecast request."""

    backtest_evaluations: dict[str, BacktestEvaluation]
    all_metrics: dict[str, dict[str, float]]
    comparison_summary: str
    resolved_loss: str
    loss_resolution_source: str
    loss_rationale: str
    reasoning_steps: list[dict[str, Any]]
    token_usage: dict[str, int]


def _has_required_metrics(result: ForecastAdapterResult) -> bool:
    """Compatibility wrapper for the contract's rankability rule."""
    return result.is_rankable


def _format_metric(value: float | None, fmt: str) -> str:
    """Format a nullable metric, returning 'not available' when ``None``."""
    if value is None or not np.isfinite(value):
        return "not available"
    return format(value, fmt)


def _business_context(options: dict[str, Any]) -> str:
    """Format decision-relevant context for loss recommendation."""
    keys = (
        "user_context",
        "data_domain",
        "units",
        "interventions",
        "censoring_or_stockouts",
        "known_future_covariates",
        "aggregation",
        "minimum_value",
        "maximum_value",
    )
    lines = [f"- {key}: {options[key]}" for key in keys if options.get(key) is not None]
    return "\n".join(lines) or "No decision-specific business context was provided."


def _resolve_loss_preference(requested: str, llm_text: str | None) -> tuple[str, str]:
    """Resolve an explicit or LLM-recommended loss to a supported metric."""
    normalized = str(requested or "auto").strip().lower()
    if normalized in _SUPPORTED_LOSS_METRICS:
        return normalized, "user_selected"
    if normalized not in _AUTO_LOSS_VALUES:
        return "mase", "invalid_setting_fallback"
    if llm_text:
        match = re.search(
            r"recommended\s+decision\s+loss\s*:\s*(mase|wape|rmse|mae)\b",
            llm_text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).lower(), "llm_recommended"
    return "mase", "llm_unavailable_fallback"


def _loss_recommendation_rationale(
    resolved: str,
    source: str,
    llm_text: str | None,
) -> str:
    """Return a concise auditable rationale for the resolved loss."""
    if source == "user_selected":
        return "The user explicitly selected this decision-loss objective."
    if source == "llm_recommended" and llm_text:
        match = re.search(
            r"decision-loss\s+rationale\s*:\s*([^\r\n]+)",
            llm_text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()[:300]
        return f"The forecasting assistant recommended {resolved.upper()} from the supplied context."
    if source == "invalid_setting_fallback":
        return "The requested setting was unsupported, so MASE was used safely."
    return "The automatic recommendation was unavailable, so MASE was used safely."


def run_forecasting_agent_with_evidence(
    series: pd.Series,
    model_selection: ModelSelectionResult,
    stat_result: StatisticalResult,
    forecast_horizon: int,
    freq: str,
    existing_metrics: dict[str, dict[str, float]] | None = None,
    disabled_tests: list[str] | None = None,
    loss_preference: str = "auto",
    preprocessing_options: dict[str, Any] | None = None,
    exclude_models: list[str] | None = None,
    evaluation_evidence: ForecastEvaluationEvidence | None = None,
) -> tuple[
    ForecastResult,
    dict[str, dict[str, float]],
    ForecastEvaluationEvidence,
]:
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
    excluded_models = set(exclude_models or [])
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
    production_series = prepare_training_series(
        series,
        outlier_strategy=outlier_strategy,
        imputation_method=imputation_method,
        smoothing_method=smoothing_method,
    )
    results_store: dict[str, ForecastAdapterResult] = {}

    if evaluation_evidence is None:
        evaluation_started = perf_counter()
        backtest_evals = _run_backtest_evaluation(
            series,
            forecast_horizon,
            seasonal_period,
            apply_iqr_clip=False,
            imputation_method=imputation_method,
            smoothing_method=smoothing_method,
            outlier_strategy=outlier_strategy,
        )
        logger.info(
            "Performance comparison_evaluation wall_seconds=%.3f candidates=%d "
            "rss_mb=%.1f peak_rss_mb=%.1f",
            perf_counter() - evaluation_started,
            len(backtest_evals),
            memory_snapshot().current_rss_mb,
            memory_snapshot().peak_rss_mb,
        )
        comparison_summary = _build_backtest_comparison_summary(backtest_evals)
        (
            resolved_loss,
            loss_resolution_source,
            loss_rationale,
            reasoning_steps,
            token_usage,
        ) = _analyze_comparison_with_llm(
            model_selection,
            comparison_summary,
            loss_preference,
            preprocessing_options,
        )
    else:
        backtest_evals = evaluation_evidence.backtest_evaluations
        comparison_summary = evaluation_evidence.comparison_summary
        resolved_loss = evaluation_evidence.resolved_loss
        loss_resolution_source = evaluation_evidence.loss_resolution_source
        loss_rationale = evaluation_evidence.loss_rationale
        reasoning_steps = list(evaluation_evidence.reasoning_steps)
        token_usage = dict(evaluation_evidence.token_usage)
        logger.info(
            "Performance comparison_evaluation reused=true candidates=%d",
            len(backtest_evals),
        )

    # ── Select from common rolling-origin evidence ───────────────────────────
    selected = model_selection.selected_model
    sensitivity_winners: dict[str, str] = {}
    production_order: list[str] = []
    if model_selection.selection_method != "forced":
        rankable = {
            name: evaluation
            for name, evaluation in backtest_evals.items()
            if evaluation.is_rankable
        }
        if rankable:
            candidate_evidence = [
                CandidateEvidence(
                    name=name,
                    backtest=evaluation,
                    is_baseline=name in _BASELINE_NAMES,
                )
                for name, evaluation in rankable.items()
            ]
            outcomes = {
                metric: select_model_deterministic(
                    candidate_evidence,
                    exclude_models=list(excluded_models),
                    user_loss_preference=metric,
                )
                for metric in _SUPPORTED_LOSS_METRICS
            }
            sensitivity_winners = {
                metric: metric_outcome.selected_model
                for metric, metric_outcome in outcomes.items()
                if metric_outcome.selected_model
            }
            outcome = outcomes[resolved_loss]
            if outcome.selected_model:
                selected = outcome.selected_model
            production_order = [name for name, _ in outcome.ranking]
    if not production_order:
        production_order = [
            name
            for name, evaluation in sorted(
                backtest_evals.items(),
                key=lambda item: (
                    getattr(item[1].pooled_metrics, resolved_loss, None)
                    if getattr(item[1].pooled_metrics, resolved_loss, None) is not None
                    else float("inf")
                ),
            )
            if evaluation.is_rankable and name not in excluded_models
        ]
    production_order = [selected] + [
        name for name in production_order if name != selected
    ]

    res: ForecastAdapterResult | None = None
    for production_candidate in production_order:
        if production_candidate in excluded_models:
            continue
        production_started = perf_counter()
        try:
            candidate_result = _fit_production_candidate(
                production_candidate,
                production_series,
                forecast_horizon,
                seasonal_period,
                backtest_evals.get(production_candidate),
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Production refit failed for %s: %s", production_candidate, exc
            )
            candidate_result = ForecastAdapterResult(
                status=ForecastFitStatus.FAILED,
                failure_reason=str(exc),
                fitted_configuration={"model": production_candidate},
            )
        results_store[production_candidate] = candidate_result
        logger.info(
            "Performance production_fit model=%s wall_seconds=%.3f status=%s "
            "rss_mb=%.1f peak_rss_mb=%.1f",
            production_candidate,
            perf_counter() - production_started,
            candidate_result.status.value,
            memory_snapshot().current_rss_mb,
            memory_snapshot().peak_rss_mb,
        )
        if candidate_result.is_rankable:
            selected = production_candidate
            res = candidate_result
            break
        logger.warning(
            "Production candidate %s was not rankable; trying the next candidate.",
            production_candidate,
        )
    if res is None:
        raise RuntimeError("No forecasting model produced a valid production fit.")
    if selected != model_selection.selected_model:
        res = res.model_copy(update={"is_fallback": True})

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
    all_metrics = (
        {
            name: dict(metrics)
            for name, metrics in evaluation_evidence.all_metrics.items()
        }
        if evaluation_evidence is not None
        else _build_all_metrics(backtest_evals)
    )
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
    calibration_scores: list[float] = []
    selected_backtest = backtest_evals.get(selected)
    if selected_backtest is not None:
        for fold_result in selected_backtest.folds:
            if fold_result.status != ForecastFitStatus.OK:
                continue
            fold = fold_result.fold
            actual = series.iloc[fold.test_start_index : fold.test_end_index]
            calibration_scores.extend(
                interval_nonconformity_scores(
                    actual.to_numpy(dtype=float),
                    fold_result.lower_ci,
                    fold_result.upper_ci,
                )
            )
    if len(calibration_scores) >= 10 and lower_ci and upper_ci:
        lower_ci, upper_ci = calibrate_interval_width(
            lower_ci,
            upper_ci,
            calibration_scores=calibration_scores,
            nominal_coverage=(
                residual_diagnostics.nominal_coverage
                if residual_diagnostics is not None
                else 0.95
            ),
        )
        # Preserve the public label while using finite-sample conformal
        # calibration rather than the former aggregate-coverage heuristic.
        interval_label = "calibrated_prediction_interval"

    logger.info("Forecasting complete. Selected: %s", selected)

    selected_evaluation = backtest_evals.get(selected)
    validation_design = (
        dict(selected_evaluation.validation_design) if selected_evaluation else {}
    )
    distinct_winners = set(sensitivity_winners.values())
    validation_design["decision_loss"] = {
        "requested": loss_preference,
        "resolved": resolved_loss,
        "resolution_source": loss_resolution_source,
        "rationale": loss_rationale,
        "winners_by_metric": sensitivity_winners,
        "selection_sensitive": len(distinct_winners) > 1,
    }
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
        validation_design=validation_design,
        selection_metrics=reported_metrics.model_dump(
            include={"rmse", "mae", "mape", "wape", "mase", "smape", "rmsse"}
        ),
        final_test_metrics=(
            selected_evaluation.final_test_metrics.model_dump()
            if selected_evaluation is not None
            else {}
        ),
    )
    evidence = ForecastEvaluationEvidence(
        backtest_evaluations=backtest_evals,
        all_metrics=all_metrics,
        comparison_summary=comparison_summary,
        resolved_loss=resolved_loss,
        loss_resolution_source=loss_resolution_source,
        loss_rationale=loss_rationale,
        reasoning_steps=reasoning_steps,
        token_usage=token_usage,
    )
    return forecast_result, all_metrics, evidence


def run_forecasting_agent(
    series: pd.Series,
    model_selection: ModelSelectionResult,
    stat_result: StatisticalResult,
    forecast_horizon: int,
    freq: str,
    existing_metrics: dict[str, dict[str, float]] | None = None,
    disabled_tests: list[str] | None = None,
    loss_preference: str = "auto",
    preprocessing_options: dict[str, Any] | None = None,
    exclude_models: list[str] | None = None,
) -> tuple[ForecastResult, dict[str, dict[str, float]]]:
    """Compatibility wrapper returning the historical two-value result."""
    forecast, metrics, _ = run_forecasting_agent_with_evidence(
        series,
        model_selection,
        stat_result,
        forecast_horizon,
        freq,
        existing_metrics=existing_metrics,
        disabled_tests=disabled_tests,
        loss_preference=loss_preference,
        preprocessing_options=preprocessing_options,
        exclude_models=exclude_models,
    )
    return forecast, metrics


def _build_backtest_comparison_summary(
    evaluations: dict[str, BacktestEvaluation],
) -> str:
    """Format rolling-origin evidence without requiring production fits."""
    lines = ["Model comparison metrics (lower is better):"]
    for name, evaluation in evaluations.items():
        if not evaluation.is_rankable:
            reason = evaluation.unavailable_reasons.get(
                "all", "required metrics unavailable"
            )
            lines.append(f"- {name}: [status=not_estimable] {reason}")
            continue
        metrics = evaluation.pooled_metrics
        warning_text = (
            f" [warnings: {'; '.join(evaluation.warnings)}]"
            if evaluation.warnings
            else ""
        )
        lines.append(
            f"- {name}: [status=ok]{warning_text} "
            f"RMSE={_format_metric(metrics.rmse, '.4f')}, "
            f"MAE={_format_metric(metrics.mae, '.4f')}, "
            f"MAPE={_format_metric(metrics.mape, '.2f')}%, "
            f"WAPE={_format_metric(metrics.wape, '.2%')}, "
            f"MASE={_format_metric(metrics.mase, '.4f')} "
            f"(n_origins={evaluation.n_origins})"
        )
    return "\n".join(lines) + "\n"


def _analyze_comparison_with_llm(
    model_selection: ModelSelectionResult,
    comparison_summary: str,
    loss_preference: str,
    preprocessing_options: dict[str, Any],
) -> tuple[str, str, str, list[dict[str, Any]], dict[str, int]]:
    """Resolve the decision loss and generate optional comparison narration."""
    prompt = FORECASTING_PROMPT
    token_usage: dict[str, int] = {}
    loss_context = _business_context(preprocessing_options)
    resolved_loss, source = _resolve_loss_preference(loss_preference, None)
    rationale = _loss_recommendation_rationale(resolved_loss, source, None)
    try:
        inputs = {
            "selected": model_selection.selected_model,
            "summary": comparison_summary,
            "requested_loss": loss_preference,
            "business_context": loss_context,
        }
        response = (prompt | get_llm(temperature=0)).invoke(inputs)
        resolved_loss, source = _resolve_loss_preference(
            loss_preference, str(response.content)
        )
        rationale = _loss_recommendation_rationale(
            resolved_loss, source, str(response.content)
        )
        token_usage = extract_token_usage(
            response, input_text=estimate_input_text(prompt, inputs)
        )
        reasoning = [
            {
                "thought": "Evaluated forecasting candidates on common folds.",
                "observation": comparison_summary,
            },
            {
                "thought": "Analyzing metrics for performance comparison...",
                "observation": response.content,
            },
        ]
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Forecasting agent LLM call failed: %s", exc)
        reasoning = [
            {
                "thought": "LLM analysis failed, relying on direct Python metrics.",
                "observation": comparison_summary,
            }
        ]
    return resolved_loss, source, rationale, reasoning, token_usage


def _build_all_metrics(
    evaluations: dict[str, BacktestEvaluation],
) -> dict[str, dict[str, float]]:
    """Build the comparison-chart metrics from rolling-origin evidence."""
    all_metrics: dict[str, dict[str, float]] = {}
    for name, evaluation in evaluations.items():
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
    return all_metrics


def _reusable_configuration(
    evaluation: BacktestEvaluation | None,
) -> dict[str, object]:
    """Return final-test configuration or the newest successful fold config."""
    if evaluation is None:
        return {}
    if evaluation.final_test_fitted_configuration:
        return dict(evaluation.final_test_fitted_configuration)
    for fold in reversed(evaluation.folds):
        if fold.status == ForecastFitStatus.OK and fold.fitted_configuration:
            return dict(fold.fitted_configuration)
    return {}


def _fit_base_production(
    name: str,
    series: pd.Series,
    horizon: int,
    seasonal_period: int,
    evaluation: BacktestEvaluation | None,
) -> ForecastAdapterResult:
    """Fit one base model, preferring its leak-safe backtest configuration."""
    configuration = _reusable_configuration(evaluation)
    metrics = (
        evaluation.pooled_metrics if evaluation and evaluation.is_rankable else None
    )
    reused: ForecastAdapterResult | None = None
    if metrics is not None and configuration:
        if name == "ARIMA":
            reused = refit_arima_from_configuration(
                series, horizon, configuration, metrics
            )
        elif name == "SARIMA":
            reused = refit_sarima_from_configuration(
                series, horizon, seasonal_period, configuration, metrics
            )
        elif name == "Holt-Winters":
            reused = refit_holt_winters_from_configuration(
                series, horizon, seasonal_period, configuration, metrics
            )
        elif name == "EWMA":
            reused = refit_ewma_from_configuration(
                series, horizon, configuration, metrics
            )
        if reused is not None and reused.status != ForecastFitStatus.NOT_ESTIMABLE:
            return reused
        logger.warning(
            "Reusable configuration was incomplete for %s; using safe search fallback.",
            name,
        )

    if name == "ARIMA":
        return fit_arima(series, horizon, mase_period=seasonal_period)
    if name == "SARIMA":
        return fit_sarima(
            series,
            horizon,
            seasonal_period=seasonal_period,
            mase_period=seasonal_period,
        )
    if name == "Holt-Winters":
        return fit_holt_winters(
            series,
            horizon,
            seasonal_period=seasonal_period,
            mase_period=seasonal_period,
        )
    if name == "EWMA":
        return fit_ewma(series, horizon, mase_period=seasonal_period)
    raise ValueError(f"Unsupported production model: {name}")


def _fit_production_candidate(
    name: str,
    series: pd.Series,
    horizon: int,
    seasonal_period: int,
    evaluation: BacktestEvaluation | None,
) -> ForecastAdapterResult:
    """Fit exactly one selected base, transformed, or baseline candidate."""
    if name in _BASELINE_NAMES:
        return _fit_baseline_production(
            name, series, horizon, seasonal_period, evaluation
        )
    if " + " in name:
        return _fit_transformed_production(
            name, series, horizon, seasonal_period, evaluation
        )
    return _fit_base_production(name, series, horizon, seasonal_period, evaluation)


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
    result = _fit_base_production(
        base_name,
        transformed,
        horizon,
        mase_period,
        evaluation,
    )
    configuration = dict(result.fitted_configuration)
    configuration["preprocessing"] = transform.transform.model_dump()
    configuration["retransformation_bias"] = "residual_smearing"
    residuals = np.asarray(result.innovations, dtype=float)
    return result.model_copy(
        update={
            "forecast": bias_adjusted_inverse(
                transform, result.forecast, residuals
            ).tolist(),
            "lower_ci": (
                transform.inverse_transform(result.lower_ci).tolist()
                if result.lower_ci
                else []
            ),
            "upper_ci": (
                transform.inverse_transform(result.upper_ci).tolist()
                if result.upper_ci
                else []
            ),
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
        final_test_size=(
            forecast_horizon if len(series) >= 3 * forecast_horizon else 0
        ),
    )

    def _arima_fn(train: pd.Series, fold: BacktestFold) -> FoldPrediction | None:
        from forecasting.pmdarima_compat import fit_auto_arima_memory_aware  # local

        try:
            model = fit_auto_arima_memory_aware(
                train,
                seasonal_period=1,
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
        from forecasting.pmdarima_compat import fit_auto_arima_memory_aware  # local

        use_seasonal = len(train) >= 2 * seasonal_period
        try:
            model = fit_auto_arima_memory_aware(
                train,
                seasonal_period=seasonal_period if use_seasonal else 1,
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
