"""Forecasting agent that selects and runs statistical model implementations."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from core.llm_factory import get_llm
from core.logging_config import get_logger
from forecasting.backtesting import evaluate_candidates
from forecasting.contracts import (
    BacktestEvaluation,
    ForecastAdapterResult,
    ForecastFitStatus,
)
from forecasting.residual_diagnostics import (
    analyze_backtest_errors,
    analyze_innovations,
)
from forecasting.selection_policy import CandidateEvidence, select_model_deterministic
from forecasting.backtesting import evaluate_final_candidate
from forecasting.engine import BASELINES, ForecastEngine, backtest_config
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

_SUPPORTED_LOSS_METRICS = ("mase", "wape", "rmse", "mae", "pinball")
_AUTO_LOSS_VALUES = {"auto", "ai", "recommended", "let ai decide"}


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
    """Select a complete procedure, refit it, and audit its untouched test.

    All numerical evidence comes from identical rolling origins. Production
    failures trigger the same selection policy with the failed procedure
    excluded. Neither final-test scores nor legacy adapter holdouts rank models.
    """
    options = dict(preprocessing_options or {})
    if (
        model_selection.selection_method == "forced"
        and model_selection.selected_model == "Intermittent Demand"
    ):
        options["demand_pattern"] = "intermittent"
    if str(loss_preference).lower() == "pinball":
        options["point_quantile"] = options.get("forecast_quantile", 0.5)
    engine = ForecastEngine(freq, options)
    config = backtest_config(len(series), forecast_horizon, freq, options)
    excluded = set(exclude_models or [])
    # Excluding a base model also excludes procedures that use that model.
    excluded.update(
        name
        for name, procedure in engine.candidates.items()
        if procedure.base_name in excluded
        or any(m in excluded for m in procedure.members)
    )
    candidates = {
        name: proc for name, proc in engine.candidates.items() if name not in excluded
    }
    backtests = evaluate_candidates(series, candidates, config)
    all_metrics = {
        name: {
            metric.upper(): value
            for metric, value in evaluation.pooled_metrics.model_dump().items()
            if metric
            in {"rmse", "mae", "mape", "wape", "mase", "smape", "rmsse", "pinball"}
            and value is not None
        }
        for name, evaluation in backtests.items()
        if evaluation.is_rankable
    }
    comparison_summary = (
        "Common rolling-origin metrics (lower is better):\n"
        + "\n".join(f"- {name}: {metrics}" for name, metrics in all_metrics.items())
    )
    resolved_loss, loss_source = _resolve_loss_preference(loss_preference, None)
    loss_rationale = _loss_recommendation_rationale(resolved_loss, loss_source, None)
    token_usage: dict[str, int] = {}
    reasoning_steps = [
        {
            "thought": "Compared enabled forecasting procedures on common origins.",
            "observation": comparison_summary,
        }
    ]
    # The model may interpret context, but receives no final-test observations.
    try:
        llm = get_llm(temperature=0)
        inputs = {
            "selected": "Pending common rolling-origin selection",
            "summary": comparison_summary,
            "requested_loss": loss_preference,
            "business_context": _business_context(options),
        }
        response = (FORECASTING_PROMPT | llm).invoke(inputs)
        resolved_loss, loss_source = _resolve_loss_preference(
            loss_preference, str(response.content)
        )
        loss_rationale = _loss_recommendation_rationale(
            resolved_loss, loss_source, str(response.content)
        )
        token_usage = extract_token_usage(
            response, input_text=estimate_input_text(FORECASTING_PROMPT, inputs)
        )
        reasoning_steps.append(
            {
                "thought": "Interpreted the comparison and business objective.",
                "observation": str(response.content),
            }
        )
    except Exception as exc:
        logger.warning("Forecast explanation unavailable: %s", exc)

    if not any(
        item.is_rankable
        and getattr(item.pooled_metrics, resolved_loss, None) is not None
        for item in backtests.values()
    ):
        unavailable_loss = resolved_loss
        resolved_loss = "mae"
        loss_source = "unavailable_metric_fallback"
        loss_rationale = (
            f"{unavailable_loss.upper()} was not estimable on the common validation "
            "data for any eligible candidate, so selection used MAE."
        )

    evidence = [
        CandidateEvidence(name=name, backtest=value, is_baseline=name in BASELINES)
        for name, value in backtests.items()
    ]
    forced = model_selection.selection_method == "forced"
    results: dict[str, ForecastAdapterResult] = {}
    failed: list[str] = []
    while True:
        if forced:
            selected = model_selection.selected_model
            if selected not in candidates:
                raise ValueError(
                    f"Requested model '{selected}' is disabled, excluded, or unknown."
                )
        else:
            outcome = select_model_deterministic(
                evidence,
                exclude_models=list(excluded),
                user_loss_preference=resolved_loss,
            )
            selected = outcome.selected_model
            if not selected:
                raise RuntimeError(
                    "No forecasting procedure passed all common validation folds (minimum two origins)."
                )
        try:
            result = candidates[selected].fit(series, forecast_horizon)
            if (
                result.status != ForecastFitStatus.OK
                or not np.isfinite(result.forecast).all()
            ):
                raise ValueError(
                    result.failure_reason or "Production forecast is invalid."
                )
            result.metrics = backtests[selected].pooled_metrics
            result.is_fallback = bool(failed)
            results[selected] = result
            break
        except Exception as exc:
            results[selected] = ForecastAdapterResult(
                status=ForecastFitStatus.FAILED, failure_reason=str(exc)
            )
            if forced:
                raise RuntimeError(
                    f"Requested model '{selected}' failed: {exc}"
                ) from exc
            failed.append(selected)
            excluded.add(selected)
            logger.warning(
                "Production fit failed for %s; reranking common evidence: %s",
                selected,
                exc,
            )

    # Freeze the winning procedure before consulting any final-test targets.
    evaluation = backtests[selected]
    final_metrics, final_fold = evaluate_final_candidate(
        selected, series, candidates[selected], config
    )
    evaluation.final_test_metrics = final_metrics
    final_interval_diagnostics = {}
    if final_fold is not None and final_fold.status == ForecastFitStatus.OK:
        final_actual = series.iloc[final_fold.fold.test_start_index :].tolist()
        final_interval_diagnostics = analyze_backtest_errors(
            [final_fold.residuals],
            fold_actuals=[final_actual],
            fold_lower=[final_fold.lower_ci],
            fold_upper=[final_fold.upper_ci],
        ).model_dump(
            include={
                "interval_coverage",
                "interval_mean_width",
                "winkler_score",
                "n_errors",
            }
        )

    sensitivity = {
        metric: select_model_deterministic(
            evidence, exclude_models=list(excluded), user_loss_preference=metric
        ).selected_model
        for metric in _SUPPORTED_LOSS_METRICS
    }
    design = dict(evaluation.validation_design)
    monitoring_baselines = {}
    for name in ("Naive", "Seasonal Naive"):
        try:
            monitoring_baselines[name] = (
                engine.candidates[name].fit(series, forecast_horizon).forecast
            )
        except ValueError:
            pass
    design.update(
        {
            "monitoring_baselines": monitoring_baselines,
            "decision_loss": {
                "requested": loss_preference,
                "resolved": resolved_loss,
                "quantile": options.get("point_quantile"),
                "resolution_source": loss_source,
                "rationale": loss_rationale,
                "winners_by_metric": sensitivity,
                "selection_sensitive": len(set(sensitivity.values())) > 1,
            },
            "production_failures": failed,
            "excluded_models": sorted(excluded),
            "final_test_used_for_selection": False,
            "final_test_interval_diagnostics": final_interval_diagnostics,
            "interval_calibration": "none; model-based intervals audited out of sample",
            "by_horizon_metrics": {
                str(h): m.model_dump() for h, m in evaluation.by_horizon_metrics.items()
            },
            "preprocessing": engine.preprocessing,
        }
    )
    if options.get("missing_strategy") == "drop":
        result.warnings.append(
            "Missing timestamps retained; training gaps imputed and missing actuals excluded from scores."
        )
    if not evaluation.is_rankable:
        result.status = ForecastFitStatus.DEGRADED
        result.warnings.append(
            "Forced forecast lacks complete common validation evidence."
        )
    if config.horizon < forecast_horizon:
        result.warnings.append(
            "Requested horizons beyond the evaluated horizon have no validation evidence."
        )
    residual_diagnostics = _run_residual_diagnostics(
        result, evaluation, series, disabled_tests
    )
    if isinstance(series.index, pd.DatetimeIndex):
        dates = pd.date_range(
            start=series.index[-1], periods=forecast_horizon + 1, freq=freq
        )[1:]
        # Preserve time-of-day for subdaily series.
        date_strings = [date.isoformat() for date in dates]
    else:
        date_strings = [str(i + 1) for i in range(forecast_horizon)]
    candidate_results = []
    for name, item in backtests.items():
        production = results.get(name)
        candidate_results.append(
            ForecastCandidateResult(
                model=name,
                status=(
                    production.status
                    if production
                    else (
                        ForecastFitStatus.OK
                        if item.is_rankable
                        else ForecastFitStatus.NOT_ESTIMABLE
                    )
                ),
                failure_reason=production.failure_reason if production else None,
                is_fallback=production.is_fallback if production else False,
                **item.pooled_metrics.model_dump(
                    include={
                        "rmse",
                        "mae",
                        "mape",
                        "wape",
                        "mase",
                        "smape",
                        "rmsse",
                        "pinball",
                        "n_missing",
                    }
                ),
                n_evaluated=item.n_evaluated,
                fitted_configuration=(
                    production.fitted_configuration if production else {}
                ),
                warnings=[*item.warnings, *(production.warnings if production else [])],
                interval_label=(
                    production.interval_label if production else "backtest_only"
                ),
                validation_design=item.validation_design,
                metric_intervals=item.metric_intervals,
                skill_scores=item.skill_scores,
                final_test_metrics=item.final_test_metrics.model_dump(),
            )
        )
    return (
        ForecastResult(
            model_used=selected,
            status=result.status,
            failure_reason=result.failure_reason,
            is_fallback=result.is_fallback,
            forecast=result.forecast,
            lower_ci=result.lower_ci,
            upper_ci=result.upper_ci,
            forecast_dates=date_strings,
            **result.metrics.model_dump(
                include={
                    "rmse",
                    "mae",
                    "mape",
                    "wape",
                    "mase",
                    "smape",
                    "rmsse",
                    "pinball",
                }
            ),
            residual_diagnostics=residual_diagnostics,
            candidate_results=candidate_results,
            reasoning_steps=reasoning_steps,
            token_usage=token_usage,
            interval_label=result.interval_label,
            validation_design=design,
            selection_metrics=result.metrics.model_dump(
                include={
                    "rmse",
                    "mae",
                    "mape",
                    "wape",
                    "mase",
                    "smape",
                    "rmsse",
                    "pinball",
                }
            ),
            final_test_metrics=final_metrics.model_dump(),
        ),
        all_metrics,
    )


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

    innovation_diagnostics = {}
    if result.innovations and diag.error_type == "backtest_errors":
        innovation = analyze_innovations(
            np.asarray(result.innovations),
            ar_ma_order=int(result.fitted_configuration.get("ar_ma_order", 0)),
            disabled_tests=disabled_tests or [],
        )
        innovation_diagnostics = innovation.model_dump()
        for attribute in (
            "ljung_box_p_value",
            "ljung_box_lag",
            "ljung_box_df_adjust",
            "is_uncorrelated",
            "shapiro_p_value",
            "is_normal",
        ):
            setattr(diag, attribute, getattr(innovation, attribute))
        diag.error_type = "backtest_errors_with_innovation_checks"
        diag.warnings.append(
            "Bias and coverage use backtests; autocorrelation and normality use fitted one-step innovations."
        )
    return ResidualDiagnostics(
        innovation_diagnostics=innovation_diagnostics,
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
