"""Forecasting agent that selects and runs statistical model implementations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.llm_factory import get_llm
from core.logging_config import get_logger
from forecasting.arima_model import fit_arima
from forecasting.backtesting import BacktestConfig, evaluate_candidates
from forecasting.contracts import ForecastAdapterResult, ForecastFitStatus
from forecasting.ewma_model import fit_ewma
from forecasting.holt_winters import fit_holt_winters
from forecasting.residual_diagnostics import analyze_innovations
from forecasting.sarima_model import fit_sarima
from prompts.forecasting_prompt import FORECASTING_PROMPT
from schemas import (
    ForecastCandidateResult,
    ForecastResult,
    ModelSelectionResult,
    ResidualDiagnostics,
    StatisticalResult,
)
from utils.statistical_analysis import analyze_residuals
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
    seasonal_period = stat_result.seasonal_period or 12
    results_store: dict[str, ForecastAdapterResult] = {}

    # ── Fit all models directly in Python ─────────────────────────────────────
    for name, fn, kwargs in [
        ("Holt-Winters", fit_holt_winters, {"mase_period": seasonal_period}),
        ("ARIMA", fit_arima, {"mase_period": seasonal_period}),
        (
            "SARIMA",
            fit_sarima,
            {"seasonal_period": seasonal_period, "mase_period": seasonal_period},
        ),
        ("EWMA", fit_ewma, {"mase_period": seasonal_period}),
    ]:
        try:
            results_store[name] = fn(series, forecast_horizon, **kwargs)
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
    backtest_evals = _run_backtest_evaluation(series, forecast_horizon, seasonal_period)

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

    # ── Select result for the chosen model ───────────────────────────────────
    selected = model_selection.selected_model
    if selected not in results_store:
        # Try to fit the selected model directly
        try:
            if selected == "Holt-Winters":
                results_store[selected] = fit_holt_winters(
                    series, forecast_horizon, mase_period=seasonal_period
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
    for name, r in results_store.items():
        if not _has_required_metrics(r):
            continue
        all_metrics[name] = {
            "RMSE": r.metrics.rmse if r.metrics.rmse is not None else float("nan"),
            "MAE": r.metrics.mae if r.metrics.mae is not None else float("nan"),
            "MAPE": r.metrics.mape if r.metrics.mape is not None else float("nan"),
            "WAPE": r.metrics.wape if r.metrics.wape is not None else float("nan"),
            "MASE": r.metrics.mase if r.metrics.mase is not None else float("nan"),
        }
    # Merge any pre-existing metrics (e.g. baselines) passed in by the caller
    # so re-runs preserve previously computed results.
    if existing_metrics is not None:
        for name, metrics in existing_metrics.items():
            all_metrics.setdefault(name, metrics)

    # ── Residual Analysis ───────────────────────────────────────────────────
    residual_diagnostics = _run_residual_diagnostics(res, disabled_tests)

    logger.info("Forecasting complete. Selected: %s", selected)

    forecast_result = ForecastResult(
        model_used=selected,
        status=res.status,
        failure_reason=res.failure_reason,
        is_fallback=res.is_fallback,
        forecast=res.forecast,
        lower_ci=res.lower_ci,
        upper_ci=res.upper_ci,
        forecast_dates=forecast_dates,
        rmse=res.metrics.rmse,
        mae=res.metrics.mae,
        mape=res.metrics.mape,
        wape=res.metrics.wape,
        mase=res.metrics.mase,
        residual_diagnostics=residual_diagnostics,
        candidate_results=[
            ForecastCandidateResult(
                model=name,
                status=candidate.status,
                failure_reason=candidate.failure_reason,
                is_fallback=candidate.is_fallback,
                rmse=candidate.metrics.rmse,
                mae=candidate.metrics.mae,
                mape=candidate.metrics.mape,
                wape=candidate.metrics.wape,
                mase=candidate.metrics.mase,
                n_evaluated=candidate.metrics.n_evaluated,
                n_missing=candidate.metrics.n_missing,
                fitted_configuration=candidate.fitted_configuration,
                warnings=candidate.warnings,
                interval_label=candidate.interval_label,
            )
            for name, candidate in results_store.items()
        ],
        reasoning_steps=reasoning_steps,
        token_usage=token_usage,
        interval_label=res.interval_label,
    )
    return forecast_result, all_metrics


def _run_backtest_evaluation(
    series: pd.Series,
    forecast_horizon: int,
    seasonal_period: int,
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
        max_origins=5,
        mase_period=seasonal_period,
    )

    def _arima_fn(train: pd.Series, fold: BacktestFold) -> FoldPrediction | None:
        from forecasting.pmdarima_compat import import_pmdarima  # local

        pm = import_pmdarima()
        try:
            model = pm.auto_arima(
                train,
                seasonal=False,
                stepwise=True,
                max_p=3,
                max_q=3,
                error_action="ignore",
                suppress_warnings=True,
                information_criterion="aic",
            )
            preds, _ = model.predict(n_periods=fold.horizon, return_conf_int=True)
            return FoldPrediction(predictions=np.asarray(preds, dtype=float))
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
                max_p=2,
                max_q=2,
                max_P=1,
                max_Q=1,
                error_action="ignore",
                suppress_warnings=True,
                information_criterion="aic",
            )
            preds, _ = model.predict(n_periods=fold.horizon, return_conf_int=True)
            return FoldPrediction(predictions=np.asarray(preds, dtype=float))
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Backtest SARIMA fold %d failed: %s", fold.fold_index, exc)
            return None

    def _hw_fn(train: pd.Series, fold: BacktestFold) -> FoldPrediction | None:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # local

        use_seasonal = len(train) >= 2 * seasonal_period
        try:
            fit = ExponentialSmoothing(
                train,
                trend="add",
                seasonal="add" if use_seasonal else None,
                seasonal_periods=seasonal_period if use_seasonal else None,
            ).fit(optimized=True)
            preds = fit.forecast(fold.horizon)
            return FoldPrediction(predictions=np.asarray(preds, dtype=float))
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Backtest Holt-Winters fold %d failed: %s", fold.fold_index, exc
            )
            return None

    def _ewma_fn(train: pd.Series, fold: BacktestFold) -> FoldPrediction | None:
        try:
            level = float(train.ewm(alpha=0.3, adjust=False).mean().iloc[-1])
            preds = np.full(fold.horizon, level, dtype=float)
            return FoldPrediction(predictions=preds)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Backtest EWMA fold %d failed: %s", fold.fold_index, exc)
            return None

    candidates = {
        "ARIMA": _arima_fn,
        "SARIMA": _sarima_fn,
        "Holt-Winters": _hw_fn,
        "EWMA": _ewma_fn,
    }
    try:
        return evaluate_candidates(series, candidates, config=config)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Backtest evaluation failed: %s", exc)
        return {}


def _run_residual_diagnostics(
    result: ForecastAdapterResult,
    disabled_tests: list[str] | None,
) -> ResidualDiagnostics | None:
    """Run residual diagnostics on the selected model's innovations.

    Args:
        result:         The selected model's typed adapter result.
        disabled_tests: Residual diagnostic tests to skip.

    Returns:
        A :class:`ResidualDiagnostics` schema, or ``None`` when no
        innovations are available.
    """
    if not result.innovations:
        return None

    ar_ma_order = int(result.fitted_configuration.get("ar_ma_order", 0))
    try:
        diag = analyze_innovations(
            np.asarray(result.innovations, dtype=float),
            ar_ma_order=ar_ma_order,
            disabled_tests=disabled_tests or [],
        )
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
        nominal_coverage=diag.nominal_coverage,
        coverage_estimable=diag.coverage_estimable,
        warnings=diag.warnings,
    )
