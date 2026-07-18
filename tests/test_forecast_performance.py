"""Regression tests for resource-efficient forecast execution."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from agents import forecasting_agent
from forecasting import arima_model
from forecasting.backtesting import BacktestConfig, FoldPrediction, evaluate_candidates
from forecasting.contracts import (
    BacktestEvaluation,
    ForecastAdapterResult,
    ForecastFitStatus,
    ForecastMetrics,
)
from schemas import (
    ForecastResult,
    ModelSelectionResult,
    StatisticalResult,
    StatisticalReviewResult,
)
from services import pipeline_service


def _metrics(rmse: float, mase: float) -> ForecastMetrics:
    return ForecastMetrics(
        rmse=rmse,
        mae=rmse * 0.8,
        mape=5.0,
        wape=0.05,
        mase=mase,
        smape=5.0,
        rmsse=mase,
        n_evaluated=10,
    )


def _evaluation(name: str, rmse: float, mase: float) -> BacktestEvaluation:
    return BacktestEvaluation(
        model_name=name,
        pooled_metrics=_metrics(rmse, mase),
        final_test_metrics=_metrics(rmse, mase),
        final_test_fitted_configuration={
            "order": [1, 1, 0],
            "seasonal_order": [0, 0, 0, 0],
            "with_intercept": True,
        },
        n_origins=2,
        n_evaluated=10,
    )


def _statistical_result() -> StatisticalResult:
    return StatisticalResult(
        is_stationary_adf=False,
        adf_statistic=0.0,
        adf_p_value=0.9,
        is_stationary_kpss=False,
        kpss_statistic=1.0,
        kpss_p_value=0.01,
        has_trend=True,
        trend_slope=1.0,
        seasonal_period=1,
        summary="Trend series.",
    )


def _production_result(metrics: ForecastMetrics) -> ForecastAdapterResult:
    return ForecastAdapterResult(
        status=ForecastFitStatus.OK,
        forecast=[10.0, 11.0],
        lower_ci=[9.0, 10.0],
        upper_ci=[11.0, 12.0],
        metrics=metrics,
        fitted_configuration={"model": "ARIMA", "order": [1, 1, 0]},
    )


def test_final_test_configuration_is_retained() -> None:
    """The untouched final fit must expose reusable production parameters."""
    series = pd.Series(np.arange(20, dtype=float))

    def candidate(train: pd.Series, fold: Any) -> FoldPrediction:
        return FoldPrediction(
            predictions=np.repeat(float(train.iloc[-1]), fold.horizon),
            fitted_configuration={"training_size": len(train)},
        )

    result = evaluate_candidates(
        series,
        {"Candidate": candidate},
        BacktestConfig(horizon=2, max_origins=2, final_test_size=2),
    )["Candidate"]

    assert result.final_test_fitted_configuration == {"training_size": 18}


def test_arima_reuses_backtest_configuration(monkeypatch: Any) -> None:
    """A reusable ARIMA order must bypass order discovery."""
    captured: dict[str, Any] = {}

    def refit(**kwargs: Any) -> ForecastAdapterResult:
        captured.update(kwargs)
        return _production_result(kwargs["metrics"])

    monkeypatch.setattr(arima_model, "_refit_full_series_arima", refit)
    metrics = _metrics(1.0, 0.5)
    result = arima_model.refit_arima_from_configuration(
        pd.Series(np.arange(20, dtype=float)),
        2,
        {"order": [2, 1, 0], "with_intercept": False},
        metrics,
    )

    assert result.status == ForecastFitStatus.OK
    assert captured["order"] == (2, 1, 0)
    assert captured["with_intercept"] is False
    assert captured["train_model"] is True


def test_forecasting_fits_only_selected_production_candidate(
    monkeypatch: Any,
) -> None:
    """Comparison candidates must not receive eager production fits."""
    evaluations = {
        "ARIMA": _evaluation("ARIMA", 1.0, 0.5),
        "SARIMA": _evaluation("SARIMA", 2.0, 1.0),
    }
    fitted: list[str] = []
    monkeypatch.setattr(
        forecasting_agent,
        "_run_backtest_evaluation",
        lambda *args, **kwargs: evaluations,
    )
    monkeypatch.setattr(
        forecasting_agent,
        "_analyze_comparison_with_llm",
        lambda *args, **kwargs: (
            "mase",
            "user_selected",
            "User selected MASE.",
            [],
            {},
        ),
    )
    monkeypatch.setattr(
        forecasting_agent,
        "_fit_production_candidate",
        lambda name, *args, **kwargs: (
            fitted.append(name) or _production_result(evaluations[name].pooled_metrics)
        ),
    )
    monkeypatch.setattr(
        forecasting_agent,
        "_run_residual_diagnostics",
        lambda *args, **kwargs: None,
    )

    series = pd.Series(
        np.arange(24, dtype=float),
        index=pd.date_range("2024-01-01", periods=24, freq="D"),
    )
    forecast, _, evidence = forecasting_agent.run_forecasting_agent_with_evidence(
        series,
        ModelSelectionResult(
            selected_model="ARIMA",
            explanation="Initial selection.",
        ),
        _statistical_result(),
        2,
        "D",
        loss_preference="mase",
    )

    assert forecast.model_used == "ARIMA"
    assert fitted == ["ARIMA"]
    assert evidence.backtest_evaluations is evaluations


def test_review_retry_reuses_forecast_evidence(monkeypatch: Any) -> None:
    """An authorized retry must use retained evidence instead of backtesting."""
    evaluations = {
        "ARIMA": _evaluation("ARIMA", 1.0, 0.5),
        "SARIMA": _evaluation("SARIMA", 2.0, 1.0),
    }
    evidence = forecasting_agent.ForecastEvaluationEvidence(
        backtest_evaluations=evaluations,
        all_metrics={"ARIMA": {"MASE": 0.5}, "SARIMA": {"MASE": 1.0}},
        comparison_summary="cached",
        resolved_loss="mase",
        loss_resolution_source="user_selected",
        loss_rationale="User selected MASE.",
        reasoning_steps=[],
        token_usage={},
    )
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        pipeline_service,
        "run_model_selection_agent",
        lambda *args, **kwargs: ModelSelectionResult(
            selected_model="SARIMA",
            explanation="Retry selection.",
            selection_method="deterministic",
        ),
    )

    def reused_forecast(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        captured["evidence"] = kwargs["evaluation_evidence"]
        captured["exclusions"] = kwargs["exclude_models"]
        return (
            ForecastResult(
                model_used="SARIMA",
                status=ForecastFitStatus.OK,
                forecast=[10.0],
                lower_ci=[9.0],
                upper_ci=[11.0],
                forecast_dates=["2025-01-01"],
                rmse=2.0,
                mae=1.6,
                validation_design={"decision_loss": {"resolved": "mase"}},
            ),
            evidence.all_metrics,
            evidence,
        )

    monkeypatch.setattr(
        pipeline_service,
        "run_forecasting_agent_with_evidence",
        reused_forecast,
    )
    monkeypatch.setattr(
        pipeline_service,
        "run_forecasting_agent",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("full comparison rerun was used")
        ),
    )
    monkeypatch.setattr(
        pipeline_service,
        "run_statistical_review_agent",
        lambda *args, **kwargs: StatisticalReviewResult(
            verdict="pass", summary="Retry passed."
        ),
    )

    output = pipeline_service._maybe_retry_forecast_after_review(
        pd.Series([1.0, 2.0, 3.0]),
        _statistical_result(),
        ModelSelectionResult(
            selected_model="ARIMA",
            explanation="Initial selection.",
            selection_method="deterministic",
        ),
        ForecastResult(
            model_used="ARIMA",
            status=ForecastFitStatus.OK,
            forecast=[10.0],
            lower_ci=[9.0],
            upper_ci=[11.0],
            forecast_dates=["2025-01-01"],
            rmse=1.0,
            mae=0.8,
            validation_design={"decision_loss": {"resolved": "mase"}},
        ),
        StatisticalReviewResult(
            verdict="fail",
            flags=[{"severity": "critical", "agent": "model_selection"}],
            summary="Typed override.",
            can_override_selection=True,
        ),
        evidence.all_metrics,
        1,
        "D",
        [],
        None,
        {},
        lambda *_: None,
        evaluation_evidence=evidence,
    )

    assert captured["evidence"] is evidence
    assert captured["exclusions"] == ["ARIMA"]
    assert output.evaluation_evidence is evidence
