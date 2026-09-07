"""Regression tests for review-triggered model-selection retries."""

from __future__ import annotations

from typing import Any

import pandas as pd

from agents.model_selection_agent import (
    _format_metrics_text,
    build_model_rejection_reasons,
    run_model_selection_agent,
)
from agents.statistical_review_agent import (
    _check_residual_autocorrelation,
    _compute_override_eligibility,
    _merge_review_flags,
)
from forecasting.contracts import ForecastFitStatus
from schemas import (
    ForecastResult,
    ModelSelectionResult,
    ResidualDiagnostics,
    StatisticalResult,
    StatisticalReviewResult,
)
from services import pipeline_service


def test_review_flags_semantically_deduplicate_residual_autocorrelation() -> None:
    """An LLM paraphrase must not duplicate the deterministic residual warning."""
    deterministic = [
        {
            "agent": "forecasting",
            "severity": "warning",
            "issue": "Model residuals are autocorrelated (Ljung-Box p-value=0.001).",
            "recommendation": "Monitor the residual dependence.",
        }
    ]
    llm_flags = [
        {
            "agent": "forecasting",
            "severity": "warning",
            "issue": "Residual autocorrelation remains statistically significant.",
            "recommendation": "Review the residual pattern.",
        }
    ]

    assert _merge_review_flags(deterministic, llm_flags) == deterministic


def _statistical_result() -> StatisticalResult:
    return StatisticalResult(
        is_stationary_adf=False,
        adf_statistic=0.0,
        adf_p_value=0.99,
        is_stationary_kpss=False,
        kpss_statistic=1.0,
        kpss_p_value=0.01,
        has_trend=True,
        trend_slope=2.0,
        seasonal_period=12,
        summary="Seasonal trend.",
    )


def _forecast(model: str) -> ForecastResult:
    return ForecastResult(
        model_used=model,
        status=ForecastFitStatus.OK,
        forecast=[10.0],
        lower_ci=[9.0],
        upper_ci=[11.0],
        forecast_dates=["2025-01-01"],
        rmse=1.0,
        mae=0.8,
        validation_design={
            "decision_loss": {"resolved": "mase", "selection_sensitive": False}
        },
    )


def test_metric_text_does_not_scale_mape_twice() -> None:
    """MAPE is already percentage points while WAPE is stored as a ratio."""
    text = _format_metrics_text(
        {"Holt-Winters": {"MAPE": 3.6606, "WAPE": 0.0365}}
    )

    assert "MAPE=3.66%" in text
    assert "WAPE=3.65%" in text
    assert "366.06%" not in text


def test_final_rejection_reasons_never_reject_selected_model() -> None:
    """Reasons must be rebuilt after final deterministic model selection."""
    reasons = build_model_rejection_reasons(
        "Holt-Winters",
        _statistical_result(),
        {
            "Holt-Winters": {"MASE": 0.54},
            "SARIMA": {"MASE": 0.80},
        },
    )

    assert reasons["Holt-Winters"] is None
    assert reasons["SARIMA"] is not None
    assert "Higher forecast error" in reasons["SARIMA"]


def test_residual_autocorrelation_warns_without_forcing_override() -> None:
    """One model's residual warning cannot prove an alternative is better."""
    forecast = _forecast("Holt-Winters").model_copy(
        update={
            "residual_diagnostics": ResidualDiagnostics(
                mean=0.0,
                is_uncorrelated=False,
                ljung_box_p_value=0.001,
            )
        }
    )

    flag = _check_residual_autocorrelation(forecast)

    assert flag is not None
    assert flag["severity"] == "warning"
    assert "compare residual diagnostics" in flag["recommendation"]
    can_override, reasons = _compute_override_eligibility(
        ModelSelectionResult(
            selected_model="Holt-Winters",
            explanation="Selected model: Holt-Winters.",
            selection_method="deterministic",
        ),
        [flag],
    )
    assert can_override is False
    assert reasons == []


def test_review_retry_describes_best_eligible_model_and_exclusion_once() -> None:
    """Retry rationale must distinguish exclusion from inferior performance."""
    result = run_model_selection_agent(
        _statistical_result(),
        review_feedback="Typed review issue.",
        exclude_model="Holt-Winters",
        all_metrics={
            "Holt-Winters": {"MASE": 0.54, "RMSE": 16.39, "MAE": 12.33},
            "SARIMA": {"MASE": 0.80, "RMSE": 22.21, "MAE": 18.20},
        },
    )

    assert result.selected_model == "SARIMA"
    assert "eligible empirical validation metrics" in result.explanation
    assert result.explanation.count("[Statistical Review Feedback]") == 1
    assert result.holt_winters_rejected_reason is not None
    assert "Excluded following statistical review" in result.holt_winters_rejected_reason


def test_retry_preserves_exclusion_and_synchronizes_final_model(
    monkeypatch: Any,
) -> None:
    """A review retry must not display one model while forecasting with another."""
    initial_selection = ModelSelectionResult(
        selected_model="Holt-Winters",
        explanation="Selected model: Holt-Winters.",
        selection_method="deterministic",
    )
    initial_review = StatisticalReviewResult(
        verdict="fail",
        flags=[{"severity": "critical", "agent": "model_selection"}],
        summary="Review found a typed consistency violation.",
        can_override_selection=True,
    )
    retry_selection = ModelSelectionResult(
        selected_model="SARIMA",
        explanation=(
            "Selected model: SARIMA.\n"
            "[Statistical Review Feedback]: Review found a typed consistency violation."
        ),
        selection_method="deterministic",
    )
    captured: dict[str, Any] = {}

    def fake_select(*args: Any, **kwargs: Any) -> ModelSelectionResult:
        assert kwargs["exclude_model"] == "Holt-Winters"
        assert kwargs["loss_preference"] == "mase"
        return retry_selection

    def fake_forecast(*args: Any, **kwargs: Any) -> tuple[ForecastResult, dict[str, dict[str, float]]]:
        captured["exclude_models"] = kwargs["exclude_models"]
        return _forecast("SARIMA"), {"SARIMA": {"MASE": 0.8}}

    monkeypatch.setattr(pipeline_service, "run_model_selection_agent", fake_select)
    monkeypatch.setattr(pipeline_service, "run_forecasting_agent", fake_forecast)
    monkeypatch.setattr(
        pipeline_service,
        "run_statistical_review_agent",
        lambda *args, **kwargs: StatisticalReviewResult(
            verdict="pass", summary="Retry is consistent."
        ),
    )

    output = pipeline_service._maybe_retry_forecast_after_review(
        pd.Series([1.0, 2.0, 3.0]),
        _statistical_result(),
        initial_selection,
        _forecast("Holt-Winters"),
        initial_review,
        {"Holt-Winters": {"MASE": 0.5}, "SARIMA": {"MASE": 0.8}},
        1,
        "MS",
        [],
        None,
        {},
        lambda *_: None,
    )

    assert captured["exclude_models"] == ["Holt-Winters"]
    assert output.model_selection.selected_model == output.forecast.model_used
    assert output.model_selection.selected_model == "SARIMA"
    assert output.model_selection.explanation.count(
        "[Statistical Review Feedback]"
    ) == 1
