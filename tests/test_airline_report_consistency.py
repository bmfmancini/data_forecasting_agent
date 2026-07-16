"""Golden evidence-consistency checks for the airline-passenger workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agents.report_generation_agent import _compute_visual_strategy
from forecasting.contracts import ForecastFitStatus
from forecasting.diagnostics import assess_stationarity
from forecasting.selection_policy import _check_invented_metrics
from prompts.forecasting_prompt import FORECASTING_PROMPT
from prompts.report_generation_prompt import (
    DATA_QUALITY_NARRATIVE_PROMPT,
    RECOMMENDATION_NARRATIVE_PROMPT,
)
from prompts.statistical_analysis_prompt import STATISTICAL_ANALYSIS_PROMPT
from prompts.statistical_review_prompt import STATISTICAL_REVIEW_PROMPT
from report.builder import ExecutiveReportBuilder
from report.dashboard import recommended_action
from report.models import ConfidenceAssessment, DataQualitySection
from report.narrative import (
    _contradictory_data_quality_rating,
    _fallback_forecast_outlook,
    _fallback_narrative,
    _unsupported_anomaly_significance_claim,
    _unsupported_interval_calibration_claim,
    _unsupported_recommendation_claims,
)
from report.rules import (
    CONFIDENCE_DEDUCTIONS,
    RECENT_HOLDOUT_RMSE_RATIO_THRESHOLD,
)
from schemas import (
    ForecastResult,
    ModelSelectionResult,
    ResidualDiagnostics,
    StatisticalReviewResult,
    StatisticalResult,
    ValidationResult,
)
from services.pipeline_service import _apply_agent_remediation
from utils.data_cleaning import time_index_quality
from utils.preflight import run_preflight_checks
from utils.visualization import plot_forecast


def _airline_frame() -> pd.DataFrame:
    frame = pd.read_csv("data_forecaster/data/sample_airline_passengers.csv")
    frame["Month"] = pd.to_datetime(frame["Month"])
    return frame


def _statistical() -> StatisticalResult:
    return StatisticalResult(
        is_stationary_adf=False,
        adf_statistic=0.8154,
        adf_p_value=0.9919,
        is_stationary_kpss=False,
        kpss_statistic=1.6513,
        kpss_p_value=0.01,
        has_trend=True,
        trend_slope=2.657,
        seasonal_period=12,
        summary="Seasonal trend.",
    )


def _forecast() -> ForecastResult:
    return ForecastResult(
        model_used="Holt-Winters",
        status=ForecastFitStatus.OK,
        forecast=[441.1, 432.3],
        lower_ci=[421.0, 393.5],
        upper_ci=[461.7, 542.0],
        forecast_dates=["1961-01-01", "1961-12-01"],
        rmse=16.3929,
        mae=12.3311,
        mape=3.66,
        residual_diagnostics=ResidualDiagnostics(
            mean=0.0,
            is_uncorrelated=False,
            ljung_box_p_value=0.0,
        ),
    )


def _validation() -> ValidationResult:
    return ValidationResult(
        is_valid=True,
        row_count=144,
        missing_timestamps=0,
        duplicate_timestamps=0,
        missing_values=0,
        is_regular=True,
        frequency="MS",
        issues=[],
        summary="Valid monthly data.",
    )


def test_month_start_index_is_regular_and_complete() -> None:
    """Calendar months remain regular even though their day counts differ."""
    frame = _airline_frame()
    index = pd.DatetimeIndex(frame["Month"])

    missing, is_regular, frequency = time_index_quality(index, "MS")
    preflight = run_preflight_checks(frame, "Month", "Passengers", 12)

    assert (missing, is_regular, frequency) == (0, True, "MS")
    assert preflight.missing_timestamps == 0
    assert preflight.is_regular is True
    assert "Irregular time intervals detected." not in preflight.issues


def test_airline_stationarity_statistics_are_preserved() -> None:
    """Displayed test statistics must come from statsmodels, not placeholders."""
    series = _airline_frame().set_index("Month")["Passengers"]

    evidence = assess_stationarity(series)

    assert evidence.adf_statistic == pytest.approx(0.8153688792)
    assert evidence.adf_p_value == pytest.approx(0.9918802434)
    assert evidence.kpss_statistic == pytest.approx(1.6513122354)
    assert evidence.kpss_p_value == pytest.approx(0.01)


def test_rounded_rmse_is_supported_evidence() -> None:
    """Normal display rounding must not be reported as an invented metric."""
    evidence = {"all_metrics": {"Holt-Winters": {"RMSE": 16.3929}}}

    assert _check_invented_metrics("RMSE=16.39", evidence) == []
    assert _check_invented_metrics("RMSE=19.0", evidence)


def test_report_does_not_recommend_fixing_zero_collection_defects() -> None:
    """Outliers alone must not trigger a recommendation about missing data."""
    quality = DataQualitySection(
        rating="Fair",
        rating_explanation="Potential anomalies require review.",
        missing_values=0,
        duplicate_timestamps=0,
        missing_timestamps=0,
        outlier_count=11,
        outlier_ratio=0.076,
        is_regular=True,
        frequency="MS",
        completeness_pct=100.0,
    )
    recommendations = ExecutiveReportBuilder()._build_recommendations(
        _statistical(),
        _forecast(),
        None,
        ConfidenceAssessment(
            score=70,
            label="Medium",
            explanation="Some monitoring is warranted.",
        ),
        quality,
    )

    text = " ".join(item.recommendation for item in recommendations)
    assert "0 missing values" not in text
    assert "data collection processes" not in text


def test_explainability_agrees_with_residual_warning() -> None:
    """Autocorrelated residuals cannot be described as an acceptable fit."""
    explanation = ExecutiveReportBuilder()._build_explainability(
        _statistical(),
        _forecast(),
        ConfidenceAssessment(
            score=70,
            label="Medium",
            explanation="Some monitoring is warranted.",
        ),
    )
    findings = [item.finding for item in explanation.findings]

    assert "Residual diagnostics require monitoring" in findings
    assert "Residual diagnostics indicate acceptable model fit" not in findings


def test_executive_fallback_calls_endpoint_change_by_its_name() -> None:
    """A seasonal endpoint comparison must not be labelled forecast growth."""
    section = ExecutiveReportBuilder()._build_executive_summary(
        _forecast(),
        _statistical(),
        ConfidenceAssessment(
            score=70,
            label="Medium",
            explanation="Some monitoring is warranted.",
        ),
        DataQualitySection(
            rating="Good",
            rating_explanation="Complete monthly data.",
            missing_values=0,
            duplicate_timestamps=0,
            missing_timestamps=0,
            outlier_count=0,
            outlier_ratio=0.0,
            is_regular=True,
            frequency="MS",
            completeness_pct=100.0,
        ),
        None,
    )
    narrative = _fallback_narrative(section, "executive_summary")

    assert "endpoint change" in narrative.lower()
    assert "expected growth" not in narrative.lower()


def test_holt_winters_assumption_does_not_claim_stationarity_transformation() -> None:
    """Component models must not inherit ARIMA transformation language."""
    assumptions = ExecutiveReportBuilder()._build_assumptions(
        _statistical(),
        ValidationResult(
            is_valid=True,
            row_count=144,
            missing_timestamps=0,
            duplicate_timestamps=0,
            missing_values=0,
            is_regular=True,
            frequency="MS",
            issues=[],
            summary="Valid monthly data.",
        ),
        _forecast(),
    )

    text = " ".join(item.assumption for item in assumptions)
    assert "required transformation" not in text
    assert "level, trend, and seasonal structure" in text


def test_recent_holdout_degradation_is_interpreted() -> None:
    """A materially weaker latest holdout must be stated, not only tabulated."""
    forecast = _forecast().model_copy(
        update={"final_test_metrics": {"rmse": 29.7418, "mae": 26.7118}}
    )

    metrics = ExecutiveReportBuilder()._build_forecast_metrics(forecast)

    assert metrics.final_test_assessment is not None
    assert "1.81×" in metrics.final_test_assessment
    assert "weaker performance" in metrics.final_test_assessment


def test_recent_holdout_degradation_reduces_confidence_once_at_threshold() -> None:
    """The shared 1.25 ratio produces one deterministic confidence deduction."""
    builder = ExecutiveReportBuilder()
    base_forecast = _forecast().model_copy(update={"selection_metrics": {"rmse": 16.0}})
    below_threshold = base_forecast.model_copy(
        update={
            "final_test_metrics": {
                "rmse": 16.0 * (RECENT_HOLDOUT_RMSE_RATIO_THRESHOLD - 0.001)
            }
        }
    )
    at_threshold = base_forecast.model_copy(
        update={
            "final_test_metrics": {"rmse": 16.0 * RECENT_HOLDOUT_RMSE_RATIO_THRESHOLD}
        }
    )

    base = builder._compute_confidence(
        base_forecast, _statistical(), _validation(), None
    )
    below = builder._compute_confidence(
        below_threshold, _statistical(), _validation(), None
    )
    limited = builder._compute_confidence(
        at_threshold, _statistical(), _validation(), None
    )

    assert below.score == base.score
    assert limited.score == (
        base.score - CONFIDENCE_DEDUCTIONS["recent_holdout_degradation"]
    )
    matching_factors = [
        factor
        for factor in limited.contributing_factors
        if "untouched-holdout RMSE" in factor
    ]
    assert len(matching_factors) == 1
    assert "1.25×" in limited.explanation

    quality = builder._compute_data_quality(_validation(), _statistical())
    recommendation = builder._build_recommendations(
        _statistical(), at_threshold, None, limited, quality
    )[0]
    assert "at or above" in recommendation.rationale


def test_review_warning_does_not_double_count_high_mape() -> None:
    """A review flag that only repeats MAPE must not add a generic penalty."""
    forecast = _forecast().model_copy(update={"mape": 25.0})
    review = StatisticalReviewResult(
        verdict="warn",
        flags=[
            {
                "agent": "forecasting",
                "severity": "warning",
                "issue": "Forecast MAPE is 25.00%, indicating high prediction error.",
                "recommendation": "Review model adequacy.",
            }
        ],
        endorsements=[],
        summary="High MAPE warning.",
    )
    builder = ExecutiveReportBuilder()

    direct_only = builder._compute_confidence(
        forecast, _statistical(), _validation(), None
    )
    with_review = builder._compute_confidence(
        forecast, _statistical(), _validation(), review
    )

    assert with_review.score == direct_only.score
    assert "Statistical review raised warnings" not in with_review.contributing_factors


def test_airline_data_quality_separates_collection_and_anomaly_policy() -> None:
    """A clean index and elevated anomaly ratio produce one coherent Fair rating."""
    statistical = _statistical().model_copy(
        update={"outlier_count": 11, "outlier_ratio": 0.076}
    )

    quality = ExecutiveReportBuilder()._compute_data_quality(_validation(), statistical)

    assert quality.rating == "Fair"
    assert "Collection quality is good" in quality.rating_explanation
    assert "7.6%" in quality.rating_explanation
    assert "5% review threshold" in quality.rating_explanation
    assert "limiting the overall rating to Fair" in quality.rating_explanation
    assert "insignificant" not in quality.rating_explanation.lower()
    action, _ = recommended_action(None, quality)
    assert "anomalies" in action.lower()
    assert "data collection" not in action.lower()


def test_data_quality_explains_validation_issue_only_rating() -> None:
    """Validation issues that drive Fair must appear in its explanation/action."""
    validation = _validation().model_copy(
        update={"issues": ["Series is short for seasonal validation."]}
    )
    statistical = _statistical().model_copy(
        update={"outlier_count": 0, "outlier_ratio": 0.0}
    )

    quality = ExecutiveReportBuilder()._compute_data_quality(validation, statistical)
    action, _ = recommended_action(None, quality)

    assert quality.rating == "Fair"
    assert "1 validation issue" in quality.rating_explanation
    assert "Series is short for seasonal validation." in quality.rating_explanation
    assert "validation issues" in action.lower()
    assert "anomal" not in action.lower()


def test_structural_break_actions_validate_before_model_changes() -> None:
    """Candidate change points must trigger validation before modelling options."""
    builder = ExecutiveReportBuilder()
    quality = builder._compute_data_quality(_validation(), _statistical())
    confidence = ConfidenceAssessment(
        score=75,
        label="High",
        explanation="Evidence is generally stable.",
    )
    recommendations = builder._build_recommendations(
        _statistical(),
        _forecast(),
        None,
        confidence,
        quality,
        has_structural_breaks=True,
    )
    risks = builder._build_risks(
        _statistical(),
        _forecast(),
        None,
        quality,
        has_structural_breaks=True,
    )
    break_recommendation = next(
        item
        for item in recommendations
        if any(ref.metric == "Change Points" for ref in item.supporting_evidence)
    )
    break_risk = next(
        item for item in risks if "candidate breaks" in item.description.lower()
    )

    for text in (break_recommendation.recommendation, break_risk.mitigation):
        normalized = text.lower()
        validation_position = normalized.index("validate")
        for option in (
            "intervention terms",
            "recency weighting",
            "segmentation",
            "regime-specific models",
        ):
            assert validation_position < normalized.index(option)
        assert "break dates" in normalized
        assert "effect sizes" in normalized
        assert "persistence" in normalized
    assert "segment the data by regime" not in break_risk.mitigation.lower()


def test_pipeline_change_point_note_is_validation_first() -> None:
    """Pipeline evidence must not immediately prescribe segmentation."""
    statistical = _statistical().model_copy(
        update={
            "recommended_remediation": ["change_point_analysis"],
            "summary": "Initial evidence.",
        }
    )
    series = pd.Series([1.0, 2.0, 3.0])

    result = _apply_agent_remediation(series, statistical, [], {})

    pd.testing.assert_series_equal(result, series)
    summary = statistical.summary.lower()
    assert summary.index("validate") < summary.index("segmentation")
    assert "break dates" in summary
    assert "effect sizes" in summary
    assert "persistence" in summary
    assert "consider segmenting" not in summary


def test_structural_break_prompts_preserve_validation_order() -> None:
    """Every relevant LLM prompt carries the same validation-first guardrail."""
    prompts_and_inputs = (
        (STATISTICAL_ANALYSIS_PROMPT, {"profile": "evidence"}),
        (
            STATISTICAL_REVIEW_PROMPT,
            {
                "statistical_profile": "evidence",
                "model_selection": "selection",
                "forecast_results": "forecast",
                "all_metrics": "metrics",
                "pre_check_flags": "flags",
            },
        ),
        (
            FORECASTING_PROMPT,
            {
                "selected": "Holt-Winters",
                "requested_loss": "mase",
                "business_context": "context",
                "summary": "results",
            },
        ),
        (RECOMMENDATION_NARRATIVE_PROMPT, {"section_json": "{}"}),
    )

    for prompt, inputs in prompts_and_inputs:
        rendered = " ".join(
            str(message.content) for message in prompt.format_messages(**inputs)
        ).lower()
        for phrase in (
            "break dates",
            "effect sizes",
            "persistence",
            "recency weighting",
            "segmentation",
            "regime-specific models",
        ):
            assert phrase in rendered
        assert rendered.index("break dates") < rendered.index("segmentation")


def test_holdout_degradation_drives_future_actual_monitoring() -> None:
    """Completed validation is acknowledged while weaker recent evidence is monitored."""
    forecast = _forecast().model_copy(
        update={
            "selection_metrics": {"rmse": 16.3929},
            "final_test_metrics": {"rmse": 29.7418, "mae": 26.7118},
        }
    )
    builder = ExecutiveReportBuilder()
    quality = builder._compute_data_quality(_validation(), _statistical())
    confidence = builder._compute_confidence(
        forecast, _statistical(), _validation(), None
    )

    recommendation = builder._build_recommendations(
        _statistical(), forecast, None, confidence, quality
    )[0]
    summary = builder._build_executive_summary(
        forecast, _statistical(), confidence, quality, None
    )

    assert "future actuals" in recommendation.recommendation.lower()
    assert "1.81×" in recommendation.rationale
    assert "confirmed with out-of-sample" not in recommendation.rationale.lower()
    assert "future actuals" in summary.recommended_action.lower()
    assert "1.81×" in summary.recommended_action
    dashboard_action, dashboard_status = recommended_action(None, quality, forecast)
    assert "future actuals" in dashboard_action.lower()
    assert "latest untouched holdout" in dashboard_action.lower()
    assert "1.81×" in dashboard_action
    assert dashboard_status == "warning"

    prompt_text = " ".join(
        str(message.content)
        for message in RECOMMENDATION_NARRATIVE_PROMPT.format_messages(
            section_json="{}"
        )
    ).lower()
    assert "completed out-of-sample validation" in prompt_text
    assert "ongoing monitoring" in prompt_text


def test_narrative_guards_reject_report_review_claims() -> None:
    """Unsupported anomaly and interval-calibration claims use deterministic fallback."""
    anomaly_warnings = _unsupported_anomaly_significance_claim(
        "The outliers have minimal impact on the rating.",
        {"outlier_count": 11, "outlier_ratio": 0.076},
    )
    rating_warnings = _contradictory_data_quality_rating(
        "Overall data quality is Good despite the anomalies.",
        {"rating": "Fair", "outlier_count": 11, "outlier_ratio": 0.076},
    )
    interval_warnings = _unsupported_interval_calibration_claim(
        "Each forecast has a calibrated 95% prediction interval.",
        {"metrics": {"prediction_intervals": []}},
    )
    fallback = _fallback_forecast_outlook(
        {
            "metrics": ExecutiveReportBuilder()
            ._build_forecast_metrics(_forecast())
            .model_dump()
        }
    )

    assert anomaly_warnings
    for equivalent_claim in (
        "The flagged values are harmless.",
        "The anomalies are not concerning.",
        "The outliers did not warrant a downgrade.",
    ):
        assert _unsupported_anomaly_significance_claim(
            equivalent_claim,
            {"outlier_count": 11, "outlier_ratio": 0.076},
        )
    assert rating_warnings
    assert interval_warnings
    assert "model-based 95% prediction range" in fallback
    assert "calibrated" not in fallback.lower()

    data_prompt = " ".join(
        str(message.content)
        for message in DATA_QUALITY_NARRATIVE_PROMPT.format_messages(section_json="{}")
    ).lower()
    assert "describe completeness and interval regularity separately" in data_prompt
    assert "never call anomalies or outliers insignificant" in data_prompt


def test_recommendation_narrative_guard_preserves_deterministic_sequence() -> None:
    """LLM prose cannot override structural or completed-validation safeguards."""
    builder = ExecutiveReportBuilder()
    quality = builder._compute_data_quality(_validation(), _statistical())
    forecast = _forecast().model_copy(
        update={"final_test_metrics": {"rmse": 29.7418, "mae": 26.7118}}
    )
    confidence = builder._compute_confidence(
        forecast, _statistical(), _validation(), None
    )
    recommendations = builder._build_recommendations(
        _statistical(),
        forecast,
        None,
        confidence,
        quality,
        has_structural_breaks=True,
    )
    monitoring = recommendations[0].model_dump()
    structural = recommendations[1].model_dump()

    assert _unsupported_recommendation_claims(
        "Segment the history immediately and fit separate regimes.", structural
    )
    assert not _unsupported_recommendation_claims(
        "Validate the break dates and persistence, then compare segmentation "
        "with the other options only if the break is confirmed.",
        structural,
    )
    assert _unsupported_recommendation_claims(
        "Use future actuals for the first out-of-sample validation.", monitoring
    )
    assert _fallback_narrative(recommendations[1], "recommendation") == (
        recommendations[1].recommendation
    )


def test_forecast_chart_uses_estimated_interval_label() -> None:
    """Chart legends must not imply unevidenced empirical calibration."""
    history = pd.Series(
        [100.0, 110.0],
        index=pd.date_range("2020-01-01", periods=2, freq="MS"),
    )

    chart = plot_forecast(history, _forecast())
    trace_names = [trace.get("name", "") for trace in chart["data"]]

    assert "Model-based 95% prediction interval" in trace_names
    assert not any("calibrated" in name.lower() for name in trace_names)

    experimental = _forecast().model_copy(update={"interval_label": "experimental"})
    experimental_names = [
        trace.get("name", "") for trace in plot_forecast(history, experimental)["data"]
    ]
    assert "Estimated 95% prediction interval (coverage not evaluated)" in (
        experimental_names
    )

    unavailable = _forecast().model_copy(
        update={"lower_ci": [], "upper_ci": [], "interval_label": "unavailable"}
    )
    unavailable_names = [
        trace.get("name", "") for trace in plot_forecast(history, unavailable)["data"]
    ]
    assert not any("interval" in name.lower() for name in unavailable_names)


def test_high_error_risk_respects_interval_provenance() -> None:
    """Risk mitigation must not imply interval evidence that is unavailable."""
    builder = ExecutiveReportBuilder()
    quality = builder._compute_data_quality(_validation(), _statistical())
    high_error = _forecast().model_copy(update={"mape": 25.0})

    def mitigation(forecast: ForecastResult) -> str:
        risks = builder._build_risks(_statistical(), forecast, None, quality)
        return next(
            risk.mitigation
            for risk in risks
            if "Forecast validation error is high" in risk.description
        )

    assert "model-based 95% prediction intervals" in mitigation(high_error)
    assert "estimated 95% prediction intervals" in mitigation(
        high_error.model_copy(update={"interval_label": "experimental"})
    )
    unavailable_mitigation = mitigation(
        high_error.model_copy(
            update={
                "lower_ci": [],
                "upper_ci": [],
                "interval_label": "unavailable",
            }
        )
    )
    assert "Prediction-interval bounds are unavailable" in unavailable_mitigation
    assert "without inferring a 95% planning range" in unavailable_mitigation


def test_visual_strategy_uses_prediction_interval_provenance() -> None:
    """Report strategy must not relabel forecast bands as confidence intervals."""
    model_selection = ModelSelectionResult(
        selected_model="Holt-Winters",
        explanation="Lowest rolling-origin MASE.",
    )
    forecast = _forecast().model_copy(update={"mape": 25.0})

    strategy = _compute_visual_strategy(_statistical(), forecast, model_selection)
    strategy_text = " ".join(f"{item['chart']} {item['reason']}" for item in strategy)

    assert "Model-Based 95% Prediction Intervals" in strategy_text
    assert "confidence interval" not in strategy_text.lower()
    assert "95% CI" not in strategy_text

    experimental_strategy = _compute_visual_strategy(
        _statistical(),
        forecast.model_copy(update={"interval_label": "experimental"}),
        model_selection,
    )
    assert any(
        item["chart"] == "Estimated 95% Prediction Intervals (coverage not evaluated)"
        for item in experimental_strategy
    )

    unavailable_strategy = _compute_visual_strategy(
        _statistical(),
        forecast.model_copy(update={"interval_label": "unavailable"}),
        model_selection,
    )
    assert any(
        item["chart"] == "Forecast Error Monitoring" for item in unavailable_strategy
    )
    assert not any("95%" in item["chart"] for item in unavailable_strategy)


def test_frontend_interval_labels_are_conservative() -> None:
    """The structured report template labels both interval provenance branches."""
    template = Path("data_forecaster/frontend/templates/main/report.html").read_text()
    forecast_template = Path(
        "data_forecaster/frontend/templates/main/forecast.html"
    ).read_text()

    assert "Model-Based 95% Forecast Range" in template
    assert "Estimated 95% Forecast Range (coverage not evaluated)" in template
    assert "calibrated" not in template.lower()
    assert "Model-based 95% prediction-interval bounds" in forecast_template
    assert "Estimated 95% prediction-interval bounds" in forecast_template
    assert "Prediction-interval bounds are unavailable" in forecast_template
    assert "Lower CI" not in forecast_template
