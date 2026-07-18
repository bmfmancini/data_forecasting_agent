"""Unit tests for the ExecutiveReportBuilder (Stage 1)."""

from __future__ import annotations

import pytest

from forecasting.contracts import ForecastFitStatus
from report.builder import ExecutiveReportBuilder
from report.models import (
    DashboardItem,
    EvidenceRef,
    ExecutiveReport,
    PredictionInterval,
    Recommendation,
    ReportMetadata,
)
from schemas import (
    ForecastResult,
    ModelSelectionResult,
    StatisticalResult,
    StatisticalReviewResult,
    ValidationResult,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_validation() -> ValidationResult:
    """A clean validation result with no issues."""
    return ValidationResult(
        is_valid=True,
        row_count=144,
        missing_timestamps=0,
        duplicate_timestamps=0,
        missing_values=0,
        is_regular=True,
        frequency="MS",
        frequency_alias="M",
        issues=[],
        summary="Data is clean and regular.",
    )


@pytest.fixture
def sample_statistical() -> StatisticalResult:
    """A statistical result with trend and seasonality."""
    return StatisticalResult(
        is_stationary_adf=False,
        adf_statistic=0.8,
        adf_p_value=0.9,
        is_stationary_kpss=False,
        kpss_statistic=0.5,
        kpss_p_value=0.04,
        has_trend=True,
        trend_slope=2.5,
        outlier_count=2,
        outlier_ratio=0.01,
        is_white_noise=False,
        white_noise_p_value=0.001,
        recommended_remediation=[],
        domain="Airline Passengers",
        seasonal_period=12,
        dominant_period=12.0,
        summary="Upward trend with strong annual seasonality.",
    )


@pytest.fixture
def sample_model_selection() -> ModelSelectionResult:
    """A model selection result with SARIMA selected."""
    return ModelSelectionResult(
        selected_model="SARIMA",
        explanation="SARIMA selected for strong seasonality and trend.",
        holt_winters_rejected_reason="Higher RMSE than SARIMA.",
        arima_rejected_reason="Does not handle seasonality.",
        sarima_rejected_reason=None,
        ewma_rejected_reason="Lags behind trend.",
    )


@pytest.fixture
def sample_forecast() -> ForecastResult:
    """A forecast result with 12 periods and prediction intervals."""
    return ForecastResult(
        model_used="SARIMA",
        status=ForecastFitStatus.OK,
        forecast=[
            400.0,
            410.0,
            420.0,
            430.0,
            440.0,
            450.0,
            460.0,
            470.0,
            480.0,
            490.0,
            500.0,
            510.0,
        ],
        lower_ci=[
            380.0,
            390.0,
            400.0,
            410.0,
            420.0,
            430.0,
            440.0,
            450.0,
            460.0,
            470.0,
            480.0,
            490.0,
        ],
        upper_ci=[
            420.0,
            430.0,
            440.0,
            450.0,
            460.0,
            470.0,
            480.0,
            490.0,
            500.0,
            510.0,
            520.0,
            530.0,
        ],
        forecast_dates=[
            "1961-01-31",
            "1961-02-28",
            "1961-03-31",
            "1961-04-30",
            "1961-05-31",
            "1961-06-30",
            "1961-07-31",
            "1961-08-31",
            "1961-09-30",
            "1961-10-31",
            "1961-11-30",
            "1961-12-31",
        ],
        rmse=15.0,
        mae=12.0,
        mape=3.5,
    )


@pytest.fixture
def sample_review() -> StatisticalReviewResult:
    """A passing statistical review."""
    return StatisticalReviewResult(
        verdict="pass",
        flags=[],
        endorsements=["Model selection is appropriate for the data."],
        summary="The analysis is well-supported by the evidence.",
    )


@pytest.fixture
def sample_all_metrics() -> dict[str, dict[str, float]]:
    """All model comparison metrics."""
    return {
        "SARIMA": {"RMSE": 15.0, "MAE": 12.0, "MAPE": 3.5},
        "ARIMA": {"RMSE": 30.0, "MAE": 25.0, "MAPE": 7.2},
        "Holt-Winters": {"RMSE": 20.0, "MAE": 18.0, "MAPE": 5.1},
        "EWMA": {"RMSE": 50.0, "MAE": 45.0, "MAPE": 12.0},
    }


@pytest.fixture
def built_report(
    sample_validation: ValidationResult,
    sample_statistical: StatisticalResult,
    sample_model_selection: ModelSelectionResult,
    sample_forecast: ForecastResult,
    sample_review: StatisticalReviewResult,
    sample_all_metrics: dict[str, dict[str, float]],
) -> ExecutiveReport:
    """A fully built ExecutiveReport from sample pipeline results."""
    builder = ExecutiveReportBuilder()
    return builder.build(
        validation=sample_validation,
        statistical=sample_statistical,
        model_selection=sample_model_selection,
        forecast=sample_forecast,
        statistical_review=sample_review,
        all_metrics=sample_all_metrics,
        report_title="Q4 report",
        prepared_by="alice",
    )


# ── Tests ────────────────────────────────────────────────────────────────────


class TestExecutiveReportBuilder:
    """Tests for the ExecutiveReportBuilder.build() method."""

    def test_returns_executive_report(self, built_report: ExecutiveReport) -> None:
        assert isinstance(built_report, ExecutiveReport)

    def test_all_sections_populated(self, built_report: ExecutiveReport) -> None:
        assert built_report.metadata is not None
        assert built_report.dashboard is not None
        assert built_report.executive_summary is not None
        assert built_report.data_quality is not None
        assert built_report.historical_analysis is not None
        assert built_report.forecast_outlook is not None
        assert built_report.model_comparison is not None
        assert built_report.confidence is not None
        assert len(built_report.health_indicators) > 0
        assert built_report.explainability is not None
        assert built_report.statistical_audit is not None
        assert len(built_report.risks) > 0
        assert len(built_report.recommendations) > 0
        assert len(built_report.assumptions) > 0
        assert built_report.appendix is not None

    def test_narrative_fields_empty(self, built_report: ExecutiveReport) -> None:
        """Stage 1 should leave narrative fields empty."""
        assert built_report.executive_summary.narrative == ""
        assert built_report.data_quality.narrative == ""
        assert built_report.historical_analysis.narrative == ""
        assert built_report.forecast_outlook.narrative == ""
        assert built_report.model_comparison.narrative == ""
        assert built_report.statistical_audit.narrative == ""
        assert built_report.explainability.narrative == ""

    def test_dashboard_has_seven_items(self, built_report: ExecutiveReport) -> None:
        assert len(built_report.dashboard.widgets) == 7
        for item in built_report.dashboard.widgets:
            assert isinstance(item, DashboardItem)
            assert item.title
            assert item.value
            assert item.status
            assert item.description
            assert item.icon
            assert item.priority > 0

    def test_dashboard_items_sorted_by_priority(
        self, built_report: ExecutiveReport
    ) -> None:
        priorities = [item.priority for item in built_report.dashboard.widgets]
        assert priorities == sorted(priorities)

    def test_confidence_score_in_range(self, built_report: ExecutiveReport) -> None:
        assert 0 <= built_report.confidence.score <= 100
        assert built_report.confidence.label in ("High", "Medium", "Low")

    def test_confidence_explanation_non_empty(
        self, built_report: ExecutiveReport
    ) -> None:
        assert len(built_report.confidence.explanation) > 0
        assert len(built_report.confidence.contributing_factors) > 0

    def test_prediction_intervals_match_forecast(
        self, built_report: ExecutiveReport, sample_forecast: ForecastResult
    ) -> None:
        intervals = built_report.forecast_outlook.metrics.prediction_intervals
        assert len(intervals) == len(sample_forecast.forecast)
        for i, pi in enumerate(intervals):
            assert isinstance(pi, PredictionInterval)
            assert pi.date == sample_forecast.forecast_dates[i]
            assert pi.forecast == round(sample_forecast.forecast[i], 4)
            assert pi.lower_ci == round(sample_forecast.lower_ci[i], 4)
            assert pi.upper_ci == round(sample_forecast.upper_ci[i], 4)
            assert pi.confidence_level == "95%"

    def test_unavailable_intervals_do_not_fabricate_zero_bounds(
        self, sample_forecast: ForecastResult
    ) -> None:
        forecast = sample_forecast.model_copy(
            update={"lower_ci": [], "upper_ci": [], "interval_label": "unavailable"}
        )

        metrics = ExecutiveReportBuilder()._build_forecast_metrics(forecast)

        assert metrics.interval_label == "unavailable"
        assert metrics.prediction_intervals == []

    def test_model_comparison_entries(
        self,
        built_report: ExecutiveReport,
        sample_all_metrics: dict[str, dict[str, float]],
    ) -> None:
        mc = built_report.model_comparison
        assert len(mc.entries) == len(sample_all_metrics)
        assert mc.selected_model == "SARIMA"
        for entry in mc.entries:
            assert entry.model in sample_all_metrics
            if entry.model == "SARIMA":
                assert entry.selected is True
                assert entry.rejected_reason is None
            else:
                assert entry.selected is False
                assert entry.rejected_reason is not None

    def test_recommendations_have_evidence(self, built_report: ExecutiveReport) -> None:
        for rec in built_report.recommendations:
            assert isinstance(rec, Recommendation)
            assert rec.priority in ("High", "Medium", "Low")
            assert rec.recommendation
            assert rec.rationale
            assert rec.expected_outcome
            assert len(rec.supporting_evidence) > 0
            for ev in rec.supporting_evidence:
                assert isinstance(ev, EvidenceRef)
                assert ev.metric
                assert ev.value
                assert ev.source_section

    def test_health_indicators_count(self, built_report: ExecutiveReport) -> None:
        assert len(built_report.health_indicators) == 6
        indicators = [hi.indicator for hi in built_report.health_indicators]
        assert "Data Quality" in indicators
        assert "Trend Stability" in indicators
        assert "Seasonality" in indicators
        assert "Forecast Confidence" in indicators
        assert "Structural Breaks" in indicators
        assert "Residual Diagnostics" in indicators

    def test_data_quality_rating_good(self, built_report: ExecutiveReport) -> None:
        assert built_report.data_quality.rating == "Good"

    def test_metadata_populated(self, built_report: ExecutiveReport) -> None:
        meta = built_report.metadata
        assert meta.engine_version
        assert meta.generated_at
        assert meta.generated_at.endswith("+00:00")
        assert meta.title == "Q4 report"
        assert meta.prepared_by == "alice"
        assert meta.forecast_horizon == 12
        assert meta.selected_model == "SARIMA"
        assert meta.dataset_frequency == "MS"
        assert meta.row_count == 144

    def test_metadata_defaults_support_legacy_serialized_reports(self) -> None:
        metadata = ReportMetadata.model_validate(
            {
                "engine_version": "1.0",
                "generated_at": "2026-07-18T01:00:00+00:00",
                "forecast_horizon": 3,
                "models_evaluated": ["ARIMA"],
                "selected_model": "ARIMA",
                "dataset_frequency": "MS",
                "data_quality_rating": "Good",
                "row_count": 24,
            }
        )

        assert metadata.title == "Forecast Report"
        assert metadata.prepared_by == "Unknown"

    def test_assumptions_count(self, built_report: ExecutiveReport) -> None:
        assert len(built_report.assumptions) >= 4

    def test_risks_non_empty(self, built_report: ExecutiveReport) -> None:
        assert len(built_report.risks) >= 1

    def test_explainability_items(self, built_report: ExecutiveReport) -> None:
        assert len(built_report.explainability.findings) >= 1
        for item in built_report.explainability.findings:
            assert item.finding
            assert item.evidence
            assert item.interpretation


class TestConfidenceScoreDeductions:
    """Tests that confidence score deductions are applied correctly."""

    def test_high_mape_reduces_score(
        self,
        sample_validation: ValidationResult,
        sample_statistical: StatisticalResult,
        sample_model_selection: ModelSelectionResult,
        sample_review: StatisticalReviewResult,
        sample_all_metrics: dict[str, dict[str, float]],
    ) -> None:
        forecast = ForecastResult(
            model_used="SARIMA",
            status=ForecastFitStatus.OK,
            forecast=[400.0, 410.0],
            lower_ci=[380.0, 390.0],
            upper_ci=[420.0, 430.0],
            forecast_dates=["2024-01-31", "2024-02-28"],
            rmse=50.0,
            mae=45.0,
            mape=25.0,
        )
        builder = ExecutiveReportBuilder()
        report = builder.build(
            validation=sample_validation,
            statistical=sample_statistical,
            model_selection=sample_model_selection,
            forecast=forecast,
            statistical_review=sample_review,
            all_metrics=sample_all_metrics,
        )
        assert report.confidence.score < 100

    def test_non_stationary_reduces_score(
        self,
        sample_validation: ValidationResult,
        sample_model_selection: ModelSelectionResult,
        sample_forecast: ForecastResult,
        sample_review: StatisticalReviewResult,
        sample_all_metrics: dict[str, dict[str, float]],
    ) -> None:
        statistical = StatisticalResult(
            is_stationary_adf=False,
            adf_statistic=0.8,
            adf_p_value=0.9,
            is_stationary_kpss=False,
            kpss_statistic=0.5,
            kpss_p_value=0.04,
            has_trend=True,
            trend_slope=2.5,
            outlier_count=0,
            outlier_ratio=0.0,
            is_white_noise=False,
            white_noise_p_value=0.001,
            recommended_remediation=[],
            domain="Test",
            seasonal_period=12,
            dominant_period=12.0,
            summary="Non-stationary.",
        )
        builder = ExecutiveReportBuilder()
        report = builder.build(
            validation=sample_validation,
            statistical=statistical,
            model_selection=sample_model_selection,
            forecast=sample_forecast,
            statistical_review=sample_review,
            all_metrics=sample_all_metrics,
        )
        assert report.confidence.score < 100

    def test_review_fail_reduces_score(
        self,
        sample_validation: ValidationResult,
        sample_statistical: StatisticalResult,
        sample_model_selection: ModelSelectionResult,
        sample_forecast: ForecastResult,
        sample_all_metrics: dict[str, dict[str, float]],
    ) -> None:
        review = StatisticalReviewResult(
            verdict="fail",
            flags=[{"severity": "critical", "issue": "Test", "recommendation": "Fix"}],
            endorsements=[],
            summary="Failed.",
        )
        builder = ExecutiveReportBuilder()
        report = builder.build(
            validation=sample_validation,
            statistical=sample_statistical,
            model_selection=sample_model_selection,
            forecast=sample_forecast,
            statistical_review=review,
            all_metrics=sample_all_metrics,
        )
        assert report.confidence.score < 100

    def test_clean_data_high_confidence(
        self,
        sample_validation: ValidationResult,
        sample_model_selection: ModelSelectionResult,
        sample_review: StatisticalReviewResult,
        sample_all_metrics: dict[str, dict[str, float]],
    ) -> None:
        statistical = StatisticalResult(
            is_stationary_adf=True,
            adf_statistic=-3.5,
            adf_p_value=0.01,
            is_stationary_kpss=True,
            kpss_statistic=0.1,
            kpss_p_value=0.6,
            has_trend=True,
            trend_slope=2.5,
            outlier_count=0,
            outlier_ratio=0.0,
            is_white_noise=False,
            white_noise_p_value=0.001,
            recommended_remediation=[],
            domain="Test",
            seasonal_period=12,
            dominant_period=12.0,
            summary="Clean stationary data.",
        )
        forecast = ForecastResult(
            model_used="SARIMA",
            status=ForecastFitStatus.OK,
            forecast=[400.0, 410.0],
            lower_ci=[380.0, 390.0],
            upper_ci=[420.0, 430.0],
            forecast_dates=["2024-01-31", "2024-02-28"],
            rmse=5.0,
            mae=4.0,
            mape=2.0,
        )
        builder = ExecutiveReportBuilder()
        report = builder.build(
            validation=sample_validation,
            statistical=statistical,
            model_selection=sample_model_selection,
            forecast=forecast,
            statistical_review=sample_review,
            all_metrics=sample_all_metrics,
        )
        assert report.confidence.score >= 75
        assert report.confidence.label == "High"
