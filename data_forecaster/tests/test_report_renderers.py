"""Unit tests for the report renderers (Markdown and HTML)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")),
)

from report.builder import ExecutiveReportBuilder  # noqa: E402
from report.renderers import HTMLRenderer, MarkdownRenderer  # noqa: E402
from schemas import (  # noqa: E402
    ForecastResult,
    ModelSelectionResult,
    StatisticalResult,
    StatisticalReviewResult,
    ValidationResult,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_report() -> "object":
    """Build a sample ExecutiveReport for renderer tests."""
    validation = ValidationResult(
        is_valid=True,
        row_count=144,
        missing_timestamps=0,
        duplicate_timestamps=0,
        missing_values=0,
        is_regular=True,
        frequency="MS",
        frequency_alias="M",
        issues=[],
        summary="Clean data.",
    )
    statistical = StatisticalResult(
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
        domain="Airline",
        seasonal_period=12,
        dominant_period=12.0,
        summary="Upward trend with seasonality.",
    )
    model_selection = ModelSelectionResult(
        selected_model="SARIMA",
        explanation="SARIMA selected for seasonality.",
        holt_winters_rejected_reason="Higher RMSE.",
        arima_rejected_reason="No seasonality.",
        sarima_rejected_reason=None,
        ewma_rejected_reason="Lags trend.",
    )
    forecast = ForecastResult(
        model_used="SARIMA",
        forecast=[400.0, 410.0, 420.0],
        lower_ci=[380.0, 390.0, 400.0],
        upper_ci=[420.0, 430.0, 440.0],
        forecast_dates=["2024-01-31", "2024-02-28", "2024-03-31"],
        rmse=15.0,
        mae=12.0,
        mape=3.5,
    )
    review = StatisticalReviewResult(
        verdict="pass",
        flags=[],
        endorsements=["Good model selection."],
        summary="Well-supported analysis.",
    )
    all_metrics = {
        "SARIMA": {"RMSE": 15.0, "MAE": 12.0, "MAPE": 3.5},
        "ARIMA": {"RMSE": 30.0, "MAE": 25.0, "MAPE": 7.2},
    }
    builder = ExecutiveReportBuilder()
    return builder.build(
        validation=validation,
        statistical=statistical,
        model_selection=model_selection,
        forecast=forecast,
        statistical_review=review,
        all_metrics=all_metrics,
    )


# ── Markdown Renderer Tests ──────────────────────────────────────────────────


class TestMarkdownRenderer:
    """Tests for the MarkdownRenderer."""

    def test_render_produces_non_empty_string(
        self, sample_report: "object"
    ) -> None:
        renderer = MarkdownRenderer()
        md = renderer.render(sample_report)
        assert isinstance(md, str)
        assert len(md) > 0

    def test_all_twelve_sections_present(
        self, sample_report: "object"
    ) -> None:
        renderer = MarkdownRenderer()
        md = renderer.render(sample_report)
        for i in range(1, 13):
            assert f"## {i}." in md, f"Section {i} missing"

    def test_appendix_present(self, sample_report: "object") -> None:
        renderer = MarkdownRenderer()
        md = renderer.render(sample_report)
        assert "## Appendix" in md

    def test_prediction_intervals_table_present(
        self, sample_report: "object"
    ) -> None:
        renderer = MarkdownRenderer()
        md = renderer.render(sample_report)
        assert "Prediction Intervals" in md
        assert "Lower Bound" in md
        assert "Upper Bound" in md

    def test_visual_tags_present(self, sample_report: "object") -> None:
        renderer = MarkdownRenderer()
        md = renderer.render(sample_report)
        assert "[VISUAL:HISTORICAL]" in md
        assert "[VISUAL:STL]" in md
        assert "[VISUAL:FORECAST]" in md
        assert "[VISUAL:ACF_PACF]" in md
        assert "[VISUAL:COMPARISON]" in md

    def test_dashboard_table_present(self, sample_report: "object") -> None:
        renderer = MarkdownRenderer()
        md = renderer.render(sample_report)
        assert "## 1. Executive Dashboard" in md
        assert "Forecast Direction" in md
        assert "Expected Growth" in md

    def test_confidence_score_in_output(
        self, sample_report: "object"
    ) -> None:
        renderer = MarkdownRenderer()
        md = renderer.render(sample_report)
        assert "Confidence Score:" in md
        assert "/100" in md

    def test_health_indicators_table(self, sample_report: "object") -> None:
        renderer = MarkdownRenderer()
        md = renderer.render(sample_report)
        assert "Forecast Health Indicators" in md
        assert "Data Quality" in md
        assert "Trend Stability" in md

    def test_recommendations_with_evidence(
        self, sample_report: "object"
    ) -> None:
        renderer = MarkdownRenderer()
        md = renderer.render(sample_report)
        assert "## 11. Executive Recommendations" in md
        assert "Supporting Evidence" in md
        assert "MAPE" in md

    def test_metadata_table(self, sample_report: "object") -> None:
        renderer = MarkdownRenderer()
        md = renderer.render(sample_report)
        assert "Engine Version" in md
        assert "Generated At" in md
        assert "Forecast Horizon" in md

    def test_no_fabricated_financials(self, sample_report: "object") -> None:
        """Fallback narratives should not contain fabricated financials."""
        renderer = MarkdownRenderer()
        md = renderer.render(sample_report)
        assert "$" not in md or "million" not in md.lower()


# ── HTML Renderer Tests ──────────────────────────────────────────────────────


class TestHTMLRenderer:
    """Tests for the HTMLRenderer."""

    def test_render_produces_html_string(
        self, sample_report: "object"
    ) -> None:
        renderer = HTMLRenderer()
        html = renderer.render(sample_report)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_dashboard_cards_present(self, sample_report: "object") -> None:
        renderer = HTMLRenderer()
        html = renderer.render(sample_report)
        assert "dashboard-card" in html
        assert "Forecast Direction" in html

    def test_confidence_badge_present(self, sample_report: "object") -> None:
        renderer = HTMLRenderer()
        html = renderer.render(sample_report)
        assert "Forecast Confidence:" in html
        assert "badge" in html

    def test_health_indicators_table(self, sample_report: "object") -> None:
        renderer = HTMLRenderer()
        html = renderer.render(sample_report)
        assert "Forecast Health Indicators" in html
        assert "<table" in html
        assert "Data Quality" in html

    def test_prediction_intervals_table(
        self, sample_report: "object"
    ) -> None:
        renderer = HTMLRenderer()
        html = renderer.render(sample_report)
        assert "Prediction Intervals" in html
        assert "Lower Bound" in html

    def test_recommendations_with_evidence(
        self, sample_report: "object"
    ) -> None:
        renderer = HTMLRenderer()
        html = renderer.render(sample_report)
        assert "Executive Recommendations" in html
        assert "Supporting Evidence" in html or "MAPE" in html

    def test_metadata_table(self, sample_report: "object") -> None:
        renderer = HTMLRenderer()
        html = renderer.render(sample_report)
        assert "Report Metadata" in html
        assert "Engine Version" in html