"""Unit tests for the report renderers (Markdown and HTML)."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from jinja2 import Environment, FileSystemLoader

from forecasting.contracts import ForecastFitStatus
import report.narrative as narrative
from report.builder import ExecutiveReportBuilder
from report.renderers import HTMLRenderer, MarkdownRenderer
from schemas import (
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
        status=ForecastFitStatus.OK,
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

    def test_render_produces_non_empty_string(self, sample_report: "object") -> None:
        renderer = MarkdownRenderer()
        md = renderer.render(sample_report)
        assert isinstance(md, str)
        assert len(md) > 0

    def test_all_twelve_sections_present(self, sample_report: "object") -> None:
        renderer = MarkdownRenderer()
        md = renderer.render(sample_report)
        for i in range(1, 13):
            assert f"## {i}." in md, f"Section {i} missing"

    def test_appendix_present(self, sample_report: "object") -> None:
        renderer = MarkdownRenderer()
        md = renderer.render(sample_report)
        assert "## Appendix" in md

    def test_prediction_intervals_table_present(self, sample_report: "object") -> None:
        renderer = MarkdownRenderer()
        md = renderer.render(sample_report)
        assert "Model-Based 95% Prediction Intervals" in md
        assert "calibrated" not in md.lower()
        assert "Lower Bound" in md
        assert "Upper Bound" in md

    def test_experimental_interval_caption_is_not_model_based(
        self, sample_report: "object"
    ) -> None:
        report = sample_report.model_copy(deep=True)
        report.forecast_outlook.metrics.interval_label = "experimental"
        for interval in report.forecast_outlook.metrics.prediction_intervals:
            interval.interval_label = "experimental"
            interval.confidence_level = "95% (experimental)"

        section = MarkdownRenderer()._render_forecast_outlook(report)

        assert "Estimated 95% Prediction Intervals (coverage not evaluated)" in section
        assert "empirical coverage was not evaluated" in section
        assert "model-based 95% planning range" not in section.lower()

    def test_unavailable_intervals_are_explicit(
        self, sample_report: "object"
    ) -> None:
        report = sample_report.model_copy(deep=True)
        report.forecast_outlook.metrics.prediction_intervals = []
        report.forecast_outlook.metrics.interval_label = "unavailable"

        section = MarkdownRenderer()._render_forecast_outlook(report)

        assert "Prediction Intervals Unavailable" in section
        assert "no 95% planning range is shown" in section
        assert "Model-Based 95%" not in section

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
        assert "Forecast Pattern" in md
        assert "Forecast Endpoint Change" in md

    def test_confidence_score_in_output(self, sample_report: "object") -> None:
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

    def test_recommendations_with_evidence(self, sample_report: "object") -> None:
        renderer = MarkdownRenderer()
        md = renderer.render(sample_report)
        assert "## 11. Executive Recommendations" in md
        assert "Evidence" in md
        assert "MAPE" in md

    def test_metadata_table(self, sample_report: "object") -> None:
        renderer = MarkdownRenderer()
        md = renderer.render(sample_report)
        assert "Engine Version" in md
        assert "Generated At" in md
        assert "Forecast Horizon" in md

    def test_metadata_identifies_deterministic_narrative_fallback(
        self, sample_report: "object"
    ) -> None:
        report = sample_report.model_copy(deep=True)
        report.metadata.llm_narrative_fallback = True
        report.metadata.llm_fallback_sections = [
            "historical_analysis",
            "forecast_outlook",
        ]

        md = MarkdownRenderer().render(report)

        assert "Narrative Generation" in md
        assert "Deterministic fallback used" in md
        assert "historical analysis, forecast outlook" in md

    def test_no_fabricated_financials(self, sample_report: "object") -> None:
        """Fallback narratives should not contain fabricated financials."""
        renderer = MarkdownRenderer()
        md = renderer.render(sample_report)
        # Detect explicit currency amounts followed by scale words like
        # "million" or "billion" — a reliable signal of fabricated figures.
        financial_pattern = re.compile(
            r"\$\s?\d[\d,]*\.?\d*\s*(?:million|billion|trillion)",
            re.IGNORECASE,
        )
        assert not financial_pattern.search(
            md
        ), "Rendered markdown contains fabricated financial figures."


# ── Section 10–12 flowing-prose / fallback tests ─────────────────────────────


class TestSectionFallbackProse:
    """Risks/Recommendations/Assumptions render as merged prose without labels."""

    def test_risks_fallback_merges_seeds_without_labels(
        self, sample_report: "object"
    ) -> None:
        md = MarkdownRenderer().render(sample_report)
        assert "## 10. Strategic Risks" in md
        # Horizon length must not be presented as measured accuracy decay.
        assert "The horizon alone does not establish how accuracy changes" in md
        assert "Refresh the forecast as actual results arrive" in md
        assert "### Longer-term commitments need updated forecasts" in md
        assert "**Business implication:**" in md
        assert "**Recommended action:**" in md
        assert "### Model — Low" not in md
        # Old field labels must not appear.
        assert "**Risk:**" not in md
        assert "**Potential Impact:**" not in md
        assert "**Mitigation:**" not in md

    def test_recommendations_fallback_merges_seeds_without_labels(
        self, sample_report: "object"
    ) -> None:
        md = MarkdownRenderer().render(sample_report)
        assert "## 11. Executive Recommendations" in md
        assert "**Action:**" not in md
        assert "**Rationale:**" not in md
        assert "**Expected Outcome:**" not in md
        # Evidence bullets are still present.
        assert "**Evidence:**" in md

    def test_assumptions_fallback_merges_seeds_without_labels(
        self, sample_report: "object"
    ) -> None:
        md = MarkdownRenderer().render(sample_report)
        assert "## 12. Critical Business Assumptions" in md
        assert "Consequence if false" not in md
        # Numbered flowing sentences are rendered.
        assert "1. " in md

    def test_html_risks_recommendations_drop_labels(
        self, sample_report: "object"
    ) -> None:
        html = HTMLRenderer().render(sample_report)
        assert "<h6>Longer-term commitments need updated forecasts</h6>" in html
        assert "<strong>Business implication:</strong>" in html
        assert "<strong>Recommended action:</strong>" in html
        assert "Model — Low" not in html
        assert "<strong>Risk:</strong>" not in html
        assert "<strong>Potential Impact:</strong>" not in html
        assert "<strong>Mitigation:</strong>" not in html
        assert "Expected outcome:" not in html
        assert "Consequence if false" not in html


def test_business_context_threaded_into_section_prompts(
    sample_report: "object", monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distilled business context is appended to every section prompt."""
    monkeypatch.setattr(narrative, "get_llm", lambda **_kwargs: object())
    monkeypatch.setattr(
        narrative, "get_llm_config", lambda: SimpleNamespace(temperature=0.0)
    )
    report = sample_report.model_copy(deep=True)
    report.metadata.business_context = {"domain": "Retail sales", "units": "USD"}

    captured: list[str] = []

    def capture(
        _llm: object,
        _prompt: object,
        _section: object,
        _section_name: str,
        _total_usage: dict[str, int],
        extra_instructions: str,
        _fallback_sections: list[str],
        _business_context: dict,
    ) -> str:
        captured.append(extra_instructions)
        return "narrative"

    monkeypatch.setattr(narrative, "_generate_section", capture)
    narrative.generate_narratives(report)

    assert captured, "no section prompts were invoked"
    for extra in captured:
        assert "BUSINESS CONTEXT" in extra
        assert "Retail sales" in extra
        assert "USD" in extra


def test_no_business_context_leaves_prompts_unchanged(
    sample_report: "object", monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no context was distilled, prompts carry no business-context block."""
    monkeypatch.setattr(narrative, "get_llm", lambda **_kwargs: object())
    monkeypatch.setattr(
        narrative, "get_llm_config", lambda: SimpleNamespace(temperature=0.0)
    )
    captured: list[str] = []

    def capture(
        _llm: object,
        _prompt: object,
        _section: object,
        _section_name: str,
        _total_usage: dict[str, int],
        extra_instructions: str,
        _fallback_sections: list[str],
        _business_context: dict,
    ) -> str:
        captured.append(extra_instructions)
        return "narrative"

    monkeypatch.setattr(narrative, "_generate_section", capture)
    narrative.generate_narratives(sample_report)

    assert captured
    for extra in captured:
        assert "BUSINESS CONTEXT" not in extra


def test_structured_known_context_threaded_into_prompts(
    sample_report: "object", monkeypatch: pytest.MonkeyPatch
) -> None:
    """Holidays, custom events, and covariates render as a nested context block."""
    monkeypatch.setattr(narrative, "get_llm", lambda **_kwargs: object())
    monkeypatch.setattr(
        narrative, "get_llm_config", lambda: SimpleNamespace(temperature=0.0)
    )
    report = sample_report.model_copy(deep=True)
    report.metadata.business_context = {
        "domain": "Retail sales",
        "known_context": {
            "holidays_country": "US",
            "events_by_type": {"spike": 1, "lull": 1},
            "event_count": 2,
            "covariates": ["price"],
        },
    }

    captured: list[str] = []

    def capture(
        _llm: object,
        _prompt: object,
        _section: object,
        _section_name: str,
        _total_usage: dict[str, int],
        extra_instructions: str,
        _fallback_sections: list[str],
        _business_context: dict,
    ) -> str:
        captured.append(extra_instructions)
        return "narrative"

    monkeypatch.setattr(narrative, "_generate_section", capture)
    narrative.generate_narratives(report)

    assert captured
    for extra in captured:
        assert "BUSINESS CONTEXT" in extra
        assert "known context" in extra
        assert "holiday calendar: US" in extra
        assert "1 spike" in extra
        assert "1 lull" in extra
        assert "declared covariates: price" in extra


def test_narrative_generation_records_section_fallbacks(
    sample_report: "object", monkeypatch: pytest.MonkeyPatch
) -> None:
    """Section-level fallbacks must propagate to the report-level status."""
    monkeypatch.setattr(narrative, "get_llm", lambda **_kwargs: object())
    monkeypatch.setattr(
        narrative, "get_llm_config", lambda: SimpleNamespace(temperature=0.0)
    )

    def always_fallback(
        _llm: object,
        _prompt: object,
        _section: object,
        section_name: str,
        _total_usage: dict[str, int],
        _extra_instructions: str,
        fallback_sections: list[str],
        _business_context: dict,
    ) -> str:
        fallback_sections.append(section_name)
        return f"Fallback for {section_name}"

    monkeypatch.setattr(narrative, "_generate_section", always_fallback)

    report, _ = narrative.generate_narratives(sample_report)

    assert report.metadata.llm_narrative_fallback is True
    assert report.metadata.llm_fallback_sections == [
        "executive_summary",
        "data_quality",
        "historical_analysis",
        "forecast_outlook",
        "model_comparison",
        "statistical_audit",
        "explainability",
        *["recommendation"] * len(report.recommendations),
        *["assumption"] * len(report.assumptions),
    ]


# ── HTML Renderer Tests ──────────────────────────────────────────────────────


class TestHTMLRenderer:
    """Tests for the HTMLRenderer."""

    def test_render_produces_html_string(self, sample_report: "object") -> None:
        renderer = HTMLRenderer()
        html = renderer.render(sample_report)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_dashboard_cards_present(self, sample_report: "object") -> None:
        renderer = HTMLRenderer()
        html = renderer.render(sample_report)
        assert "dashboard-card" in html
        assert "Forecast Pattern" in html

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

    def test_prediction_intervals_table(self, sample_report: "object") -> None:
        renderer = HTMLRenderer()
        html = renderer.render(sample_report)
        assert "Model-Based Prediction Intervals (95%)" in html
        assert "calibrated" not in html.lower()
        assert "Lower Bound" in html

    def test_experimental_interval_heading_is_not_nested(
        self, sample_report: "object"
    ) -> None:
        report = sample_report.model_copy(deep=True)
        report.forecast_outlook.metrics.interval_label = "experimental"
        for interval in report.forecast_outlook.metrics.prediction_intervals:
            interval.interval_label = "experimental"
            interval.confidence_level = "95% (experimental)"

        section = HTMLRenderer()._render_prediction_intervals(report)

        assert "Estimated Prediction Intervals (95%; coverage not evaluated)" in section
        assert "95% (experimental);" not in section

    def test_unavailable_intervals_are_explicit(
        self, sample_report: "object"
    ) -> None:
        report = sample_report.model_copy(deep=True)
        report.forecast_outlook.metrics.prediction_intervals = []
        report.forecast_outlook.metrics.interval_label = "unavailable"

        section = HTMLRenderer()._render_prediction_intervals(report)

        assert "Prediction Intervals Unavailable" in section
        assert "no 95% planning range is shown" in section
        assert "Model-Based" not in section

    def test_frontend_template_renders_interval_provenance_branches(
        self, sample_report: "object"
    ) -> None:
        template_root = "data_forecaster/frontend/templates"
        environment = Environment(loader=FileSystemLoader(template_root))
        environment.globals.update(
            csrf_token=lambda: "",
            current_user=SimpleNamespace(
                is_authenticated=False,
                is_admin=False,
                username="",
            ),
            get_flashed_messages=lambda **_kwargs: [],
            request=SimpleNamespace(endpoint="", blueprint=""),
            session={},
            url_for=lambda *_args, **_kwargs: "#",
        )
        template = environment.get_template("main/report.html")

        experimental = sample_report.model_copy(deep=True)
        experimental.forecast_outlook.metrics.interval_label = "experimental"
        for interval in experimental.forecast_outlook.metrics.prediction_intervals:
            interval.interval_label = "experimental"
        experimental_html = template.render(
            er=experimental,
            segments=[],
            llm_fallback=False,
            export_url="#",
            custom_settings=[],
        )

        unavailable = sample_report.model_copy(deep=True)
        unavailable.forecast_outlook.metrics.interval_label = "unavailable"
        unavailable.forecast_outlook.metrics.prediction_intervals = []
        unavailable_html = template.render(
            er=unavailable,
            segments=[],
            llm_fallback=False,
            export_url="#",
            custom_settings=[],
        )

        assert "Estimated 95% Forecast Range (coverage not evaluated)" in experimental_html
        assert "Model-Based 95% Forecast Range" not in experimental_html
        assert "Prediction Intervals Unavailable" in unavailable_html
        assert "Model-Based 95% Forecast Range" not in unavailable_html

    def test_forecast_template_treats_partial_bounds_as_unavailable(self) -> None:
        environment = Environment(
            loader=FileSystemLoader("data_forecaster/frontend/templates")
        )
        environment.globals.update(
            csrf_token=lambda: "",
            current_user=SimpleNamespace(
                is_authenticated=False,
                is_admin=False,
                username="",
            ),
            get_flashed_messages=lambda **_kwargs: [],
            request=SimpleNamespace(endpoint="", blueprint=""),
            session={},
            url_for=lambda *_args, **_kwargs: "#",
        )

        html = environment.get_template("main/forecast.html").render(
            fc={"interval_label": "prediction_interval"},
            forecast_rows=[
                {
                    "date": "2026-01-01",
                    "forecast": 100.0,
                    "lower_ci": None,
                    "upper_ci": None,
                }
            ],
            forecast_json=None,
        )

        assert "Prediction-interval bounds are unavailable" in html
        assert "Model-based 95% prediction-interval bounds" not in html
        assert "<th>Lower Bound</th>" not in html

    def test_recommendations_with_evidence(self, sample_report: "object") -> None:
        renderer = HTMLRenderer()
        html = renderer.render(sample_report)
        assert "Executive Recommendations" in html
        assert "Supporting Evidence" in html or "MAPE" in html

    def test_metadata_table(self, sample_report: "object") -> None:
        renderer = HTMLRenderer()
        html = renderer.render(sample_report)
        assert "Report Metadata" in html
        assert "Engine Version" in html

    def test_metadata_identifies_deterministic_narrative_fallback(
        self, sample_report: "object"
    ) -> None:
        report = sample_report.model_copy(deep=True)
        report.metadata.llm_narrative_fallback = True
        report.metadata.llm_fallback_sections = ["forecast_outlook"]

        html = HTMLRenderer().render(report)

        assert "Narrative Generation" in html
        assert "Deterministic fallback used (forecast outlook)" in html
