"""Typed Pydantic models for the structured executive report.

Every section of the executive report is represented by a strongly-typed
Pydantic model.  The top-level :class:`ExecutiveReport` aggregates all
sections and flows through the reporting pipeline as the central object.

Design principles:
- **Facts vs narrative**: deterministic fields (scores, counts, intervals)
  are computed in Python.  ``narrative`` fields are filled by the LLM in
  Stage 2 and left empty (``""``) by the builder.
- **Evidence traceability**: every :class:`Recommendation` carries
  ``supporting_evidence`` refs that point to the metric, value, and source
  section — making the report auditable.
- **Dynamic dashboard**: :class:`Dashboard` is a list of
  :class:`DashboardItem` widgets iterated by renderers and templates — no
  hardcoded cards.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Dashboard ────────────────────────────────────────────────────────────────


class DashboardItem(BaseModel):
    """A single reusable dashboard widget rendered as a card.

    Attributes:
        title:       Short label (e.g. "Forecast Direction").
        value:       Primary value to display (e.g. "Upward").
        status:      Status token used for colour coding (e.g. "positive",
                     "warning", "neutral").
        description: One-sentence executive context.
        icon:        Emoji or icon identifier for the card header.
        priority:    Sort order (lower = earlier).
    """

    title: str
    value: str
    status: str
    description: str
    icon: str
    priority: int


class Dashboard(BaseModel):
    """Collection of dashboard widgets iterated by renderers."""

    items: list[DashboardItem] = Field(default_factory=list)


# ── Executive Summary ────────────────────────────────────────────────────────


class ExecutiveSummary(BaseModel):
    """Section 2 — one-minute executive summary.

    Attributes:
        strategic_outlook:  1-2 sentence forecast direction summary.
        expected_growth:    Projected growth/decline as a formatted string.
        confidence_level:   Confidence label + score (e.g. "87/100 — High").
        primary_risk:       Top risk identified from the analysis.
        recommended_action: Immediate action for leadership.
        narrative:          LLM-generated polished prose (Stage 2).
    """

    strategic_outlook: str
    expected_growth: str
    confidence_level: str
    primary_risk: str
    recommended_action: str
    narrative: str = ""


# ── Data Quality ─────────────────────────────────────────────────────────────


class DataQualitySection(BaseModel):
    """Section 3 — data quality summary.

    Attributes:
        rating:              "Good", "Fair", or "Poor".
        rating_explanation:  1-2 sentence justification.
        missing_values:      Count of missing values.
        duplicate_timestamps: Count of duplicate timestamps.
        missing_timestamps:  Count of gaps in the time index.
        outlier_count:       Number of detected outliers.
        outlier_ratio:       Outlier ratio (0.0–1.0).
        is_regular:          Whether the series has regular intervals.
        frequency:           Detected frequency string.
        issues:              List of human-readable issue descriptions.
        completeness_pct:    Estimated completeness percentage (0–100).
        narrative:           LLM-generated prose (Stage 2).
    """

    rating: str
    rating_explanation: str
    missing_values: int
    duplicate_timestamps: int
    missing_timestamps: int
    outlier_count: int
    outlier_ratio: float
    is_regular: bool
    frequency: str
    issues: list[str] = Field(default_factory=list)
    completeness_pct: float
    narrative: str = ""


# ── Forecast Metrics & Prediction Intervals ──────────────────────────────────


class PredictionInterval(BaseModel):
    """A single forecast period with its prediction interval.

    Attributes:
        date:             Forecast date string.
        forecast:         Point forecast value.
        lower_ci:         Lower bound of the prediction interval.
        upper_ci:         Upper bound of the prediction interval.
        confidence_level: Confidence level label (e.g. "95%").
    """

    date: str
    forecast: float
    lower_ci: float
    upper_ci: float
    confidence_level: str


class ForecastMetrics(BaseModel):
    """Quantitative forecast results.

    Attributes:
        model_used:    Name of the selected forecasting model.
        horizon:       Number of forecast periods.
        first_date:    First forecast date string.
        last_date:     Last forecast date string.
        first_value:   First forecast point value.
        last_value:    Last forecast point value.
        pct_change:    Projected percentage change over the horizon.
        rmse:          Root mean squared error (validation).
        mae:           Mean absolute error (validation).
        mape:          Mean absolute percentage error (validation).
        prediction_intervals: Per-period prediction intervals.
    """

    model_used: str
    horizon: int
    first_date: str
    last_date: str
    first_value: float
    last_value: float
    pct_change: float
    rmse: float
    mae: float
    mape: float
    prediction_intervals: list[PredictionInterval] = Field(default_factory=list)


# ── Model Comparison ─────────────────────────────────────────────────────────


class ModelComparisonEntry(BaseModel):
    """A single model's validation metrics and selection status.

    Attributes:
        model:          Model name (e.g. "SARIMA").
        rmse:           Root mean squared error.
        mae:            Mean absolute error.
        mape:           Mean absolute percentage error.
        selected:       Whether this model was selected.
        rejected_reason: Why this model was rejected (if not selected).
    """

    model: str
    rmse: float
    mae: float
    mape: float
    selected: bool
    rejected_reason: str | None = None


class ModelComparison(BaseModel):
    """Section 6 — forecasting approach and model comparison.

    Attributes:
        entries:            All evaluated models with metrics.
        selected_model:     Name of the selected model.
        selection_rationale: Why the selected model was chosen.
        narrative:          LLM-generated prose (Stage 2).
    """

    entries: list[ModelComparisonEntry] = Field(default_factory=list)
    selected_model: str
    selection_rationale: str
    narrative: str = ""


# ── Confidence & Health ──────────────────────────────────────────────────────


class ConfidenceAssessment(BaseModel):
    """Forecast confidence score and explanation.

    Attributes:
        score:               Numeric confidence score (0–100).
        label:               "High", "Medium", or "Low".
        explanation:         1-2 sentence justification.
        contributing_factors: List of factors that influenced the score.
    """

    score: int
    label: str
    explanation: str
    contributing_factors: list[str] = Field(default_factory=list)


class HealthIndicator(BaseModel):
    """A single forecast health indicator row.

    Attributes:
        indicator: Indicator name (e.g. "Data Quality").
        status:    Status value (e.g. "Good", "Stable", "Strong").
        detail:    One-sentence explanation.
    """

    indicator: str
    status: str
    detail: str


# ── Risks ────────────────────────────────────────────────────────────────────


class Risk(BaseModel):
    """A strategic risk identified from the analysis.

    Attributes:
        category:          Risk category (e.g. "Data", "Model", "Market").
        description:       What the risk is.
        potential_impact:  Business impact if the risk materialises.
        mitigation:        Suggested mitigation approach.
        evidence:          Supporting evidence strings from the analysis.
        severity:          "High", "Medium", or "Low".
    """

    category: str
    description: str
    potential_impact: str
    mitigation: str
    evidence: list[str] = Field(default_factory=list)
    severity: str


# ── Recommendations & Evidence ───────────────────────────────────────────────


class EvidenceRef(BaseModel):
    """A traceable reference to a metric that supports a recommendation.

    Attributes:
        metric:        Metric name (e.g. "MAPE").
        value:         Metric value as a string (e.g. "0.67%").
        source_section: Report section where the metric appears.
    """

    metric: str
    value: str
    source_section: str


class Recommendation(BaseModel):
    """A deterministic, evidence-backed recommendation.

    Attributes:
        priority:             "High", "Medium", or "Low".
        recommendation:       The action to take (deterministic).
        rationale:            Why this action is recommended.
        supporting_evidence:  Traceable evidence references.
        expected_outcome:     What the action is expected to achieve.
        narrative:            LLM-rewritten executive prose (Stage 2).
    """

    priority: str
    recommendation: str
    rationale: str
    supporting_evidence: list[EvidenceRef] = Field(default_factory=list)
    expected_outcome: str
    narrative: str | None = None


# ── Assumptions ──────────────────────────────────────────────────────────────


class Assumption(BaseModel):
    """A core business assumption the forecast depends on.

    Attributes:
        assumption:          The assumption statement.
        consequence_if_false: Material consequence if the assumption fails.
    """

    assumption: str
    consequence_if_false: str


# ── Statistical Audit ────────────────────────────────────────────────────────


class StatisticalAudit(BaseModel):
    """Section 9 — independent statistical assessment.

    Attributes:
        verdict:              Review verdict ("pass", "warn", "fail").
        strongest_evidence:   Well-supported aspects of the analysis.
        key_concerns:         Concerns raised by the review.
        recommended_follow_up: Recommended follow-up actions.
        narrative:            LLM-generated prose (Stage 2).
    """

    verdict: str
    strongest_evidence: list[str] = Field(default_factory=list)
    key_concerns: list[str] = Field(default_factory=list)
    recommended_follow_up: list[str] = Field(default_factory=list)
    narrative: str = ""


# ── Explainability ───────────────────────────────────────────────────────────


class ExplainabilityItem(BaseModel):
    """A single explainability finding.

    Attributes:
        finding:       What was detected (e.g. "Strong annual seasonality").
        evidence:      The statistical evidence supporting the finding.
        interpretation: Plain-language explanation for executives.
    """

    finding: str
    evidence: str
    interpretation: str


class Explainability(BaseModel):
    """Section 8 — why the AI reached its conclusions.

    Attributes:
        items:     List of explainability findings.
        narrative: LLM-generated prose (Stage 2).
    """

    items: list[ExplainabilityItem] = Field(default_factory=list)
    narrative: str = ""


# ── Historical Analysis ──────────────────────────────────────────────────────


class HistoricalAnalysis(BaseModel):
    """Section 4 — historical performance and trend analysis.

    Attributes:
        trend_direction:   "Upward", "Downward", or "Flat".
        trend_slope:       Linear regression slope.
        has_trend:         Whether a statistically significant trend exists.
        seasonal_period:   Detected seasonal period (if any).
        dominant_period:   Dominant cycle length from periodogram.
        is_stationary:     Whether the series is stationary.
        narrative:         LLM-generated prose (Stage 2).
    """

    trend_direction: str
    trend_slope: float
    has_trend: bool
    seasonal_period: int | None = None
    dominant_period: float | None = None
    is_stationary: bool
    narrative: str = ""


# ── Forecast Outlook ─────────────────────────────────────────────────────────


class ForecastOutlook(BaseModel):
    """Section 5 — future growth and forecast outlook.

    Attributes:
        metrics:   Quantitative forecast metrics + prediction intervals.
        narrative: LLM-generated prose (Stage 2).
    """

    metrics: ForecastMetrics
    narrative: str = ""


# ── Metadata & Appendix ──────────────────────────────────────────────────────


class ReportMetadata(BaseModel):
    """Structured metadata about the report generation run.

    Attributes:
        engine_version:      Forecast engine version string.
        generated_at:        ISO-8601 timestamp.
        forecast_horizon:    Number of periods forecast.
        models_evaluated:    List of model names that were evaluated.
        selected_model:      The model selected for the final forecast.
        dataset_frequency:   Frequency of the input dataset.
        data_quality_rating: Overall data quality rating.
        row_count:           Number of rows in the input dataset.
    """

    engine_version: str
    generated_at: str
    forecast_horizon: int
    models_evaluated: list[str] = Field(default_factory=list)
    selected_model: str
    dataset_frequency: str
    data_quality_rating: str
    row_count: int


class Appendix(BaseModel):
    """Appendix — raw metrics and visual tag placements.

    Attributes:
        raw_metrics:   Raw metric dict for auditability.
        visual_tags:   Visual tag strings emitted by renderers.
    """

    raw_metrics: dict[str, Any] = Field(default_factory=dict)
    visual_tags: list[str] = Field(default_factory=list)


# ── Top-Level ExecutiveReport ────────────────────────────────────────────────


class ExecutiveReport(BaseModel):
    """Central structured report model flowing through the pipeline.

    Attributes:
        metadata:           Report generation metadata.
        dashboard:          Dynamic dashboard widgets.
        executive_summary:  Section 2 — executive summary.
        data_quality:       Section 3 — data quality summary.
        historical_analysis: Section 4 — historical performance.
        forecast_outlook:   Section 5 — forecast outlook.
        model_comparison:   Section 6 — forecasting approach.
        confidence:         Confidence score and explanation.
        health_indicators:  Forecast health indicator table.
        explainability:     Section 8 — AI reasoning explanation.
        statistical_audit:  Section 9 — statistical audit summary.
        risks:              Section 10 — strategic risks.
        recommendations:    Section 11 — executive recommendations.
        assumptions:        Section 12 — critical business assumptions.
        appendix:           Raw metrics and visual tags.
    """

    metadata: ReportMetadata
    dashboard: Dashboard
    executive_summary: ExecutiveSummary
    data_quality: DataQualitySection
    historical_analysis: HistoricalAnalysis
    forecast_outlook: ForecastOutlook
    model_comparison: ModelComparison
    confidence: ConfidenceAssessment
    health_indicators: list[HealthIndicator] = Field(default_factory=list)
    explainability: Explainability
    statistical_audit: StatisticalAudit
    risks: list[Risk] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    appendix: Appendix