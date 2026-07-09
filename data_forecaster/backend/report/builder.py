"""Stage 1 — deterministic executive report builder.

The :class:`ExecutiveReportBuilder` computes all factual metrics for the
executive report from pipeline results.  It produces a fully-populated
:class:`ExecutiveReport` model with ``narrative`` fields left empty
(those are filled in Stage 2 by :func:`report.narrative.generate_narratives`).

Every method is a pure function with typed inputs — no LLM calls, no I/O,
fully unit-testable.  Business rules are sourced from :mod:`report.rules`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from report.models import (
    Appendix,
    Assumption,
    ConfidenceAssessment,
    Dashboard,
    DashboardItem,
    DataQualitySection,
    EvidenceRef,
    ExecutiveReport,
    ExecutiveSummary,
    Explainability,
    ExplainabilityItem,
    ForecastMetrics,
    ForecastOutlook,
    HealthIndicator,
    HistoricalAnalysis,
    ModelComparison,
    ModelComparisonEntry,
    PredictionInterval,
    Recommendation,
    ReportMetadata,
    Risk,
    StatisticalAudit,
)
from report.rules import (
    CONFIDENCE_DEDUCTIONS,
    DASHBOARD_STATUS_COLORS,
    FORECAST_DIRECTIONS,
    HEALTH_STATUS,
    RECOMMENDATION_PRIORITIES,
    confidence_label,
    data_quality_rating,
    mape_quality,
)
from schemas import (
    ForecastResult,
    ModelSelectionResult,
    StatisticalResult,
    StatisticalReviewResult,
    ValidationResult,
)

_ENGINE_VERSION = "1.0.0"
_CONFIDENCE_LEVEL = "95%"
_REVIEW_CRITICAL_MSG = "Statistical review identified critical issues"


class ExecutiveReportBuilder:
    """Build an :class:`ExecutiveReport` from pipeline results (Stage 1).

    All methods are deterministic and side-effect-free.  The builder does
    not invoke the LLM — narrative fields are left as empty strings for
    Stage 2.
    """

    def build(
        self,
        validation: ValidationResult,
        statistical: StatisticalResult,
        model_selection: ModelSelectionResult,
        forecast: ForecastResult,
        statistical_review: StatisticalReviewResult | None,
        all_metrics: dict[str, dict[str, float]],
    ) -> ExecutiveReport:
        """Construct the full :class:`ExecutiveReport` model.

        Args:
            validation:          Data validation agent output.
            statistical:         Statistical analysis agent output.
            model_selection:     Model selection agent output.
            forecast:            Forecasting agent output.
            statistical_review:  Statistical review (QA) agent output (optional).
            all_metrics:         All model comparison metrics dict.

        Returns:
            A populated :class:`ExecutiveReport` with empty narrative fields.
        """
        has_structural_breaks = (
            "change_point_analysis" in statistical.recommended_remediation
        )
        confidence = self._compute_confidence(
            forecast, statistical, validation, statistical_review,
            has_structural_breaks,
        )
        data_quality = self._compute_data_quality(validation, statistical)
        health_indicators = self._compute_health_indicators(
            statistical, validation, forecast, statistical_review, confidence,
            data_quality, has_structural_breaks,
        )
        forecast_metrics = self._build_forecast_metrics(forecast)
        model_comparison = self._build_model_comparison(
            all_metrics, model_selection
        )
        recommendations = self._build_recommendations(
            statistical, forecast, statistical_review, confidence, data_quality,
            has_structural_breaks,
        )
        risks = self._build_risks(
            statistical, forecast, statistical_review, data_quality,
            has_structural_breaks,
        )
        assumptions = self._build_assumptions(statistical, validation)
        explainability = self._build_explainability(
            statistical, forecast, confidence
        )
        statistical_audit = self._build_statistical_audit(statistical_review)
        historical = self._build_historical_analysis(statistical)
        forecast_outlook = ForecastOutlook(metrics=forecast_metrics)
        dashboard = self._build_dashboard(
            forecast, statistical, model_selection, confidence, data_quality,
            statistical_review, has_structural_breaks,
        )
        executive_summary = self._build_executive_summary(
            forecast, statistical, confidence, data_quality,
            statistical_review, has_structural_breaks,
        )
        metadata = self._build_metadata(
            validation, forecast, model_selection, all_metrics, data_quality,
        )
        appendix = self._build_appendix(forecast, all_metrics)

        return ExecutiveReport(
            metadata=metadata,
            dashboard=dashboard,
            executive_summary=executive_summary,
            data_quality=data_quality,
            historical_analysis=historical,
            forecast_outlook=forecast_outlook,
            model_comparison=model_comparison,
            confidence=confidence,
            health_indicators=health_indicators,
            explainability=explainability,
            statistical_audit=statistical_audit,
            risks=risks,
            recommendations=recommendations,
            assumptions=assumptions,
            appendix=appendix,
        )

    # ── Confidence ────────────────────────────────────────────────────────

    def _compute_confidence(
        self,
        forecast: ForecastResult,
        statistical: StatisticalResult,
        validation: ValidationResult,
        review: StatisticalReviewResult | None,
        has_structural_breaks: bool = False,
    ) -> ConfidenceAssessment:
        """Compute the deterministic confidence score (0–100).

        Starts at 100 and deducts points based on statistical signals,
        validation quality, and review verdict.

        Args:
            forecast:             Forecast result.
            statistical:          Statistical analysis result.
            validation:           Validation result.
            review:               Statistical review result (optional).
            has_structural_breaks: Precomputed flag indicating structural
                breaks were detected (avoids recomputing the remediation
                membership test).

        Returns:
            :class:`ConfidenceAssessment` with score, label, and explanation.
        """
        score = 100
        factors: list[str] = []

        if forecast.mape > 20:
            score -= CONFIDENCE_DEDUCTIONS["mape_above_20"]
            factors.append(f"High validation error (MAPE {forecast.mape:.1f}%)")
        elif forecast.mape > 10:
            score -= CONFIDENCE_DEDUCTIONS["mape_above_10"]
            factors.append(f"Moderate validation error (MAPE {forecast.mape:.1f}%)")
        elif forecast.mape > 5:
            score -= CONFIDENCE_DEDUCTIONS["mape_above_5"]
            factors.append(f"Minor validation error (MAPE {forecast.mape:.1f}%)")

        if not statistical.is_stationary_adf:
            score -= CONFIDENCE_DEDUCTIONS["non_stationary_adf"]
            factors.append("Series is non-stationary")

        if statistical.is_white_noise:
            score -= CONFIDENCE_DEDUCTIONS["white_noise"]
            factors.append("Series resembles random noise")

        if statistical.outlier_ratio > 0.05:
            score -= CONFIDENCE_DEDUCTIONS["outlier_ratio_high"]
            factors.append(
                f"Outlier ratio {statistical.outlier_ratio:.1%} exceeds 5%"
            )

        if validation.missing_values > 0 or validation.missing_timestamps > 0:
            score -= CONFIDENCE_DEDUCTIONS["missing_data"]
            factors.append("Missing values or gaps in the data")

        if review:
            if review.verdict == "warn":
                score -= CONFIDENCE_DEDUCTIONS["review_warn"]
                factors.append("Statistical review raised warnings")
            elif review.verdict == "fail":
                score -= CONFIDENCE_DEDUCTIONS["review_fail"]
                factors.append(_REVIEW_CRITICAL_MSG)

        if has_structural_breaks:
            score -= CONFIDENCE_DEDUCTIONS["structural_breaks"]
            factors.append("Structural breaks detected in the series")

        score = max(0, min(100, score))
        label = confidence_label(score)

        if not factors:
            factors.append("Low validation error")
            factors.append("Stable statistical properties")

        top_factors = factors[:2]
        explanation = (
            f"Confidence is {label.lower()} based on: "
            + "; ".join(top_factors)
            + "."
        )

        return ConfidenceAssessment(
            score=score,
            label=label,
            explanation=explanation,
            contributing_factors=factors,
        )

    # ── Data Quality ──────────────────────────────────────────────────────

    def _compute_data_quality(
        self,
        validation: ValidationResult,
        statistical: StatisticalResult,
    ) -> DataQualitySection:
        """Compute the data quality section from validation results.

        Args:
            validation:  Validation result.
            statistical: Statistical result (for outlier info).

        Returns:
            :class:`DataQualitySection` with rating and metrics.
        """
        issues_count = len(validation.issues)
        rating = data_quality_rating(
            validation.missing_values,
            validation.duplicate_timestamps,
            validation.missing_timestamps,
            issues_count,
            validation.is_regular,
        )
        total_possible = validation.row_count + validation.missing_timestamps
        completeness = (
            (validation.row_count / total_possible * 100)
            if total_possible > 0
            else 100.0
        )
        if rating == "Good":
            explanation = (
                "Data quality is good — no significant gaps, duplicates, "
                "or irregularities detected."
            )
        elif rating == "Fair":
            explanation = (
                "Data quality is fair — some issues were identified that "
                "may have minor influence on forecast reliability."
            )
        else:
            explanation = (
                "Data quality is poor — significant issues were detected "
                "that could materially influence forecast reliability."
            )

        return DataQualitySection(
            rating=rating,
            rating_explanation=explanation,
            missing_values=validation.missing_values,
            duplicate_timestamps=validation.duplicate_timestamps,
            missing_timestamps=validation.missing_timestamps,
            outlier_count=statistical.outlier_count,
            outlier_ratio=statistical.outlier_ratio,
            is_regular=validation.is_regular,
            frequency=validation.frequency or "unknown",
            issues=list(validation.issues),
            completeness_pct=round(completeness, 1),
        )

    # ── Health Indicators ─────────────────────────────────────────────────

    def _compute_health_indicators(
        self,
        statistical: StatisticalResult,
        validation: ValidationResult,
        forecast: ForecastResult,
        review: StatisticalReviewResult | None,
        confidence: ConfidenceAssessment,
        data_quality: DataQualitySection,
        has_structural_breaks: bool = False,
    ) -> list[HealthIndicator]:
        """Compute the forecast health indicator table rows.

        Args:
            statistical:          Statistical result.
            validation:           Validation result (retained for signature
                symmetry; data quality is provided via ``data_quality``).
            forecast:             Forecast result.
            review:               Statistical review result (optional).
            confidence:           Computed confidence assessment.
            data_quality:         Precomputed data quality section.
            has_structural_breaks: Precomputed flag indicating structural
                breaks were detected.

        Returns:
            List of 6 :class:`HealthIndicator` rows.
        """
        del validation  # Data quality is passed in; no recomputation needed.
        data_quality_status = data_quality.rating
        data_quality_detail = data_quality.rating_explanation

        # Trend Stability
        if statistical.has_trend and not statistical.is_stationary_adf:
            trend_status = HEALTH_STATUS["trend_stability"]["changing"]
            trend_detail = "A statistically significant trend is changing the baseline over time."
        else:
            trend_status = HEALTH_STATUS["trend_stability"]["stable"]
            trend_detail = "The underlying trend is stable across the observed period."

        # Seasonality
        sp = statistical.seasonal_period
        if sp and sp > 1:
            seasonality_status = HEALTH_STATUS["seasonality"]["strong"]
            seasonality_detail = f"A recurring seasonal pattern repeats every {sp} periods."
        else:
            seasonality_status = HEALTH_STATUS["seasonality"]["none"]
            seasonality_detail = "No strong seasonal pattern was detected."

        # Forecast Confidence
        conf_status = confidence.label
        conf_detail = confidence.explanation

        # Structural Breaks
        if has_structural_breaks:
            breaks_status = HEALTH_STATUS["structural_breaks"]["monitor"]
            breaks_detail = "Change points detected — monitor for regime shifts."
        else:
            breaks_status = HEALTH_STATUS["structural_breaks"]["none"]
            breaks_detail = "No structural breaks detected."

        # Residual Diagnostics
        if statistical.is_white_noise or forecast.mape > 20:
            resid_status = HEALTH_STATUS["residual_diagnostics"]["concerning"]
            resid_detail = (
                "Residual diagnostics or high validation error suggest "
                "the model may not fully capture the data structure."
            )
        else:
            resid_status = HEALTH_STATUS["residual_diagnostics"]["acceptable"]
            resid_detail = "Residual diagnostics indicate an acceptable model fit."

        return [
            HealthIndicator(
                indicator="Data Quality",
                status=data_quality_status,
                detail=data_quality_detail,
            ),
            HealthIndicator(
                indicator="Trend Stability",
                status=trend_status,
                detail=trend_detail,
            ),
            HealthIndicator(
                indicator="Seasonality",
                status=seasonality_status,
                detail=seasonality_detail,
            ),
            HealthIndicator(
                indicator="Forecast Confidence",
                status=conf_status,
                detail=conf_detail,
            ),
            HealthIndicator(
                indicator="Structural Breaks",
                status=breaks_status,
                detail=breaks_detail,
            ),
            HealthIndicator(
                indicator="Residual Diagnostics",
                status=resid_status,
                detail=resid_detail,
            ),
        ]

    # ── Forecast Metrics & Prediction Intervals ───────────────────────────

    @staticmethod
    def _forecast_pct_change(
        forecast: ForecastResult,
    ) -> tuple[float, float, float]:
        """Return (first_value, last_value, pct_change) for a forecast.

        The percentage change is guarded against a zero first value.

        Args:
            forecast: Forecast result.

        Returns:
            A tuple of (first_value, last_value, percentage_change).
        """
        first_val = forecast.forecast[0] if forecast.forecast else 0.0
        last_val = forecast.forecast[-1] if forecast.forecast else 0.0
        pct_change = (
            ((last_val - first_val) / abs(first_val)) * 100
            if first_val != 0
            else 0.0
        )
        return first_val, last_val, pct_change

    def _build_forecast_metrics(
        self,
        forecast: ForecastResult,
    ) -> ForecastMetrics:
        """Build forecast metrics with prediction intervals.

        Args:
            forecast: Forecast result.

        Returns:
            :class:`ForecastMetrics` with per-period prediction intervals.
        """
        first_val, last_val, pct_change = self._forecast_pct_change(forecast)
        first_date = (
            forecast.forecast_dates[0] if forecast.forecast_dates else "N/A"
        )
        last_date = (
            forecast.forecast_dates[-1] if forecast.forecast_dates else "N/A"
        )

        intervals: list[PredictionInterval] = []
        for i, date in enumerate(forecast.forecast_dates):
            lower = forecast.lower_ci[i] if i < len(forecast.lower_ci) else 0.0
            upper = forecast.upper_ci[i] if i < len(forecast.upper_ci) else 0.0
            point = (
                forecast.forecast[i] if i < len(forecast.forecast) else 0.0
            )
            intervals.append(
                PredictionInterval(
                    date=date,
                    forecast=round(point, 4),
                    lower_ci=round(lower, 4),
                    upper_ci=round(upper, 4),
                    confidence_level=_CONFIDENCE_LEVEL,
                )
            )

        return ForecastMetrics(
            model_used=forecast.model_used,
            horizon=len(forecast.forecast),
            first_date=first_date,
            last_date=last_date,
            first_value=round(first_val, 4),
            last_value=round(last_val, 4),
            pct_change=round(pct_change, 1),
            rmse=round(forecast.rmse, 4),
            mae=round(forecast.mae, 4),
            mape=round(forecast.mape, 2),
            prediction_intervals=intervals,
        )

    # ── Model Comparison ──────────────────────────────────────────────────

    def _build_model_comparison(
        self,
        all_metrics: dict[str, dict[str, float]],
        model_selection: ModelSelectionResult,
    ) -> ModelComparison:
        """Build the model comparison section from all model metrics.

        Args:
            all_metrics:      Dict of model → metrics.
            model_selection:  Model selection result.

        Returns:
            :class:`ModelComparison` with entries for each evaluated model.
        """
        selected = model_selection.selected_model
        rejection_map = {
            "Holt-Winters": model_selection.holt_winters_rejected_reason,
            "ARIMA": model_selection.arima_rejected_reason,
            "SARIMA": model_selection.sarima_rejected_reason,
            "EWMA": model_selection.ewma_rejected_reason,
            # Baseline models do not have rejection reasons from the selection
            # agent, as they are not candidates for the primary forecast.
            "Naive": "Baseline model for comparison.",
            "Seasonal Naive": "Baseline model for comparison.",
            "Mean Forecast": "Baseline model for comparison.",
            "Drift": "Baseline model for comparison.",
        }
        entries: list[ModelComparisonEntry] = []
        for name, metrics in all_metrics.items():
            entries.append(
                ModelComparisonEntry(
                    model=name,
                    rmse=round(metrics.get("RMSE", 0.0), 4),
                    mae=round(metrics.get("MAE", 0.0), 4),
                    mape=round(metrics.get("MAPE", 0.0), 2),
                    selected=(name == selected),
                    rejected_reason=(
                        rejection_map.get(name) if name != selected else None
                    ),
                )
            )
        return ModelComparison(
            entries=entries,
            selected_model=selected,
            selection_rationale=model_selection.explanation[:500],
        )

    # ── Recommendations ───────────────────────────────────────────────────

    def _build_recommendations(
        self,
        statistical: StatisticalResult,
        forecast: ForecastResult,
        review: StatisticalReviewResult | None,
        confidence: ConfidenceAssessment,
        data_quality: DataQualitySection,
        has_structural_breaks: bool = False,
    ) -> list[Recommendation]:
        """Build deterministic, evidence-backed recommendations.

        Recommendations are generated from business rules based on
        statistical signals — never from the LLM.  Each includes
        traceable :class:`EvidenceRef` objects.

        Args:
            statistical:          Statistical result.
            forecast:             Forecast result.
            review:               Statistical review result (optional).
            confidence:           Computed confidence assessment.
            data_quality:         Computed data quality section.
            has_structural_breaks: Precomputed flag indicating structural
                breaks were detected.

        Returns:
            List of :class:`Recommendation` objects.
        """
        recs: list[Recommendation] = []
        verdict = review.verdict if review else "pass"
        priority_map = RECOMMENDATION_PRIORITIES.get(
            verdict, RECOMMENDATION_PRIORITIES["pass"]
        )
        base_priority = priority_map.get(confidence.label, "Medium")

        # Recommendation 1: Validate forecast against actuals
        recs.append(
            Recommendation(
                priority=base_priority,
                recommendation=(
                    "Validate the forecast against next period's actuals "
                    "to confirm predictive accuracy before relying on it "
                    "for strategic decisions."
                ),
                rationale=(
                    "Forecast accuracy should be confirmed with out-of-sample "
                    "data before committing resources."
                ),
                supporting_evidence=[
                    EvidenceRef(
                        metric="MAPE",
                        value=f"{forecast.mape:.2f}%",
                        source_section="Forecast Reliability",
                    ),
                    EvidenceRef(
                        metric="Confidence Score",
                        value=f"{confidence.score}/100",
                        source_section="Forecast Reliability",
                    ),
                ],
                expected_outcome=(
                    "Confidence in the forecast's reliability for operational "
                    "planning will be established or adjustments identified."
                ),
            )
        )

        # Recommendation 2: Monitor structural breaks if detected
        if has_structural_breaks:
            recs.append(
                Recommendation(
                    priority="High",
                    recommendation=(
                        "Monitor for structural shifts in the data and "
                        "re-estimate the model if a regime change is detected."
                    ),
                    rationale=(
                        "Structural breaks were identified, which can "
                        "invalidate the current model's assumptions."
                    ),
                    supporting_evidence=[
                        EvidenceRef(
                            metric="Change Points",
                            value="Detected",
                            source_section="Statistical Analysis",
                        ),
                    ],
                    expected_outcome=(
                        "The forecast will remain valid even if the "
                        "underlying data pattern shifts."
                    ),
                )
            )

        # Recommendation 3: Data quality improvement
        if data_quality.rating != "Good":
            recs.append(
                Recommendation(
                    priority="Medium",
                    recommendation=(
                        "Improve data collection processes to reduce "
                        f"{data_quality.missing_values} missing values, "
                        f"{data_quality.duplicate_timestamps} duplicates, "
                        f"and {data_quality.missing_timestamps} gaps."
                    ),
                    rationale=(
                        "Data quality issues can materially influence "
                        "forecast reliability."
                    ),
                    supporting_evidence=[
                        EvidenceRef(
                            metric="Data Quality Rating",
                            value=data_quality.rating,
                            source_section="Data Quality Summary",
                        ),
                        EvidenceRef(
                            metric="Completeness",
                            value=f"{data_quality.completeness_pct:.1f}%",
                            source_section="Data Quality Summary",
                        ),
                    ],
                    expected_outcome=(
                        "Future forecasts will benefit from a cleaner, "
                        "more complete dataset."
                    ),
                )
            )

        # Recommendation 4: Seasonal capacity planning
        sp = statistical.seasonal_period
        if sp and sp > 1:
            recs.append(
                Recommendation(
                    priority="Medium",
                    recommendation=(
                        f"Align operational capacity planning with the "
                        f"detected {sp}-period seasonal cycle to prepare "
                        f"for predictable peak and trough periods."
                    ),
                    rationale=(
                        "A strong seasonal pattern was detected, creating "
                        "predictable demand cycles."
                    ),
                    supporting_evidence=[
                        EvidenceRef(
                            metric="Seasonal Period",
                            value=str(sp),
                            source_section="Historical Performance",
                        ),
                    ],
                    expected_outcome=(
                        "Operational readiness is expected to improve during "
                        "peak periods without over-provisioning during troughs."
                    ),
                )
            )

        # Recommendation 5: Rolling re-estimation
        recs.append(
            Recommendation(
                priority="Low",
                recommendation=(
                    f"Implement a rolling re-estimation of the forecast "
                    f"model on a {data_quality.frequency} basis to capture "
                    f"emerging shifts in the trend."
                ),
                rationale=(
                    "Regular re-estimation keeps the model aligned with "
                    "the latest data patterns."
                ),
                supporting_evidence=[
                    EvidenceRef(
                        metric="Trend Slope",
                        value=f"{statistical.trend_slope:.6f}",
                        source_section="Historical Performance",
                    ),
                ],
                expected_outcome=(
                    "The forecast will adapt to evolving patterns, "
                    "maintaining accuracy over time."
                ),
            )
        )

        return recs

    # ── Risks ─────────────────────────────────────────────────────────────

    def _build_risks(
        self,
        statistical: StatisticalResult,
        forecast: ForecastResult,
        review: StatisticalReviewResult | None,
        data_quality: DataQualitySection,
        has_structural_breaks: bool = False,
    ) -> list[Risk]:
        """Build strategic risks from statistical signals and review flags.

        Args:
            statistical:          Statistical result.
            forecast:             Forecast result.
            review:               Statistical review result (optional).
            data_quality:         Data quality section.
            has_structural_breaks: Precomputed flag indicating structural
                breaks were detected.

        Returns:
            List of :class:`Risk` objects.
        """
        risks: list[Risk] = []

        # Risk: High forecast uncertainty
        if forecast.mape > 20:
            risks.append(
                Risk(
                    category="Model",
                    description=(
                        f"Forecast validation error is high (MAPE "
                        f"{forecast.mape:.1f}%), indicating significant "
                        f"prediction uncertainty."
                    ),
                    potential_impact=(
                        "Decisions based on this forecast carry a wider "
                        "margin of error than is ideal for high-stakes planning."
                    ),
                    mitigation=(
                        "Use the prediction intervals for conservative "
                        "planning and validate against actuals before "
                        "committing to the central forecast."
                    ),
                    evidence=[
                        f"MAPE: {forecast.mape:.2f}%",
                        f"RMSE: {forecast.rmse:.4f}",
                    ],
                    severity="High",
                )
            )

        # Risk: Structural breaks
        if has_structural_breaks:
            risks.append(
                Risk(
                    category="Data",
                    description=(
                        "Structural breaks were detected, suggesting the "
                        "underlying data pattern may have shifted."
                    ),
                    potential_impact=(
                        "The current model may not accurately reflect "
                        "the new regime, leading to misleading projections."
                    ),
                    mitigation=(
                        "Segment the data by regime and re-estimate the "
                        "model on the most recent stable period."
                    ),
                    evidence=["Change point analysis detected structural breaks"],
                    severity="Medium",
                )
            )

        # Risk: Data quality
        if data_quality.rating == "Poor":
            risks.append(
                Risk(
                    category="Data",
                    description=(
                        "Data quality is poor, with significant gaps, "
                        "duplicates, or irregularities."
                    ),
                    potential_impact=(
                        "Forecast reliability is compromised by the "
                        "quality of the underlying data."
                    ),
                    mitigation=(
                        "Address data collection issues and re-run the "
                        "analysis once data quality improves."
                    ),
                    evidence=[
                        f"Data Quality Rating: {data_quality.rating}",
                        f"Missing values: {data_quality.missing_values}",
                    ],
                    severity="High",
                )
            )

        # Risk: Review concerns
        if review and review.verdict in ("warn", "fail"):
            concerns = self._review_concerns(review)
            risks.append(
                Risk(
                    category="Model",
                    description=(
                        "The independent statistical review identified "
                        f"{len(concerns)} concern(s) about the analysis."
                    ),
                    potential_impact=(
                        "Some aspects of the forecast may not be fully "
                        "supported by the evidence."
                    ),
                    mitigation=(
                        "Review the statistical audit summary and address "
                        "the recommended follow-up actions."
                    ),
                    evidence=concerns[:3] if concerns else [review.summary],
                    severity="High" if review.verdict == "fail" else "Medium",
                )
            )

        # Risk: Horizon decay (always present)
        risks.append(
            Risk(
                category="Model",
                description=(
                    "Forecast accuracy is expected to decline over longer "
                    "horizons — short-term projections are more reliable."
                ),
                potential_impact=(
                    "Long-term decisions based on distant forecast periods "
                    "carry higher uncertainty."
                ),
                mitigation=(
                    "Weight near-term projections more heavily in planning "
                    "and re-forecast as new data arrives."
                ),
                evidence=[
                    f"Forecast horizon: {len(forecast.forecast)} periods",
                ],
                severity="Low",
            )
        )

        return risks

    # ── Assumptions ───────────────────────────────────────────────────────

    def _build_assumptions(
        self,
        statistical: StatisticalResult,
        validation: ValidationResult,
    ) -> list[Assumption]:
        """Build critical business assumptions from statistical properties.

        Args:
            statistical: Statistical result.
            validation:  Validation result.

        Returns:
            List of :class:`Assumption` objects.
        """
        assumptions: list[Assumption] = []

        assumptions.append(
            Assumption(
                assumption=(
                    "The economic and operational drivers of the past will "
                    "persist — abrupt market shifts or policy changes are "
                    "not factored into this baseline."
                ),
                consequence_if_false=(
                    "A material change in the business environment would "
                    "render the current forecast obsolete."
                ),
            )
        )

        if statistical.is_stationary_adf and statistical.is_stationary_kpss:
            stationarity_note = (
                "The series is stationary, indicating a stable statistical "
                "structure."
            )
        else:
            stationarity_note = (
                "The series required transformation to achieve stationarity "
                "before modelling."
            )
        assumptions.append(
            Assumption(
                assumption=f"Statistical stability: {stationarity_note}",
                consequence_if_false=(
                    "If the statistical structure changes, the model's "
                    "underlying assumptions would no longer hold."
                ),
            )
        )

        sp = statistical.seasonal_period
        if sp and sp > 1:
            seasonal_note = (
                f"A seasonal cycle of {sp} periods is assumed to continue "
                f"predictably."
            )
        else:
            seasonal_note = (
                "No significant seasonality is assumed for this projection."
            )
        assumptions.append(
            Assumption(
                assumption=f"Seasonal stability: {seasonal_note}",
                consequence_if_false=(
                    "If seasonal patterns shift, the forecast would not "
                    "capture the new cyclical behaviour."
                ),
            )
        )

        assumptions.append(
            Assumption(
                assumption=(
                    f"Future data is assumed to arrive at the current "
                    f"{validation.frequency or 'unknown'} frequency."
                ),
                consequence_if_false=(
                    "A change in data frequency would require re-estimation "
                    "of the model."
                ),
            )
        )

        assumptions.append(
            Assumption(
                assumption=(
                    "The model operates solely on historical values; it "
                    "does not account for exogenous variables like "
                    "competitor activity or macro-economic indicators."
                ),
                consequence_if_false=(
                    "External factors not captured in the data could "
                    "materially alter the actual outcome."
                ),
            )
        )

        return assumptions

    # ── Statistical Audit ─────────────────────────────────────────────────

    @staticmethod
    def _review_concerns(
        review: StatisticalReviewResult | None,
    ) -> list[str]:
        """Extract concern issue strings from a statistical review.

        Filters review flags to those with critical or warning severity and
        returns their ``issue`` text.  Returns an empty list when the
        review is ``None``.

        Args:
            review: Statistical review result (optional).

        Returns:
            A list of concern issue strings.
        """
        if not review:
            return []
        return [
            f.get("issue", "")
            for f in review.flags
            if f.get("severity") in ("critical", "warning")
        ]

    def _build_statistical_audit(
        self,
        review: StatisticalReviewResult | None,
    ) -> StatisticalAudit:
        """Build the statistical audit section from the review result.

        Args:
            review: Statistical review result (optional).

        Returns:
            :class:`StatisticalAudit` with verdict, evidence, and concerns.
        """
        if not review:
            return StatisticalAudit(
                verdict="pass",
                strongest_evidence=[],
                key_concerns=[],
                recommended_follow_up=[
                    "No independent statistical review was performed."
                ],
            )

        strongest = list(review.endorsements)
        concerns = self._review_concerns(review)
        follow_up = [
            f.get("recommendation", "")
            for f in review.flags
            if f.get("recommendation")
        ]
        if not follow_up:
            follow_up = [
                "Validate the forecast against next period's actuals."
            ]

        return StatisticalAudit(
            verdict=review.verdict,
            strongest_evidence=strongest,
            key_concerns=concerns,
            recommended_follow_up=follow_up,
        )

    # ── Explainability ────────────────────────────────────────────────────

    def _build_explainability(
        self,
        statistical: StatisticalResult,
        forecast: ForecastResult,
        confidence: ConfidenceAssessment,
    ) -> Explainability:
        """Build the explainability section — why the AI reached its conclusions.

        Args:
            statistical: Statistical result.
            forecast:    Forecast result.
            confidence:  Computed confidence assessment.

        Returns:
            :class:`Explainability` with findings and interpretations.
        """
        items: list[ExplainabilityItem] = []

        sp = statistical.seasonal_period
        if sp and sp > 1:
            items.append(
                ExplainabilityItem(
                    finding=f"Strong seasonal pattern detected (every {sp} periods)",
                    evidence=f"Seasonal period: {sp}",
                    interpretation=(
                        "The data follows a predictable recurring cycle, "
                        "which the forecasting model captures to anticipate "
                        "periodic peaks and troughs."
                    ),
                )
            )

        if statistical.has_trend:
            direction = "upward" if statistical.trend_slope > 0 else "downward"
            items.append(
                ExplainabilityItem(
                    finding=f"Stable {direction} long-term trend identified",
                    evidence=f"Trend slope: {statistical.trend_slope:.6f}",
                    interpretation=(
                        f"The data shows a consistent {direction} trajectory "
                        f"over time, which the model projects forward."
                    ),
                )
            )

        if forecast.mape < 10:
            items.append(
                ExplainabilityItem(
                    finding="Low validation error",
                    evidence=f"MAPE: {forecast.mape:.2f}%",
                    interpretation=(
                        "The model's predictions were within a small margin "
                        "of actual historical values, supporting its reliability."
                    ),
                )
            )

        if not statistical.is_white_noise:
            items.append(
                ExplainabilityItem(
                    finding="Residual diagnostics indicate acceptable model fit",
                    evidence=f"White noise test: {'not random' if not statistical.is_white_noise else 'random'}",
                    interpretation=(
                        "The patterns in the data are not random noise — "
                        "the model is capturing meaningful structure."
                    ),
                )
            )

        items.append(
            ExplainabilityItem(
                finding=f"Overall forecast confidence: {confidence.label}",
                evidence=f"Confidence score: {confidence.score}/100",
                interpretation=confidence.explanation,
            )
        )

        return Explainability(findings=items)

    # ── Historical Analysis ───────────────────────────────────────────────

    def _build_historical_analysis(
        self,
        statistical: StatisticalResult,
    ) -> HistoricalAnalysis:
        """Build the historical analysis section (facts only).

        Args:
            statistical: Statistical result.

        Returns:
            :class:`HistoricalAnalysis` with empty narrative.
        """
        if statistical.trend_slope > 0:
            direction = FORECAST_DIRECTIONS["upward"]
        elif statistical.trend_slope < 0:
            direction = FORECAST_DIRECTIONS["downward"]
        else:
            direction = FORECAST_DIRECTIONS["flat"]
        is_stationary = (
            statistical.is_stationary_adf and statistical.is_stationary_kpss
        )
        return HistoricalAnalysis(
            trend_direction=direction,
            trend_slope=statistical.trend_slope,
            has_trend=statistical.has_trend,
            seasonal_period=statistical.seasonal_period,
            dominant_period=statistical.dominant_period,
            is_stationary=is_stationary,
        )

    # ── Dashboard ─────────────────────────────────────────────────────────

    def _build_dashboard(
        self,
        forecast: ForecastResult,
        statistical: StatisticalResult,
        model_selection: ModelSelectionResult,
        confidence: ConfidenceAssessment,
        data_quality: DataQualitySection,
        review: StatisticalReviewResult | None,
        has_structural_breaks: bool = False,
    ) -> Dashboard:
        """Build the dynamic dashboard as a list of reusable widgets.

        Args:
            forecast:             Forecast result.
            statistical:          Statistical result.
            model_selection:      Model selection result.
            confidence:           Confidence assessment.
            data_quality:         Data quality section.
            review:               Statistical review result (optional).
            has_structural_breaks: Precomputed flag indicating structural
                breaks were detected.

        Returns:
            :class:`Dashboard` with 7 :class:`DashboardItem` widgets.
        """
        first_val, last_val, pct_change = self._forecast_pct_change(forecast)
        direction, dir_status = self._direction_status(statistical.trend_slope)
        growth_str = f"{pct_change:+.1f}%"
        growth_status = self._growth_status(pct_change)
        conf_str = f"{confidence.score}/100 — {confidence.label}"
        conf_status = self._confidence_status(confidence.label)
        dq_status = self._quality_status(data_quality.rating)
        primary_risk, risk_status = self._primary_risk(
            review, data_quality, forecast, statistical, has_structural_breaks
        )
        action, action_status = self._recommended_action(review, data_quality)

        items = [
            DashboardItem(
                title="Forecast Direction",
                value=direction,
                status=dir_status,
                description=(
                    f"The metric is projected to trend {direction.lower()} "
                    f"over the {len(forecast.forecast)}-period horizon."
                ),
                icon="📈",
                priority=1,
            ),
            DashboardItem(
                title="Expected Growth",
                value=growth_str,
                status=growth_status,
                description=(
                    f"Projected change from {round(first_val, 2)} to "
                    f"{round(last_val, 2)} over the forecast horizon."
                ),
                icon="📊",
                priority=2,
            ),
            DashboardItem(
                title="Forecast Confidence",
                value=conf_str,
                status=conf_status,
                description=confidence.explanation,
                icon="🎯",
                priority=3,
            ),
            DashboardItem(
                title="Data Quality",
                value=data_quality.rating,
                status=dq_status,
                description=data_quality.rating_explanation,
                icon="🔍",
                priority=4,
            ),
            DashboardItem(
                title="Model Selected",
                value=model_selection.selected_model,
                status="info",
                description=(
                    "Selected based on validation performance and data "
                    "characteristics."
                ),
                icon="🤖",
                priority=5,
            ),
            DashboardItem(
                title="Primary Risk",
                value=primary_risk,
                status=risk_status,
                description=(
                    "The most significant risk identified from the analysis."
                ),
                icon="⚠️",
                priority=6,
            ),
            DashboardItem(
                title="Recommended Action",
                value=action,
                status=action_status,
                description=(
                    "Immediate action recommended for leadership."
                ),
                icon="✅",
                priority=7,
            ),
        ]
        return Dashboard(widgets=items)

    @staticmethod
    def _direction_status(slope: float) -> tuple[str, str]:
        """Return (direction label, status token) for a trend slope."""
        if slope > 0:
            return FORECAST_DIRECTIONS["upward"], "positive"
        if slope < 0:
            return FORECAST_DIRECTIONS["downward"], "negative"
        return FORECAST_DIRECTIONS["flat"], "neutral"

    @staticmethod
    def _growth_status(pct_change: float) -> str:
        """Return a status token for a growth percentage."""
        if pct_change > 0:
            return "positive"
        if pct_change < 0:
            return "negative"
        return "neutral"

    @staticmethod
    def _confidence_status(label: str) -> str:
        """Return a status token for a confidence label."""
        if label == "High":
            return "positive"
        if label == "Medium":
            return "warning"
        return "negative"

    @staticmethod
    def _quality_status(rating: str) -> str:
        """Return a status token for a data quality rating."""
        if rating == "Good":
            return "positive"
        if rating == "Fair":
            return "warning"
        return "negative"

    @staticmethod
    def _primary_risk(
        review: StatisticalReviewResult | None,
        data_quality: DataQualitySection,
        forecast: ForecastResult,
        statistical: StatisticalResult,
        has_structural_breaks: bool = False,
    ) -> tuple[str, str]:
        """Return (risk description, status token) for the dashboard."""
        if review and review.verdict == "fail":
            return _REVIEW_CRITICAL_MSG, "negative"
        if data_quality.rating == "Poor":
            return "Poor data quality may compromise reliability", "negative"
        if forecast.mape > 20:
            return "High forecast uncertainty (MAPE > 20%)", "warning"
        if has_structural_breaks:
            return (
                "Structural breaks detected — monitor for regime shifts",
                "warning",
            )
        return "Forecast accuracy may decline over longer horizons", "neutral"

    @staticmethod
    def _recommended_action(
        review: StatisticalReviewResult | None,
        data_quality: DataQualitySection,
    ) -> tuple[str, str]:
        """Return (action description, status token) for the dashboard."""
        if review and review.verdict in ("warn", "fail"):
            return (
                "Review statistical audit findings and validate forecast",
                "warning",
            )
        if data_quality.rating != "Good":
            return "Improve data quality and re-run analysis", "warning"
        return (
            "Use forecast for near-term planning; validate against actuals",
            "positive",
        )

    # ── Executive Summary ─────────────────────────────────────────────────

    def _build_executive_summary(
        self,
        forecast: ForecastResult,
        statistical: StatisticalResult,
        confidence: ConfidenceAssessment,
        data_quality: DataQualitySection,
        review: StatisticalReviewResult | None,
        has_structural_breaks: bool = False,
    ) -> ExecutiveSummary:
        """Build the executive summary structured fields (narrative left empty).

        Args:
            forecast:             Forecast result.
            statistical:          Statistical result.
            confidence:           Confidence assessment.
            data_quality:         Data quality section.
            review:               Statistical review result (optional).
            has_structural_breaks: Precomputed flag indicating structural
                breaks were detected.

        Returns:
            :class:`ExecutiveSummary` with empty narrative.
        """
        del has_structural_breaks  # Not used in this summary's risk wording.
        first_val, last_val, pct_change = self._forecast_pct_change(forecast)
        if statistical.trend_slope > 0:
            direction = "upward"
        elif statistical.trend_slope < 0:
            direction = "downward"
        else:
            direction = "flat"

        strategic_outlook = (
            f"The metric is projected to trend {direction} over the "
            f"{len(forecast.forecast)}-period horizon, moving from "
            f"{round(first_val, 2)} to {round(last_val, 2)}."
        )
        expected_growth = f"{pct_change:+.1f}% over the forecast horizon"
        confidence_level = f"{confidence.score}/100 — {confidence.label}"

        if review and review.verdict == "fail":
            primary_risk = _REVIEW_CRITICAL_MSG
        elif data_quality.rating == "Poor":
            primary_risk = "Poor data quality may compromise reliability"
        elif forecast.mape > 20:
            primary_risk = "High forecast uncertainty"
        else:
            primary_risk = "Forecast accuracy may decline over longer horizons"

        if review and review.verdict in ("warn", "fail"):
            recommended_action = (
                "Review the statistical audit findings and validate the "
                "forecast against actuals before strategic use."
            )
        else:
            recommended_action = (
                "Use the forecast for near-term planning and validate "
                "against next period's actuals."
            )

        return ExecutiveSummary(
            strategic_outlook=strategic_outlook,
            expected_growth=expected_growth,
            confidence_level=confidence_level,
            primary_risk=primary_risk,
            recommended_action=recommended_action,
        )

    # ── Metadata ──────────────────────────────────────────────────────────

    def _build_metadata(
        self,
        validation: ValidationResult,
        forecast: ForecastResult,
        model_selection: ModelSelectionResult,
        all_metrics: dict[str, dict[str, float]],
        data_quality: DataQualitySection,
    ) -> ReportMetadata:
        """Build report metadata.

        Args:
            validation:      Validation result.
            forecast:        Forecast result.
            model_selection: Model selection result.
            all_metrics:     All model metrics.
            data_quality:    Data quality section.

        Returns:
            :class:`ReportMetadata`.
        """
        return ReportMetadata(
            engine_version=_ENGINE_VERSION,
            generated_at=datetime.now(timezone.utc).isoformat(),
            forecast_horizon=len(forecast.forecast),
            models_evaluated=list(all_metrics.keys()),
            selected_model=model_selection.selected_model,
            dataset_frequency=validation.frequency or "unknown",
            data_quality_rating=data_quality.rating,
            row_count=validation.row_count,
        )

    # ── Appendix ──────────────────────────────────────────────────────────

    def _build_appendix(
        self,
        forecast: ForecastResult,
        all_metrics: dict[str, dict[str, float]],
    ) -> Appendix:
        """Build the appendix with raw metrics and visual tags.

        Args:
            forecast:    Forecast result.
            all_metrics: All model metrics.

        Returns:
            :class:`Appendix`.
        """
        raw_metrics: dict[str, Any] = {
            "rmse": round(forecast.rmse, 4),
            "mae": round(forecast.mae, 4),
            "mape": round(forecast.mape, 2),
            "mape_quality": mape_quality(forecast.mape),
            "all_models": {
                name: {
                    "RMSE": round(m.get("RMSE", 0.0), 4),
                    "MAE": round(m.get("MAE", 0.0), 4),
                    "MAPE": round(m.get("MAPE", 0.0), 2),
                }
                for name, m in all_metrics.items()
            },
        }
        visual_tags = ["HISTORICAL", "STL", "ACF_PACF", "FORECAST", "COMPARISON"]
        return Appendix(raw_metrics=raw_metrics, visual_tags=visual_tags)