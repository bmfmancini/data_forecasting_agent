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

import numpy as np

from report.models import (
    Appendix,
    Assumption,
    ConfidenceAssessment,
    Dashboard,
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
    format_metric,
)
from report.dashboard import build_dashboard
from report.rules import (
    CONFIDENCE_DEDUCTIONS,
    FORECAST_DIRECTIONS,
    HEALTH_STATUS,
    OUTLIER_REVIEW_RATIO_THRESHOLD,
    RECENT_HOLDOUT_RMSE_RATIO_THRESHOLD,
    RECOMMENDATION_PRIORITIES,
    confidence_label,
    data_quality_rating,
    mape_quality,
    recent_holdout_rmse_ratio,
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


def _recent_holdout_rmse_ratio(forecast: ForecastResult) -> float | None:
    """Return final-test RMSE divided by pooled rolling-origin RMSE."""
    final_rmse = forecast.final_test_metrics.get("rmse")
    pooled_rmse = forecast.selection_metrics.get("rmse")
    if not isinstance(pooled_rmse, (int, float)):
        pooled_rmse = forecast.rmse
    return recent_holdout_rmse_ratio(final_rmse, pooled_rmse)


def _has_usable_interval_bounds(forecast: ForecastResult) -> bool:
    """Return whether every dated point has finite lower and upper bounds."""
    horizon_dates = len(forecast.forecast_dates)
    if (
        forecast.interval_label == "unavailable"
        or horizon_dates == 0
        or len(forecast.forecast) < horizon_dates
        or len(forecast.lower_ci) < horizon_dates
        or len(forecast.upper_ci) < horizon_dates
    ):
        return False
    try:
        return all(
            np.isfinite(float(value))
            for value in (
                forecast.lower_ci[:horizon_dates]
                + forecast.upper_ci[:horizon_dates]
            )
        )
    except (TypeError, ValueError):
        return False


_REVIEW_CONCERN_MARKERS: dict[str, tuple[str, ...]] = {
    "mape": ("mape", "prediction error"),
    "recent_holdout": ("untouched holdout", "final-test", "final test"),
    "non_stationary": ("non-stationary", "nonstationary"),
    "white_noise": ("white noise",),
    "outliers": ("outlier", "anomal"),
    "missing_data": ("missing value", "missing timestamp", "data gap"),
    "structural_breaks": ("structural break", "change point"),
}


def _review_has_distinct_concern(
    review: StatisticalReviewResult,
    scored_concerns: set[str],
) -> bool:
    """Return whether review flags add a concern not already scored."""
    if not review.flags:
        return True
    for flag in review.flags:
        issue = str(flag.get("issue", "")).lower()
        matched = {
            concern
            for concern, markers in _REVIEW_CONCERN_MARKERS.items()
            if any(marker in issue for marker in markers)
        }
        if not matched or not matched.issubset(scored_concerns):
            return True
    return False


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
            forecast,
            statistical,
            validation,
            statistical_review,
            has_structural_breaks,
        )
        data_quality = self._compute_data_quality(validation, statistical)
        health_indicators = self._compute_health_indicators(
            statistical,
            validation,
            forecast,
            statistical_review,
            confidence,
            data_quality,
            has_structural_breaks,
        )
        forecast_metrics = self._build_forecast_metrics(forecast)
        model_comparison = self._build_model_comparison(
            all_metrics, model_selection, forecast
        )
        recommendations = self._build_recommendations(
            statistical,
            forecast,
            statistical_review,
            confidence,
            data_quality,
            has_structural_breaks,
        )
        risks = self._build_risks(
            statistical,
            forecast,
            statistical_review,
            data_quality,
            has_structural_breaks,
        )
        assumptions = self._build_assumptions(statistical, validation, forecast)
        explainability = self._build_explainability(statistical, forecast, confidence)
        statistical_audit = self._build_statistical_audit(statistical_review)
        historical = self._build_historical_analysis(statistical)
        forecast_outlook = ForecastOutlook(metrics=forecast_metrics)
        dashboard = self._build_dashboard(
            forecast,
            statistical,
            model_selection,
            confidence,
            data_quality,
            statistical_review,
            has_structural_breaks,
        )
        executive_summary = self._build_executive_summary(
            forecast,
            statistical,
            confidence,
            data_quality,
            statistical_review,
            has_structural_breaks,
        )
        metadata = self._build_metadata(
            validation,
            forecast,
            model_selection,
            all_metrics,
            data_quality,
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
        scored_concerns: set[str] = set()

        if forecast.mape is not None and forecast.mape > 20:
            score -= CONFIDENCE_DEDUCTIONS["mape_above_20"]
            factors.append(f"High validation error (MAPE {forecast.mape:.1f}%)")
            scored_concerns.add("mape")
        elif forecast.mape is not None and forecast.mape > 10:
            score -= CONFIDENCE_DEDUCTIONS["mape_above_10"]
            factors.append(f"Moderate validation error (MAPE {forecast.mape:.1f}%)")
            scored_concerns.add("mape")
        elif forecast.mape is not None and forecast.mape > 5:
            score -= CONFIDENCE_DEDUCTIONS["mape_above_5"]
            factors.append(f"Minor validation error (MAPE {forecast.mape:.1f}%)")
            scored_concerns.add("mape")

        holdout_ratio = _recent_holdout_rmse_ratio(forecast)
        if (
            holdout_ratio is not None
            and holdout_ratio >= RECENT_HOLDOUT_RMSE_RATIO_THRESHOLD
        ):
            score -= CONFIDENCE_DEDUCTIONS["recent_holdout_degradation"]
            factors.append(
                f"Recent untouched-holdout RMSE is {holdout_ratio:.2f}× pooled "
                "rolling-origin RMSE (material degradation threshold: "
                f"{RECENT_HOLDOUT_RMSE_RATIO_THRESHOLD:.2f}×)"
            )
            scored_concerns.add("recent_holdout")

        if not statistical.is_stationary_adf:
            score -= CONFIDENCE_DEDUCTIONS["non_stationary_adf"]
            factors.append("Series is non-stationary")
            scored_concerns.add("non_stationary")

        if statistical.is_white_noise:
            score -= CONFIDENCE_DEDUCTIONS["white_noise"]
            factors.append("Series resembles random noise")
            scored_concerns.add("white_noise")

        if statistical.outlier_ratio > OUTLIER_REVIEW_RATIO_THRESHOLD:
            score -= CONFIDENCE_DEDUCTIONS["outlier_ratio_high"]
            factors.append(
                f"Outlier ratio {statistical.outlier_ratio:.1%} exceeds the "
                f"{OUTLIER_REVIEW_RATIO_THRESHOLD:.0%} review threshold"
            )
            scored_concerns.add("outliers")

        if validation.missing_values > 0 or validation.missing_timestamps > 0:
            score -= CONFIDENCE_DEDUCTIONS["missing_data"]
            factors.append("Missing values or gaps in the data")
            scored_concerns.add("missing_data")

        if has_structural_breaks:
            score -= CONFIDENCE_DEDUCTIONS["structural_breaks"]
            factors.append("Detected change points may indicate a structural break")
            scored_concerns.add("structural_breaks")

        if review and _review_has_distinct_concern(review, scored_concerns):
            if review.verdict == "warn":
                score -= CONFIDENCE_DEDUCTIONS["review_warn"]
                factors.append("Statistical review raised warnings")
            elif review.verdict == "fail":
                score -= CONFIDENCE_DEDUCTIONS["review_fail"]
                factors.append(_REVIEW_CRITICAL_MSG)

        score = max(0, min(100, score))
        label = confidence_label(score)

        if not factors:
            factors.append("Low validation error")
            factors.append("Stable statistical properties")

        top_factors = factors[:2]
        explanation = (
            f"Confidence is {label.lower()} based on: " + "; ".join(top_factors) + "."
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
        collection_rating = data_quality_rating(
            validation.missing_values,
            validation.duplicate_timestamps,
            validation.missing_timestamps,
            issues_count,
            validation.is_regular,
        )
        rating = data_quality_rating(
            validation.missing_values,
            validation.duplicate_timestamps,
            validation.missing_timestamps,
            issues_count,
            validation.is_regular,
            statistical.outlier_ratio,
        )
        total_possible = validation.row_count + validation.missing_timestamps
        completeness = (
            (validation.row_count / total_possible * 100)
            if total_possible > 0
            else 100.0
        )
        collection_counts = (
            f"{validation.missing_values} missing values, "
            f"{validation.duplicate_timestamps} duplicate timestamps, and "
            f"{validation.missing_timestamps} gaps; "
            f"{issues_count} validation issue{'s' if issues_count != 1 else ''}"
        )
        if validation.issues:
            collection_counts += f" ({'; '.join(validation.issues)})"
        if collection_rating == "Good":
            collection_explanation = (
                "Collection quality is good under the completeness and regularity "
                f"policy: {collection_counts}; intervals are regular."
            )
        elif collection_rating == "Fair":
            collection_explanation = (
                "Collection quality is fair under the completeness and regularity "
                f"policy: {collection_counts}; interval regularity is "
                f"{'satisfied' if validation.is_regular else 'not satisfied'}."
            )
        else:
            collection_explanation = (
                "Collection quality is poor under the completeness and regularity "
                f"policy: {collection_counts}; interval regularity is "
                f"{'satisfied' if validation.is_regular else 'not satisfied'}."
            )

        if statistical.outlier_ratio > OUTLIER_REVIEW_RATIO_THRESHOLD:
            anomaly_explanation = (
                f"Anomaly risk requires review: {statistical.outlier_count} detected "
                f"values ({statistical.outlier_ratio:.1%}) exceed the "
                f"{OUTLIER_REVIEW_RATIO_THRESHOLD:.0%} review threshold"
                + (
                    ", limiting the overall rating to Fair."
                    if collection_rating == "Good"
                    else "."
                )
            )
        elif statistical.outlier_count:
            anomaly_explanation = (
                f"Anomaly screening found {statistical.outlier_count} values "
                f"({statistical.outlier_ratio:.1%}), which does not exceed the "
                f"{OUTLIER_REVIEW_RATIO_THRESHOLD:.0%} review threshold; this "
                "threshold comparison does not establish that individual anomalies "
                "are harmless."
            )
        else:
            anomaly_explanation = "Anomaly screening found no flagged values."
        explanation = f"{collection_explanation} {anomaly_explanation}"

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
            trend_detail = (
                "A statistically significant trend is changing the baseline over time."
            )
        else:
            trend_status = HEALTH_STATUS["trend_stability"]["stable"]
            trend_detail = "The underlying trend is stable across the observed period."

        # Seasonality
        sp = statistical.seasonal_period
        if sp and sp > 1:
            seasonality_status = HEALTH_STATUS["seasonality"]["strong"]
            seasonality_detail = (
                f"A recurring seasonal pattern repeats every {sp} periods."
            )
        else:
            seasonality_status = HEALTH_STATUS["seasonality"]["none"]
            seasonality_detail = "No strong seasonal pattern was detected."

        # Forecast Confidence
        conf_status = confidence.label
        conf_detail = confidence.explanation

        # Structural Breaks
        if has_structural_breaks:
            breaks_status = HEALTH_STATUS["structural_breaks"]["monitor"]
            breaks_detail = (
                "Candidate change points require validation of break dates, effect "
                "sizes, and persistence."
            )
        else:
            breaks_status = HEALTH_STATUS["structural_breaks"]["none"]
            breaks_detail = "No structural breaks detected."

        # Residual Diagnostics
        diag = forecast.residual_diagnostics
        if diag and diag.is_uncorrelated is False:
            resid_status = HEALTH_STATUS["residual_diagnostics"]["concerning"]
            resid_detail = "Residuals are autocorrelated, indicating the model has not captured all predictable patterns."
        elif diag and diag.is_normal is False:
            resid_status = HEALTH_STATUS["residual_diagnostics"]["concerning"]
            resid_detail = "Residuals are not normally distributed, which may affect the reliability of prediction intervals."
        elif statistical.is_white_noise or (
            forecast.mape is not None and forecast.mape > 20
        ):
            resid_status = HEALTH_STATUS["residual_diagnostics"]["concerning"]
            resid_detail = "High validation error or other signals suggest the model may not fully capture the data structure."
        else:
            resid_status = HEALTH_STATUS["residual_diagnostics"]["acceptable"]
            resid_detail = "Residual analysis confirms the model's errors are random and unbiased, indicating a good fit."

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
            ((last_val - first_val) / abs(first_val)) * 100 if first_val != 0 else 0.0
        )
        return first_val, last_val, pct_change

    @staticmethod
    def _forecast_pattern(forecast: ForecastResult) -> str:
        """Classify the plotted path without confusing endpoints with trend."""
        values = np.asarray(forecast.forecast, dtype=float)
        if values.size < 2:
            return "Flat"
        changes = np.diff(values)
        tolerance = max(float(np.nanmax(np.abs(values))) * 1e-9, 1e-12)
        if np.all(changes >= -tolerance):
            return "Upward"
        if np.all(changes <= tolerance):
            return "Downward"
        return "Seasonal / variable"

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
        forecast_pattern = self._forecast_pattern(forecast)
        if pct_change > 0:
            endpoint_direction = "Upward"
        elif pct_change < 0:
            endpoint_direction = "Downward"
        else:
            endpoint_direction = "Flat"
        peak_value = max(forecast.forecast) if forecast.forecast else None
        peak_index = (
            forecast.forecast.index(peak_value) if peak_value is not None else -1
        )
        peak_date = (
            forecast.forecast_dates[peak_index]
            if 0 <= peak_index < len(forecast.forecast_dates)
            else None
        )
        peak_change_pct = (
            ((peak_value - first_val) / abs(first_val)) * 100
            if peak_value is not None and first_val != 0
            else None
        )
        first_date = forecast.forecast_dates[0] if forecast.forecast_dates else "N/A"
        last_date = forecast.forecast_dates[-1] if forecast.forecast_dates else "N/A"

        # Carry the technical interval label while report prose uses conservative
        # model-based/estimated language. Missing or partial bounds are unavailable;
        # never fabricate zero-valued intervals.
        interval_label = getattr(forecast, "interval_label", "prediction_interval")
        bounds_available = _has_usable_interval_bounds(forecast)
        if not bounds_available:
            interval_label = "unavailable"
        confidence_label = (
            "95% (experimental)"
            if interval_label == "experimental"
            else _CONFIDENCE_LEVEL
        )

        intervals: list[PredictionInterval] = []
        if bounds_available:
            for i, date in enumerate(forecast.forecast_dates):
                intervals.append(
                    PredictionInterval(
                        date=date,
                        forecast=round(forecast.forecast[i], 4),
                        lower_ci=round(forecast.lower_ci[i], 4),
                        upper_ci=round(forecast.upper_ci[i], 4),
                        confidence_level=confidence_label,
                        interval_label=interval_label,
                    )
                )

        final_rmse = forecast.final_test_metrics.get("rmse")
        final_test_assessment = None
        ratio = _recent_holdout_rmse_ratio(forecast)
        if isinstance(final_rmse, (int, float)) and ratio is not None:
            if ratio >= RECENT_HOLDOUT_RMSE_RATIO_THRESHOLD:
                final_test_assessment = (
                    f"Recent untouched final-test RMSE was {ratio:.2f}× the "
                    "pooled rolling-origin RMSE, indicating weaker performance "
                    "on the most recent holdout."
                )
            elif ratio <= 0.8:
                final_test_assessment = (
                    f"Recent untouched final-test RMSE was {ratio:.2f}× the "
                    "pooled rolling-origin RMSE, indicating stronger performance "
                    "on the most recent holdout."
                )
            else:
                final_test_assessment = (
                    f"Recent untouched final-test RMSE was {ratio:.2f}× the "
                    "pooled rolling-origin RMSE, broadly consistent with the "
                    "rolling validation evidence."
                )

        return ForecastMetrics(
            model_used=forecast.model_used,
            horizon=len(forecast.forecast),
            first_date=first_date,
            last_date=last_date,
            first_value=round(first_val, 4),
            last_value=round(last_val, 4),
            pct_change=round(pct_change, 1),
            endpoint_direction=endpoint_direction,
            forecast_pattern=forecast_pattern,
            peak_value=round(peak_value, 4) if peak_value is not None else None,
            peak_date=peak_date,
            peak_change_pct=(
                round(peak_change_pct, 1) if peak_change_pct is not None else None
            ),
            rmse=round(forecast.rmse, 4) if forecast.rmse is not None else None,
            mae=round(forecast.mae, 4) if forecast.mae is not None else None,
            mape=round(forecast.mape, 2) if forecast.mape is not None else None,
            wape=round(forecast.wape, 2) if forecast.wape is not None else None,
            mase=round(forecast.mase, 4) if forecast.mase is not None else None,
            interval_label=interval_label,
            prediction_intervals=intervals,
            selection_metrics=forecast.selection_metrics,
            final_test_metrics=forecast.final_test_metrics,
            final_test_assessment=final_test_assessment,
        )

    # ── Model Comparison ──────────────────────────────────────────────────

    def _build_model_comparison(
        self,
        all_metrics: dict[str, dict[str, float]],
        model_selection: ModelSelectionResult,
        forecast: ForecastResult | None = None,
    ) -> ModelComparison:
        """Build the model comparison section from all model metrics.

        Args:
            all_metrics:      Dict of model → metrics.
            model_selection:  Model selection result.

        Returns:
            :class:`ModelComparison` with entries for each evaluated model.
        """
        selected = (
            forecast.model_used
            if forecast is not None
            else model_selection.selected_model
        )
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
            rmse = metrics.get("RMSE")
            mae = metrics.get("MAE")
            mape = metrics.get("MAPE")
            wape = metrics.get("WAPE")
            mase = metrics.get("MASE")
            entries.append(
                ModelComparisonEntry(
                    model=name,
                    rmse=(
                        round(rmse, 4)
                        if rmse is not None and np.isfinite(rmse)
                        else None
                    ),
                    mae=round(mae, 4) if mae is not None and np.isfinite(mae) else None,
                    mape=(
                        round(mape, 2)
                        if mape is not None and np.isfinite(mape)
                        else None
                    ),
                    wape=(
                        round(wape * 100, 2)
                        if wape is not None and np.isfinite(wape)
                        else None
                    ),
                    mase=(
                        round(mase, 4)
                        if mase is not None and np.isfinite(mase)
                        else None
                    ),
                    selected=(name == selected),
                    rejected_reason=(
                        rejection_map.get(name) if name != selected else None
                    ),
                )
            )
        return ModelComparison(
            entries=entries,
            selected_model=selected,
            selection_rationale=self._selection_rationale(
                selected, all_metrics.get(selected, {}), model_selection
            ),
        )

    @staticmethod
    def _selection_rationale(
        selected: str,
        metrics: dict[str, float],
        model_selection: ModelSelectionResult,
    ) -> str:
        """Return a conservative rationale grounded in production evidence."""
        available = []
        for name in ("MASE", "RMSE", "MAE"):
            value = metrics.get(name)
            if value is not None and np.isfinite(value):
                available.append(f"{name} {value:.4f}")
        evidence = ", ".join(available) or "available rolling-origin evidence"
        method = model_selection.selection_method or "deterministic"
        rationale = (
            f"{selected} is the production forecast model. Its reported selection "
            f"evidence includes {evidence}. The {method} decision also applies "
            "candidate eligibility, configured loss, tie-breaking, baseline "
            "retention, and any typed review constraints; the smallest value in "
            "one displayed metric alone does not necessarily determine selection."
        )
        decision_loss = model_selection.selection_evidence.get("decision_loss", {})
        resolved = decision_loss.get("resolved")
        if resolved:
            rationale += f" Decision loss: {str(resolved).upper()}."
        if decision_loss.get("selection_sensitive"):
            rationale += (
                " Sensitivity warning: another supported loss metric selects a "
                "different model."
            )
        return rationale[:500]

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

        # Recommendation 1: Monitor future actuals after completed validation
        holdout_ratio = _recent_holdout_rmse_ratio(forecast)
        final_test_rmse = forecast.final_test_metrics.get("rmse")
        has_final_test = isinstance(final_test_rmse, (int, float)) and np.isfinite(
            final_test_rmse
        )
        if (
            holdout_ratio is not None
            and holdout_ratio >= RECENT_HOLDOUT_RMSE_RATIO_THRESHOLD
        ):
            monitoring_action = (
                "Monitor forecast performance against future actuals and reassess "
                "the model if the recent weakening persists."
            )
            monitoring_rationale = (
                f"The untouched final-test RMSE was {holdout_ratio:.2f}× the pooled "
                "rolling-origin RMSE, at or above the material-degradation "
                "threshold of "
                f"{RECENT_HOLDOUT_RMSE_RATIO_THRESHOLD:.2f}×."
            )
        elif has_final_test:
            monitoring_action = (
                "Continue monitoring forecast performance against future actuals "
                "to detect drift beyond the completed rolling-origin and untouched "
                "final-test validation."
            )
            monitoring_rationale = (
                "Out-of-sample validation has been completed; future actuals provide "
                "new evidence about whether performance remains stable."
            )
        else:
            monitoring_action = (
                "Monitor forecast performance against future actuals before relying "
                "on it for high-impact strategic decisions."
            )
            monitoring_rationale = (
                "Future actuals provide additional evidence about forecast "
                "performance as operating conditions evolve."
            )
        monitoring_evidence = [
            EvidenceRef(
                metric="MAPE",
                value=(
                    f"{format_metric(forecast.mape, '.2f')}%"
                    if forecast.mape is not None
                    else "not available"
                ),
                source_section="Forecast Reliability",
            ),
            EvidenceRef(
                metric="Confidence Score",
                value=f"{confidence.score}/100",
                source_section="Forecast Reliability",
            ),
        ]
        if holdout_ratio is not None:
            monitoring_evidence.append(
                EvidenceRef(
                    metric="Final-Test / Rolling-Origin RMSE",
                    value=f"{holdout_ratio:.2f}×",
                    source_section="Forecast Outlook",
                )
            )
        recs.append(
            Recommendation(
                priority=base_priority,
                recommendation=monitoring_action,
                rationale=monitoring_rationale,
                supporting_evidence=monitoring_evidence,
                expected_outcome=(
                    "Ongoing monitoring can show whether recent performance "
                    "stabilizes or a model adjustment is warranted."
                ),
            )
        )

        # Recommendation 2: Monitor structural breaks if detected
        if has_structural_breaks:
            recs.append(
                Recommendation(
                    priority="High",
                    recommendation=(
                        "Validate the candidate break dates, effect sizes, and "
                        "persistence. Only if a durable break is confirmed, compare "
                        "intervention terms, recency weighting, segmentation, and "
                        "regime-specific models."
                    ),
                    rationale=(
                        "Detected change points can reflect transient anomalies or "
                        "persistent shifts; current evidence does not establish which."
                    ),
                    supporting_evidence=[
                        EvidenceRef(
                            metric="Change Points",
                            value="Candidates detected",
                            source_section="Statistical Analysis",
                        ),
                    ],
                    expected_outcome=(
                        "The follow-up will determine whether a modelling adjustment "
                        "is warranted and which option is supported by evidence."
                    ),
                )
            )

        # Recommendation 3: Data quality improvement
        has_collection_issue = any(
            (
                data_quality.missing_values,
                data_quality.duplicate_timestamps,
                data_quality.missing_timestamps,
            )
        ) or not data_quality.is_regular
        if has_collection_issue:
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
        if forecast.mape is not None and forecast.mape > 20:
            if not _has_usable_interval_bounds(forecast):
                interval_mitigation = (
                    "Prediction-interval bounds are unavailable; review the "
                    "untouched holdout and monitor performance against future "
                    "actuals without inferring a 95% planning range."
                )
            elif forecast.interval_label == "experimental":
                interval_mitigation = (
                    "Use the estimated 95% prediction intervals (coverage not "
                    "evaluated) for scenario planning, review the untouched holdout, "
                    "and monitor performance against future actuals."
                )
            else:
                interval_mitigation = (
                    "Use the model-based 95% prediction intervals for conservative "
                    "planning, review the untouched holdout, and monitor performance "
                    "against future actuals."
                )
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
                    mitigation=interval_mitigation,
                    evidence=[
                        f"MAPE: {forecast.mape:.2f}%",
                        f"RMSE: {format_metric(forecast.rmse, '.4f')}",
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
                        "Change-point analysis identified candidate breaks that may "
                        "indicate a structural shift."
                    ),
                    potential_impact=(
                        "If a break is validated and persists, a model fitted across "
                        "differing regimes may produce misleading projections."
                    ),
                    mitigation=(
                        "First validate the candidate break dates, effect sizes, and "
                        "persistence. If confirmed, compare intervention terms, "
                        "recency weighting, segmentation, and regime-specific models "
                        "before selecting an adjustment."
                    ),
                    evidence=["Change-point analysis identified candidate breaks"],
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
        forecast: ForecastResult,
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

        model = forecast.model_used.lower()
        if model in {"holt-winters", "holt winters", "ewma"}:
            stationarity_note = (
                "The historical level, trend, and seasonal structure are assumed "
                f"to remain sufficiently stable for {forecast.model_used}."
            )
        elif model in {"arima", "sarima"}:
            stationarity_note = (
                "The differenced dependence structure is assumed to remain "
                f"sufficiently stable for {forecast.model_used}."
            )
        elif statistical.is_stationary_adf and statistical.is_stationary_kpss:
            stationarity_note = "The observed statistical structure remains stable."
        else:
            stationarity_note = (
                "The historical pattern is assumed to remain sufficiently stable "
                "over the forecast horizon."
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
            seasonal_note = "No significant seasonality is assumed for this projection."
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
            f.get("recommendation", "") for f in review.flags if f.get("recommendation")
        ]
        if not follow_up:
            follow_up = ["Monitor forecast performance against future actuals."]

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

        if forecast.mape is not None and forecast.mape < 10:
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

        residuals = forecast.residual_diagnostics
        if residuals is not None and residuals.is_uncorrelated is True:
            items.append(
                ExplainabilityItem(
                    finding="Residual diagnostics indicate acceptable model fit",
                    evidence="Residual autocorrelation test: no significant dependence",
                    interpretation=(
                        "The remaining forecast errors do not show significant "
                        "serial dependence."
                    ),
                )
            )
        elif residuals is not None and residuals.is_uncorrelated is False:
            items.append(
                ExplainabilityItem(
                    finding="Residual diagnostics require monitoring",
                    evidence=(
                        "Residual autocorrelation detected"
                        + (
                            f" (Ljung-Box p={residuals.ljung_box_p_value:.4f})"
                            if residuals.ljung_box_p_value is not None
                            else ""
                        )
                    ),
                    interpretation=(
                        "Some predictable structure remains in the forecast "
                        "errors, so model performance should be monitored."
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
        is_stationary = statistical.is_stationary_adf and statistical.is_stationary_kpss
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
        return build_dashboard(
            forecast=forecast,
            trend_slope=statistical.trend_slope,
            model_selection=model_selection,
            confidence=confidence,
            data_quality=data_quality,
            review=review,
            forecast_change=self._forecast_pct_change(forecast),
            has_structural_breaks=has_structural_breaks,
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
        del statistical  # Historical direction is reported in its own section.
        first_val, last_val, pct_change = self._forecast_pct_change(forecast)
        pattern = self._forecast_pattern(forecast).lower()

        strategic_outlook = (
            f"The forecast follows a {pattern} path over the "
            f"{len(forecast.forecast)}-period horizon and ends at "
            f"{round(last_val, 2)}, compared with {round(first_val, 2)} in "
            "the first forecast period."
        )
        expected_growth = (
            f"{pct_change:+.1f}% from the first forecast period to the last"
        )
        confidence_level = f"{confidence.score}/100 — {confidence.label}"

        if review and review.verdict == "fail":
            primary_risk = _REVIEW_CRITICAL_MSG
        elif data_quality.rating == "Poor":
            primary_risk = "Poor data quality may compromise reliability"
        elif forecast.mape is not None and forecast.mape > 20:
            primary_risk = "High forecast uncertainty"
        else:
            primary_risk = "Forecast accuracy may decline over longer horizons"

        holdout_ratio = _recent_holdout_rmse_ratio(forecast)
        if (
            holdout_ratio is not None
            and holdout_ratio >= RECENT_HOLDOUT_RMSE_RATIO_THRESHOLD
        ):
            recommended_action = (
                "Monitor performance against future actuals because the latest "
                f"untouched holdout RMSE was {holdout_ratio:.2f}× the pooled "
                "rolling-origin RMSE."
            )
        elif review and review.verdict in ("warn", "fail"):
            recommended_action = (
                "Review the statistical audit findings and monitor forecast "
                "performance against future actuals."
            )
        else:
            recommended_action = (
                "Use the forecast for near-term planning and monitor performance "
                "against future actuals."
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
        del model_selection  # ForecastResult is authoritative after final selection.
        return ReportMetadata(
            engine_version=_ENGINE_VERSION,
            generated_at=datetime.now(timezone.utc).isoformat(),
            forecast_horizon=len(forecast.forecast),
            models_evaluated=list(all_metrics.keys()),
            selected_model=forecast.model_used,
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
            "wape": round(forecast.wape, 2) if forecast.wape is not None else None,
            "mase": round(forecast.mase, 4) if forecast.mase is not None else None,
            "mape_quality": mape_quality(forecast.mape),
            "all_models": {
                name: {
                    "RMSE": round(m.get("RMSE", 0.0), 4),
                    "MAE": round(m.get("MAE", 0.0), 4),
                    "MAPE": round(m.get("MAPE", 0.0), 2),
                    "WAPE": round(m.get("WAPE", 0.0) * 100, 2),
                    "MASE": round(m.get("MASE", 0.0), 4),
                }
                for name, m in all_metrics.items()
            },
        }
        visual_tags = [
            "HISTORICAL",
            "STL",
            "ACF_PACF",
            "FORECAST",
            "COMPARISON",
            "RESIDUALS",
        ]
        return Appendix(raw_metrics=raw_metrics, visual_tags=visual_tags)
