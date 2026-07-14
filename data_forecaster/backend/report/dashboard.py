"""Dashboard widget rules for deterministic executive reports."""

from __future__ import annotations

from report.models import (
    ConfidenceAssessment,
    Dashboard,
    DashboardItem,
    DataQualitySection,
)
from report.rules import (
    FORECAST_DIRECTIONS,
    RECENT_HOLDOUT_RMSE_RATIO_THRESHOLD,
    recent_holdout_rmse_ratio,
)
from schemas import ForecastResult, ModelSelectionResult, StatisticalReviewResult

_REVIEW_CRITICAL_MSG = "Statistical review identified critical issues"


def build_dashboard(
    forecast: ForecastResult,
    trend_slope: float,
    model_selection: ModelSelectionResult,
    confidence: ConfidenceAssessment,
    data_quality: DataQualitySection,
    review: StatisticalReviewResult | None,
    forecast_change: tuple[float, float, float],
    has_structural_breaks: bool = False,
) -> Dashboard:
    """Build reusable dashboard widgets for the executive report."""
    del model_selection  # ForecastResult is authoritative after retries/fallbacks.
    first_val, last_val, pct_change = forecast_change
    del trend_slope  # Historical trend is reported separately from forecast direction.
    pattern, dir_status = forecast_pattern_status(forecast.forecast)
    risk_label, risk_status = primary_risk(
        review, data_quality, forecast, has_structural_breaks
    )
    action, action_status = recommended_action(review, data_quality, forecast)

    return Dashboard(
        widgets=[
            DashboardItem(
                title="Forecast Pattern",
                value=pattern,
                status=dir_status,
                description=(
                    f"The plotted forecast follows a {pattern.lower()} path "
                    f"over the {len(forecast.forecast)}-period horizon."
                ),
                icon="📈",
                priority=1,
            ),
            DashboardItem(
                title="Forecast Endpoint Change",
                value=f"{pct_change:+.1f}%",
                status=growth_status(pct_change),
                description=(
                    f"Endpoint change from {round(first_val, 2)} to "
                    f"{round(last_val, 2)} over the forecast horizon."
                ),
                icon="📊",
                priority=2,
            ),
            DashboardItem(
                title="Forecast Confidence",
                value=f"{confidence.score}/100 — {confidence.label}",
                status=confidence_status(confidence.label),
                description=confidence.explanation,
                icon="🎯",
                priority=3,
            ),
            DashboardItem(
                title="Data Quality",
                value=data_quality.rating,
                status=quality_status(data_quality.rating),
                description=data_quality.rating_explanation,
                icon="🔍",
                priority=4,
            ),
            DashboardItem(
                title="Model Selected",
                value=forecast.model_used,
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
                value=risk_label,
                status=risk_status,
                description=("The most significant risk identified from the analysis."),
                icon="⚠️",
                priority=6,
            ),
            DashboardItem(
                title="Recommended Action",
                value=action,
                status=action_status,
                description="Immediate action recommended for leadership.",
                icon="✅",
                priority=7,
            ),
        ]
    )


def direction_status(endpoint_change_pct: float) -> tuple[str, str]:
    """Return direction from first-to-last forecast change."""
    if endpoint_change_pct > 0:
        return FORECAST_DIRECTIONS["upward"], "positive"
    if endpoint_change_pct < 0:
        return FORECAST_DIRECTIONS["downward"], "negative"
    return FORECAST_DIRECTIONS["flat"], "neutral"


def forecast_pattern_status(values: list[float]) -> tuple[str, str]:
    """Classify monotonic direction separately from a variable seasonal path."""
    if len(values) < 2:
        return FORECAST_DIRECTIONS["flat"], "neutral"
    changes = [current - previous for previous, current in zip(values, values[1:])]
    tolerance = max(max(abs(value) for value in values) * 1e-9, 1e-12)
    if all(change >= -tolerance for change in changes):
        return FORECAST_DIRECTIONS["upward"], "positive"
    if all(change <= tolerance for change in changes):
        return FORECAST_DIRECTIONS["downward"], "negative"
    return "Seasonal / Variable", "info"


def growth_status(pct_change: float) -> str:
    """Return a status token for a growth percentage."""
    if pct_change > 0:
        return "positive"
    if pct_change < 0:
        return "negative"
    return "neutral"


def confidence_status(label: str) -> str:
    """Return a status token for a confidence label."""
    if label == "High":
        return "positive"
    if label == "Medium":
        return "warning"
    return "negative"


def quality_status(rating: str) -> str:
    """Return a status token for a data quality rating."""
    if rating == "Good":
        return "positive"
    if rating == "Fair":
        return "warning"
    return "negative"


def primary_risk(
    review: StatisticalReviewResult | None,
    data_quality: DataQualitySection,
    forecast: ForecastResult,
    has_structural_breaks: bool = False,
) -> tuple[str, str]:
    """Return ``(risk description, status token)`` for the dashboard."""
    if review and review.verdict == "fail":
        return _REVIEW_CRITICAL_MSG, "negative"
    if data_quality.rating == "Poor":
        return "Poor data quality may compromise reliability", "negative"
    if forecast.mape is not None and forecast.mape > 20:
        return "High forecast uncertainty (MAPE > 20%)", "warning"
    if has_structural_breaks:
        return (
            "Candidate structural breaks require validation",
            "warning",
        )
    return "Forecast accuracy may decline over longer horizons", "neutral"


def recommended_action(
    review: StatisticalReviewResult | None,
    data_quality: DataQualitySection,
    forecast: ForecastResult | None = None,
) -> tuple[str, str]:
    """Return ``(action description, status token)`` for the dashboard."""
    holdout_ratio = None
    if forecast is not None:
        pooled_rmse = forecast.selection_metrics.get("rmse")
        if not isinstance(pooled_rmse, (int, float)):
            pooled_rmse = forecast.rmse
        holdout_ratio = recent_holdout_rmse_ratio(
            forecast.final_test_metrics.get("rmse"), pooled_rmse
        )
    if review and review.verdict in ("warn", "fail"):
        if (
            holdout_ratio is not None
            and holdout_ratio >= RECENT_HOLDOUT_RMSE_RATIO_THRESHOLD
        ):
            return (
                "Review statistical audit findings and monitor future actuals "
                f"closely; latest untouched holdout RMSE was {holdout_ratio:.2f}× "
                "rolling-origin RMSE",
                "warning",
            )
        return (
            "Review statistical audit findings and monitor future actuals",
            "warning",
        )
    if (
        holdout_ratio is not None
        and holdout_ratio >= RECENT_HOLDOUT_RMSE_RATIO_THRESHOLD
    ):
        return (
            "Monitor future actuals closely because latest untouched holdout RMSE "
            f"was {holdout_ratio:.2f}× rolling-origin RMSE",
            "warning",
        )
    if data_quality.rating != "Good":
        has_collection_issue = any(
            (
                data_quality.missing_values,
                data_quality.duplicate_timestamps,
                data_quality.missing_timestamps,
            )
        ) or not data_quality.is_regular
        if has_collection_issue:
            return "Improve data collection quality and re-run analysis", "warning"
        if data_quality.outlier_count:
            return "Review detected anomalies and monitor future actuals", "warning"
        if data_quality.issues:
            return "Review validation issues and monitor future actuals", "warning"
        return "Review the data-quality rating and monitor future actuals", "warning"
    return (
        "Use forecast for near-term planning; monitor future actuals",
        "positive",
    )
