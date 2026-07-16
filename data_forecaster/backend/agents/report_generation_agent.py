"""Report generation agent for turning forecast results into review narratives."""

from __future__ import annotations

from typing import Any
from report.narrative import _fallback_narrative

from core.logging_config import get_logger
from rag.knowledge_base import RAGKnowledgeBase
from report import ExecutiveReportBuilder, generate_narratives
from report.models import ExecutiveReport
from report.rules import VISUAL_STRATEGY_THRESHOLDS
from schemas import (
    ForecastResult,
    ModelSelectionResult,
    StatisticalResult,
    StatisticalReviewResult,
    ValidationResult,
)

logger = get_logger(__name__)


def run_report_agent(
    validation: ValidationResult,
    statistical: StatisticalResult,
    model_selection: ModelSelectionResult,
    forecast: ForecastResult,
    rag_kb: RAGKnowledgeBase,
    user_prompt: str | None = None,
    preflight_options: dict[str, Any] | None = None,
    statistical_review: StatisticalReviewResult | None = None,
    all_metrics: dict[str, dict[str, float]] | None = None,
) -> tuple[
    ExecutiveReport,
    list[dict[str, Any]],
    list[dict[str, str]],
    dict[str, int],
]:
    """Generate a structured executive report using the two-stage architecture.

    Stage 1: :class:`ExecutiveReportBuilder` computes all deterministic
    metrics and populates an :class:`ExecutiveReport` model.
    Stage 2: :func:`generate_narratives` fills the ``narrative`` text fields
    using the LLM, receiving the structured model as context.

    Args:
        validation:          Data validation agent output.
        statistical:         Statistical analysis agent output.
        model_selection:     Model selection agent output.
        forecast:            Forecasting agent output.
        rag_kb:              RAG knowledge base (retained for future use).
        user_prompt:         Optional extra instructions for narrative tone.
        preflight_options:   Optional preflight configuration dict.
        statistical_review:  Statistical review (QA) agent output.
        all_metrics:         All model comparison metrics dict.

    Returns:
        A tuple of (ExecutiveReport, reasoning_steps, visual_strategy,
        token_usage).
    """
    del rag_kb  # RAG context not needed for per-section narrative prompts.
    del preflight_options  # Preflight options handled by the builder inputs.

    reasoning_steps: list[dict[str, Any]] = [
        {
            "thought": "Stage 1: Computing deterministic report metrics...",
            "observation": "ExecutiveReportBuilder started",
        }
    ]

    # ── Stage 1: Build structured report (deterministic) ──────────────────
    builder = ExecutiveReportBuilder()
    report = builder.build(
        validation=validation,
        statistical=statistical,
        model_selection=model_selection,
        forecast=forecast,
        statistical_review=statistical_review,
        all_metrics=all_metrics or {},
    )
    reasoning_steps.append(
        {
            "thought": "Stage 1 complete: ExecutiveReport model populated",
            "observation": (
                f"Confidence: {report.confidence.score}/100 "
                f"({report.confidence.label}), "
                f"Dashboard widgets: {len(report.dashboard.widgets)}, "
                f"Recommendations: {len(report.recommendations)}"
            ),
        }
    )

    # ── Stage 2: Generate narratives via LLM ──────────────────────────────
    reasoning_steps.append(
        {
            "thought": "Stage 2: Generating narrative text via LLM...",
            "observation": "Narrative generation started",
        }
    )
    try:
        report, token_usage = generate_narratives(report, user_prompt)
        reasoning_steps.append(
            {
                "thought": "Stage 2 complete: Narratives generated",
                "observation": (f"Tokens: {token_usage.get('total_tokens', 0)}"),
            }
        )
    except Exception as exc:
        logger.warning(
            "Narrative generation failed: %s — using fallback narratives.",
            exc,
        )
        token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        reasoning_steps.append(
            {
                "thought": f"Narrative generation error: {exc}",
                "observation": "Fallback narratives used",
                "llm_fallback": True,
            }
        )
        # Ensure all narrative fields have a fallback value
        report.executive_summary.narrative = _fallback_narrative("executive_summary")
        report.data_quality.narrative = _fallback_narrative("data_quality")
        report.historical_analysis.narrative = _fallback_narrative(
            "historical_analysis"
        )
        report.forecast_outlook.narrative = _fallback_narrative("forecast_outlook")
        report.model_comparison.narrative = _fallback_narrative("model_comparison")
        report.statistical_audit.narrative = _fallback_narrative("statistical_audit")
        report.explainability.narrative = _fallback_narrative("explainability")
        report.recommendations = [
            rec.model_copy(update={"narrative": _fallback_narrative("recommendation")})
            for rec in report.recommendations
        ]

    # ── Visual strategy (retained for pipeline compatibility) ─────────────
    visual_strategy = _compute_visual_strategy(statistical, forecast, model_selection)

    logger.info(
        "Report generation complete. Confidence: %d/100, Sections: 12",
        report.confidence.score,
    )
    return report, reasoning_steps, visual_strategy, token_usage


def _compute_visual_strategy(
    statistical: StatisticalResult,
    forecast: ForecastResult,
    model_selection: ModelSelectionResult,
) -> list[dict[str, str]]:
    """Compute visual strategy recommendations based on data characteristics.

    Args:
        statistical:     Statistical analysis result.
        forecast:        Forecast result.
        model_selection: Model selection result.

    Returns:
        List of visual strategy recommendation dicts.
    """
    # pylint: disable=too-many-branches
    strategy: list[dict[str, str]] = []
    if statistical.seasonal_period and statistical.seasonal_period > 1:
        strategy.append(
            {
                "chart": "STL Decomposition",
                "reason": (
                    "Strong seasonal patterns detected; decomposition is "
                    "essential to isolate recurring cycles from underlying "
                    "growth."
                ),
            }
        )
    if (
        forecast.mape is not None
        and forecast.mape > VISUAL_STRATEGY_THRESHOLDS["mape_high"]
    ):
        if forecast.interval_label == "unavailable":
            strategy.append(
                {
                    "chart": "Forecast Error Monitoring",
                    "reason": (
                        "Forecast error is elevated and prediction-interval bounds "
                        "are unavailable; emphasize holdout performance and future "
                        "actuals instead of implying a 95% range."
                    ),
                }
            )
        else:
            interval_name = (
                "Estimated 95% Prediction Intervals (coverage not evaluated)"
                if forecast.interval_label == "experimental"
                else "Model-Based 95% Prediction Intervals"
            )
            strategy.append(
                {
                    "chart": interval_name,
                    "reason": (
                        "Elevated forecast error requires emphasis on the prediction-"
                        "interval ribbon to communicate risk and uncertainty."
                    ),
                }
            )
    if model_selection.selected_model == "SARIMA":
        strategy.append(
            {
                "chart": "ACF/PACF",
                "reason": (
                    "Used to validate the seasonal autoregressive components "
                    "of the selected model."
                ),
            }
        )
    if statistical.outlier_ratio > VISUAL_STRATEGY_THRESHOLDS["outlier_ratio_high"]:
        strategy.append(
            {
                "chart": "Box Plot",
                "reason": (
                    "The anomaly ratio exceeds the review threshold; a box plot "
                    "would display the distribution and flagged values without "
                    "assuming their business impact."
                ),
            }
        )
    if (
        statistical.has_trend
        and abs(statistical.trend_slope) > VISUAL_STRATEGY_THRESHOLDS["trend_slope_min"]
    ):
        strategy.append(
            {
                "chart": "Trend Analysis",
                "reason": (
                    "Clear trend detected; a trend analysis visualization "
                    "would help illustrate the direction and magnitude of "
                    "change over time."
                ),
            }
        )
    if (
        forecast.mape is not None
        and forecast.mape > VISUAL_STRATEGY_THRESHOLDS["mape_moderate"]
    ):
        strategy.append(
            {
                "chart": "Forecast Error Plot",
                "reason": (
                    "Forecast errors are significant; an error plot would "
                    "help diagnose model performance and identify patterns "
                    "in prediction accuracy."
                ),
            }
        )
    strategy.append(
        {
            "chart": "Histogram",
            "reason": (
                "A histogram of the data distribution provides insights "
                "into central tendency, spread, and shape of the time "
                "series."
            ),
        }
    )
    return strategy
