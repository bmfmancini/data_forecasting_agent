"""Pipeline orchestration service for the Data Forecaster backend.

Contains the :func:`run_pipeline` function that executes the full
6-agent forecasting pipeline.  Extracted from ``orchestrator.py`` to
separate concerns (pipeline, chat, RAG).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from agents.data_validation_agent import run_validation_agent
from agents.forecasting_agent import run_forecasting_agent
from agents.model_selection_agent import run_model_selection_agent
from agents.report_generation_agent import run_report_agent
from agents.statistical_analysis_agent import run_statistical_agent
from agents.statistical_review_agent import run_statistical_review_agent
from core.logging_config import get_logger
from report.renderers import HTMLRenderer, MarkdownRenderer
from schemas import AnalysisResponse, ModelSelectionResult
from services.baseline_service import run_baseline_models
from services.rag_service import get_rag_kb
from utils.data_cleaning import apply_iqr_clipping, apply_zscore_clipping
from utils.preflight import prepare_series_frame
from utils.statistical import apply_boxcox, compute_acf_pacf, run_stl_decomposition
from utils.visualization import (
    chart_dict_to_png_b64,
    plot_acf_pacf,
    plot_forecast,
    plot_historical,
    plot_model_comparison,
    plot_stl,
)

logger = get_logger(__name__)


def run_pipeline(
    df: pd.DataFrame,
    file_id: str,
    date_col: str,
    value_col: str,
    freq: str,
    forecast_horizon: int,
    forced_model: str | None = None,
    user_prompt: str | None = None,
    preflight_options: dict[str, Any] | None = None,
    chroma_persist_dir: str = "./chroma_db",
    progress_callback: Callable[[int, str], None] | None = None,
) -> AnalysisResponse:
    """Execute the full 6-agent pipeline and return the complete AnalysisResponse.

    Args:
        df:                Input DataFrame.
        file_id:           Identifier for the uploaded file.
        date_col:          Name of the date column.
        value_col:         Name of the value column.
        freq:              Frequency string.
        forecast_horizon:  Number of future periods to forecast.
        forced_model:      Optional model override (``"ARIMA"``, ``"SARIMA"``,
                           ``"Holt-Winters"``).
        user_prompt:       Optional extra instructions for the report agent.
        preflight_options: Optional preflight configuration dict.
        chroma_persist_dir: Path to the ChromaDB persistence directory.
        progress_callback: Optional callback ``(pct, step)`` for progress updates.

    Returns:
        The complete :class:`AnalysisResponse`.
    """
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements

    def _progress(pct: int, step: str) -> None:
        if progress_callback:
            progress_callback(pct, step)

    logger.info(
        "Pipeline start: file_id=%s date_col=%s value_col=%s freq=%s horizon=%d",
        file_id,
        date_col,
        value_col,
        freq,
        forecast_horizon,
    )

    if (preflight_options or {}).get("continue_short_series") == "stop":
        raise ValueError("Analysis stopped because the selected series is too short.")

    df, freq = prepare_series_frame(df, date_col, value_col, preflight_options)
    series = df.set_index(date_col)[value_col].astype(float)
    seasonal_period = _freq_to_period(freq)

    # ── Agent 1: Data Validation ──────────────────────────────────────────────
    logger.info("Agent 1: Data Validation")
    _progress(5, "Validating data…")
    validation_result = run_validation_agent(
        df, date_col, value_col, freq, preflight_options=preflight_options
    )
    _progress(15, "Data validation complete")

    # ── Agent 2: Statistical Analysis ────────────────────────────────────────
    logger.info("Agent 2: Statistical Analysis")
    _progress(20, "Running statistical analysis…")
    user_domain = (preflight_options or {}).get("data_domain", "Skip / Let AI Guess")
    stat_result = run_statistical_agent(
        series, seasonal_period, user_domain=user_domain
    )
    _progress(35, "Statistical analysis complete")

    # ── Agent-Driven Remediation ─────────────────────────────────────────────
    if (preflight_options or {}).get("outlier_strategy") == "Let AI Decide":
        if "iqr_clip" in stat_result.recommended_remediation:
            logger.info("Agent decided to APPLY IQR clipping.")
            series = apply_iqr_clipping(series)
            logger.info("IQR clipping applied successfully.")
        elif "zscore_clip" in stat_result.recommended_remediation:
            logger.info("Agent decided to APPLY Z-score clipping.")
            series = apply_zscore_clipping(series)
            logger.info("Z-score clipping applied successfully.")
        else:
            logger.info(
                "Agent decided to SKIP outlier clipping "
                "(likely determined outliers are signal)."
            )

    if "box_cox" in stat_result.recommended_remediation:
        logger.info("Agent decided to APPLY Box-Cox transformation.")
        try:
            series, _ = apply_boxcox(series)
            stat_result.summary += (
                "\n\n(Note: A Box-Cox transformation was applied to stabilize "
                "variance based on agent recommendation.)"
            )
        except Exception as e:
            logger.warning("Box-Cox application failed: %s", e)

    if "change_point_analysis" in stat_result.recommended_remediation:
        logger.info(
            "Agent detected significant change points. Adding note to analysis."
        )
        stat_result.summary += (
            "\n\n(Note: Change point analysis detected structural breaks. "
            "Consider segmenting the data for improved forecasting accuracy.)"
        )

    # ── Agent 3: Model Selection ──────────────────────────────────────────────
    if forced_model:
        logger.info(
            "Agent 3: Model Selection skipped — user forced model: %s", forced_model
        )
        _progress(40, f"Model manually set to {forced_model}")
        model_selection = ModelSelectionResult(
            selected_model=forced_model,
            explanation=f"Model manually selected by user: {forced_model}.",
            holt_winters_rejected_reason=(
                None
                if forced_model == "Holt-Winters"
                else "Not selected (user chose a different model)."
            ),
            arima_rejected_reason=(
                None
                if forced_model == "ARIMA"
                else "Not selected (user chose a different model)."
            ),
            sarima_rejected_reason=(
                None
                if forced_model == "SARIMA"
                else "Not selected (user chose a different model)."
            ),
            reasoning_steps=[
                {
                    "thought": (
                        f"User explicitly requested the {forced_model} model. "
                        "Skipping automated model selection logic."
                    ),
                    "observation": f"Manual selection active: {forced_model}",
                }
            ],
        )
    else:
        logger.info("Agent 3: Model Selection")
        _progress(40, "Selecting forecasting model…")
        model_selection = run_model_selection_agent(stat_result)
    _progress(55, f"Model selected: {model_selection.selected_model}")

    # ── Agent 4: Forecasting ──────────────────────────────────────────────────
    logger.info("Agent 4: Forecasting")
    _progress(60, "Running forecast…")
    forecast_result, all_metrics = run_forecasting_agent(
        series, model_selection, stat_result, forecast_horizon, freq
    )
    _progress(75, "Forecast complete")

    # ── Baseline Model Comparison ─────────────────────────────────────────────
    logger.info("Running baseline model comparisons")
    baseline_metrics = run_baseline_models(
        series, forecast_horizon, seasonal_period
    )
    all_metrics.update(baseline_metrics)
    logger.info("Baseline models complete")

    # ── Agent 4.5: Statistical Review (QA) ────────────────────────────────────
    logger.info("Agent 4.5: Statistical Review")
    _progress(77, "Statistical review…")
    statistical_review = run_statistical_review_agent(
        stat_result, model_selection, forecast_result, all_metrics
    )
    _progress(80, "Statistical review complete")

    # ── Review-Triggered Re-run (one-shot) ────────────────────────────────────
    retry_enabled = (preflight_options or {}).get(
        "statistical_review_retry_enabled", True
    )
    has_critical = any(
        f.get("severity") == "critical" for f in statistical_review.flags
    )
    if retry_enabled and (has_critical or statistical_review.verdict == "fail"):
        logger.info(
            "Statistical review flagged critical issues — re-running model "
            "selection with review feedback."
        )
        _progress(82, "Re-running model selection with review feedback…")

        # Preserve token usage from the first attempt so the pipeline
        # totals include both executions.
        prev_model_selection_usage = dict(model_selection.token_usage)
        prev_forecast_usage = dict(forecast_result.token_usage)
        prev_review_usage = dict(statistical_review.token_usage)

        review_feedback = statistical_review.summary

        if forced_model:
            # When the user forced a model, do not rerun model selection.
            # Only append review feedback to the existing explanation.
            logger.info(
                "Forced model '%s' — skipping model reselection, "
                "appending review feedback only.",
                forced_model,
            )
            model_selection = model_selection.model_copy(
                update={
                    "explanation": (
                        f"{model_selection.explanation}\n\n"
                        f"[Statistical Review Feedback]: {review_feedback}"
                    )
                }
            )
        else:
            # Re-run model selection with review feedback, exclude the
            # previously selected model, and pass the actual error metrics
            # from the prior forecasting run so the reselection is
            # evidence-based rather than relying on statistical properties
            # alone.
            prev_selected = model_selection.selected_model
            model_selection = run_model_selection_agent(
                stat_result,
                review_feedback=review_feedback,
                exclude_model=prev_selected,
                all_metrics=all_metrics,
            )
            # Append review feedback to the model selection explanation
            model_selection = model_selection.model_copy(
                update={
                    "explanation": (
                        f"{model_selection.explanation}\n\n"
                        f"[Statistical Review Feedback]: {review_feedback}"
                    )
                }
            )

        # Re-run forecasting with the new model selection
        _progress(85, "Re-running forecast with revised model…")
        forecast_result, all_metrics = run_forecasting_agent(
            series,
            model_selection,
            stat_result,
            forecast_horizon,
            freq,
            all_metrics=all_metrics,  # Pass existing metrics
        )
        # Re-run statistical review on the new outputs
        _progress(87, "Re-running statistical review…")
        statistical_review = run_statistical_review_agent(
            stat_result, model_selection, forecast_result, all_metrics
        )
        _progress(88, "Statistical review re-run complete")

        # Merge token usage from the first attempt into the retry results
        # so the pipeline totals include both executions.
        model_selection = model_selection.model_copy(
            update={
                "token_usage": _merge_token_usage(
                    prev_model_selection_usage, model_selection.token_usage
                )
            }
        )
        forecast_result = forecast_result.model_copy(
            update={
                "token_usage": _merge_token_usage(
                    prev_forecast_usage, forecast_result.token_usage
                )
            }
        )
        statistical_review = statistical_review.model_copy(
            update={
                "token_usage": _merge_token_usage(
                    prev_review_usage, statistical_review.token_usage
                )
            }
        )

    # ── Agent 5: Report Generation ────────────────────────────────────────────
    logger.info("Agent 5: Report Generation")
    _progress(90, "Generating report…")
    rag_kb = get_rag_kb(chroma_persist_dir)
    executive_report, report_reasoning, visual_strategy, report_token_usage = (
        run_report_agent(
            validation_result,
            stat_result,
            model_selection,
            forecast_result,
            rag_kb,
            user_prompt=user_prompt,
            preflight_options=preflight_options,
            statistical_review=statistical_review,
            all_metrics=all_metrics,
        )
    )
    # ── Render report to Markdown and HTML ────────────────────────────────
    # Renderers are best-effort: a renderer bug must not fail the entire
    # successful pipeline.  On failure, fall back to empty strings and log
    # a warning so the frontend can still display the structured report.
    report_md = ""
    report_html = ""
    try:
        md_renderer = MarkdownRenderer()
        html_renderer = HTMLRenderer()
        report_md = md_renderer.render(executive_report)
        report_html = html_renderer.render(executive_report)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "Report rendering failed: %s — using empty render output.", exc
        )
    _progress(92, "Report complete")

    # ── Visualizations ────────────────────────────────────────────────────────
    logger.info("Generating visualizations")
    _progress(95, "Generating visualizations…")
    sp = stat_result.seasonal_period or seasonal_period

    chart_historical = plot_historical(series)

    try:
        stl_data = run_stl_decomposition(series, period=sp)
        chart_stl = plot_stl(series, stl_data, sp)
    except Exception as exc:
        logger.warning("STL chart failed: %s", exc)
        chart_stl = {}

    try:
        acf_data = compute_acf_pacf(series)
        chart_acf_pacf = plot_acf_pacf(
            acf_data["acf_values"], acf_data["pacf_values"], acf_data["lags"]
        )
    except Exception as exc:
        logger.warning("ACF/PACF chart failed: %s", exc)
        chart_acf_pacf = ""

    chart_forecast = plot_forecast(series, forecast_result)
    chart_model_comparison = plot_model_comparison(all_metrics)

    # ── Generate PNG versions for PDF export ────────────────────────────────
    chart_historical_png = ""
    chart_stl_png = ""
    chart_forecast_png = ""
    chart_model_comparison_png = ""
    try:
        chart_historical_png = chart_dict_to_png_b64(chart_historical)
    except Exception as exc:
        logger.warning("Historical chart PNG failed: %s", exc)
    try:
        chart_stl_png = chart_dict_to_png_b64(chart_stl)
    except Exception as exc:
        logger.warning("STL chart PNG failed: %s", exc)
    try:
        chart_forecast_png = chart_dict_to_png_b64(chart_forecast)
    except Exception as exc:
        logger.warning("Forecast chart PNG failed: %s", exc)
    try:
        chart_model_comparison_png = chart_dict_to_png_b64(chart_model_comparison)
    except Exception as exc:
        logger.warning("Model comparison chart PNG failed: %s", exc)

    # ── Token Usage Aggregation ─────────────────────────────────────────────
    agent_usage = {
        "validation": validation_result.token_usage,
        "statistical": stat_result.token_usage,
        "model_selection": model_selection.token_usage,
        "forecast": forecast_result.token_usage,
        "statistical_review": statistical_review.token_usage,
        "report": report_token_usage,
    }
    grand_total = {
        "input_tokens": sum(u.get("input_tokens", 0) for u in agent_usage.values()),
        "output_tokens": sum(u.get("output_tokens", 0) for u in agent_usage.values()),
        "total_tokens": sum(u.get("total_tokens", 0) for u in agent_usage.values()),
    }
    estimated = any(u.get("estimated", False) for u in agent_usage.values())
    pipeline_token_usage = {
        "agents": agent_usage,
        "grand_total": grand_total,
        "estimated": estimated,
    }
    logger.info(
        "Pipeline token usage: input=%d output=%d total=%d (estimated=%s)",
        grand_total["input_tokens"],
        grand_total["output_tokens"],
        grand_total["total_tokens"],
        estimated,
    )

    logger.info("Pipeline complete: file_id=%s", file_id)
    _progress(100, "Analysis complete")

    # Check if LLM fallback occurred during report generation
    llm_fallback = any(step.get("llm_fallback", False) for step in report_reasoning)

    return AnalysisResponse(
        file_id=file_id,
        validation=validation_result,
        statistical=stat_result,
        model_selection=model_selection,
        forecast=forecast_result,
        statistical_review=statistical_review,
        report=report_md,
        executive_report=executive_report.model_dump(),
        report_html=report_html,
        report_reasoning=report_reasoning,
        strategic_visual_recommendations=visual_strategy,
        llm_fallback=llm_fallback,
        chart_historical=chart_historical,
        chart_stl=chart_stl,
        chart_acf_pacf=chart_acf_pacf,
        chart_forecast=chart_forecast,
        chart_model_comparison=chart_model_comparison,
        chart_historical_png=chart_historical_png,
        chart_stl_png=chart_stl_png,
        chart_forecast_png=chart_forecast_png,
        chart_model_comparison_png=chart_model_comparison_png,
        pipeline_token_usage=pipeline_token_usage,
    )


def _merge_token_usage(
    first: dict[str, int],
    second: dict[str, int],
) -> dict[str, int]:
    """Merge two token-usage dicts by summing their numeric values.

    Args:
        first:  Token usage from the first attempt.
        second: Token usage from the retry attempt.

    Returns:
        A new dict with summed values for ``input_tokens``,
        ``output_tokens``, and ``total_tokens``.  The ``estimated`` flag is
        set to ``True`` if either input was estimated.
    """
    merged: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        merged[key] = first.get(key, 0) + second.get(key, 0)
    merged["estimated"] = bool(
        first.get("estimated", False) or second.get("estimated", False)
    )
    return merged


def _freq_to_period(freq: str) -> int:
    """Convert a frequency string to a seasonal period integer.

    Args:
        freq: Pandas frequency string (e.g. ``"MS"``, ``"Q"``, ``"D"``).

    Returns:
        The number of periods in one seasonal cycle (default 12).
    """
    f = (freq or "").upper().lstrip("-")
    if f.startswith("MS") or f.startswith("M"):
        return 12
    if f.startswith("QS") or f.startswith("Q"):
        return 4
    if f.startswith("W"):
        return 52
    if f.startswith("D"):
        return 7
    if f.startswith("H"):
        return 24
    if f.startswith("Y") or f.startswith("A"):
        return 1
    return 12
