from __future__ import annotations

import pandas as pd

from agents.data_validation_agent import run_validation_agent
from agents.forecasting_agent import run_forecasting_agent
from agents.model_selection_agent import run_model_selection_agent
from agents.report_generation_agent import run_report_agent
from agents.statistical_analysis_agent import run_statistical_agent
from core.logging_config import get_logger
from rag.knowledge_base import RAGKnowledgeBase
from schemas import AnalysisResponse
from utils.statistical import compute_acf_pacf, run_stl_decomposition
from utils.visualization import (
    plot_acf_pacf,
    plot_forecast,
    plot_historical,
    plot_model_comparison,
    plot_stl,
)

logger = get_logger(__name__)

# Shared RAG knowledge base (initialised once on startup)
_rag_kb: RAGKnowledgeBase | None = None


def get_rag_kb(persist_directory: str = "./chroma_db") -> RAGKnowledgeBase:
    global _rag_kb
    if _rag_kb is None:
        _rag_kb = RAGKnowledgeBase(persist_directory=persist_directory)
        _rag_kb.load_documents()
    return _rag_kb


def run_pipeline(
    df: pd.DataFrame,
    file_id: str,
    date_col: str,
    value_col: str,
    freq: str,
    forecast_horizon: int,
    chroma_persist_dir: str = "./chroma_db",
) -> AnalysisResponse:
    """Execute the full 5-agent pipeline and return the complete AnalysisResponse."""

    logger.info(
        "Pipeline start: file_id=%s date_col=%s value_col=%s freq=%s horizon=%d",
        file_id, date_col, value_col, freq, forecast_horizon,
    )

    series = df.set_index(date_col)[value_col].astype(float)
    seasonal_period = _freq_to_period(freq)

    # ── Agent 1: Data Validation ──────────────────────────────────────────────
    logger.info("Agent 1: Data Validation")
    validation_result = run_validation_agent(df, date_col, value_col, freq)

    # ── Agent 2: Statistical Analysis ────────────────────────────────────────
    logger.info("Agent 2: Statistical Analysis")
    stat_result = run_statistical_agent(series, seasonal_period)

    # ── Agent 3: Model Selection ──────────────────────────────────────────────
    logger.info("Agent 3: Model Selection")
    model_selection = run_model_selection_agent(stat_result)

    # ── Agent 4: Forecasting ──────────────────────────────────────────────────
    logger.info("Agent 4: Forecasting")
    forecast_result, all_metrics = run_forecasting_agent(
        series, model_selection, stat_result, forecast_horizon, freq
    )

    # ── Agent 5: Report Generation ────────────────────────────────────────────
    logger.info("Agent 5: Report Generation")
    rag_kb = get_rag_kb(chroma_persist_dir)
    report = run_report_agent(validation_result, stat_result, model_selection, forecast_result, rag_kb)

    # ── Visualizations ────────────────────────────────────────────────────────
    logger.info("Generating visualizations")
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

    logger.info("Pipeline complete: file_id=%s", file_id)

    return AnalysisResponse(
        file_id=file_id,
        validation=validation_result,
        statistical=stat_result,
        model_selection=model_selection,
        forecast=forecast_result,
        report=report,
        chart_historical=chart_historical,
        chart_stl=chart_stl,
        chart_acf_pacf=chart_acf_pacf,
        chart_forecast=chart_forecast,
        chart_model_comparison=chart_model_comparison,
    )


def _freq_to_period(freq: str) -> int:
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
