from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from core.config import GEMINI_MODEL, USE_OLLAMA, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_API_KEY
from core.logging_config import get_logger
from schemas import StatisticalResult
from utils.statistical import (
    compute_acf_pacf,
    detect_trend,
    run_adf_test,
    run_kpss_test,
    run_periodogram,
    run_stl_decomposition,
)

logger = get_logger(__name__)


def run_statistical_agent(series: pd.Series, seasonal_period: int = 12) -> StatisticalResult:
    """Run statistical analysis ReAct agent and return a StatisticalResult."""

    # ── Compute all stats directly ────────────────────────────────────────────
    adf = run_adf_test(series)
    kpss_res = run_kpss_test(series)
    trend = detect_trend(series)
    periodogram = run_periodogram(series)

    # Infer seasonal period: prefer explicit arg, validate against periodogram
    dom_period = periodogram["dominant_period"]
    inferred_period: Optional[int] = seasonal_period
    if dom_period < 100:
        pg_period = int(round(dom_period))
        if pg_period > 1:
            # Prefer frequency-derived period but log any mismatch
            if abs(pg_period - seasonal_period) > 2:
                logger.info(
                    "Periodogram period %d differs from freq-derived period %d; using %d",
                    pg_period, seasonal_period, seasonal_period,
                )

    # ── Build Statistical Profile ─────────────────────────────────────────────
    stl = run_stl_decomposition(series, period=inferred_period or 12)
    acf_data = compute_acf_pacf(series)
    conf_bound = 1.96 / np.sqrt(len(series))
    sig_acf = [i for i, v in enumerate(acf_data["acf_values"][1:], 1) if abs(v) > conf_bound]
    
    profile = (
        f"STATISTICAL PROFILE:\n"
        f"- ADF: {adf['interpretation']}\n"
        f"- KPSS: {kpss_res['interpretation']}\n"
        f"- Trend: {trend['interpretation']}\n"
        f"- Dominant Period: {periodogram['dominant_period']:.2f}\n"
        f"- STL Seasonal Range: {max(stl['seasonal']) - min(stl['seasonal']):.2f}\n"
        f"- Significant ACF Lags: {sig_acf[:5]}"
    )

    # ── LLM Setup ────────────────────────────────────────────────────────────
    if USE_OLLAMA:
        llm = ChatOllama(
            model=OLLAMA_MODEL, 
            base_url=OLLAMA_BASE_URL, 
            temperature=0,
            headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else None,
        )
    else:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a statistics expert. Summarize the time series characteristics."),
        ("human", (
            "Perform a statistical analysis based on this profile:\n\n"
            "{profile}\n\n"
            "Provide a concise qualitative summary of the series stationarity, trend strength, and seasonal patterns."
        ))
    ])

    try:
        chain = prompt | llm
        response = chain.invoke({"profile": profile})
        summary = response.content
        reasoning_steps = [
            {"thought": "Running ADF, KPSS, and STL in Python...", "observation": profile},
            {"thought": "Generating qualitative interpretation...", "observation": "Complete"}
        ]
    except Exception as exc:
        logger.warning("Statistical agent LLM call failed: %s", exc)
        summary = (
            f"ADF: {'stationary' if adf['is_stationary'] else 'non-stationary'}. "
            f"KPSS: {'stationary' if kpss_res['is_stationary'] else 'non-stationary'}. "
            f"Trend: {'present' if trend['has_trend'] else 'absent'}."
        )
        reasoning_steps = [{
            "thought": f"Statistical agent failed: {str(exc)}",
            "observation": "Falling back to raw statistical test results."
        }]

    logger.info(
        "Statistical analysis complete. stationary_adf=%s seasonal_period=%s",
        adf["is_stationary"], inferred_period,
    )

    return StatisticalResult(
        is_stationary_adf=adf["is_stationary"],
        adf_statistic=adf["statistic"],
        adf_p_value=adf["p_value"],
        is_stationary_kpss=kpss_res["is_stationary"],
        kpss_statistic=kpss_res["statistic"],
        kpss_p_value=kpss_res["p_value"],
        has_trend=trend["has_trend"],
        trend_slope=trend["slope"],
        seasonal_period=inferred_period,
        dominant_period=periodogram["dominant_period"],
        summary=summary,
        reasoning_steps=reasoning_steps,
    )
