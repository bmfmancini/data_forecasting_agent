from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import GEMINI_MODEL
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

_REACT_PROMPT = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
)


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

    # ── Tool definitions (closed over series / computed stats) ───────────────

    @tool
    def get_statistical_profile(command: str) -> str:
        """Returns a comprehensive summary of ADF, KPSS, Trend, and Periodogram results."""
        # STL mini-calc
        stl = run_stl_decomposition(series, period=inferred_period or 12)
        acf_data = compute_acf_pacf(series)
        conf_bound = 1.96 / np.sqrt(len(series))
        sig_acf = [i for i, v in enumerate(acf_data["acf_values"][1:], 1) if abs(v) > conf_bound]
        
        return (
            f"STATISTICAL PROFILE:\n"
            f"- ADF: {adf['interpretation']}\n"
            f"- KPSS: {kpss_res['interpretation']}\n"
            f"- Trend: {trend['interpretation']}\n"
            f"- Dominant Period: {periodogram['dominant_period']:.2f}\n"
            f"- STL Seasonal Range: {max(stl['seasonal']) - min(stl['seasonal']):.2f}\n"
            f"- Significant ACF Lags: {sig_acf[:5]}"
        )

    # ── Run ReAct agent for qualitative summary ───────────────────────────────
    tools_list = [get_statistical_profile]
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    agent = create_react_agent(llm, tools_list, _REACT_PROMPT)
    executor = AgentExecutor(
        agent=agent, tools=tools_list, verbose=False, return_intermediate_steps=True,
        max_iterations=7, handle_parsing_errors=True,
    )

    reasoning_steps: list[dict[str, Any]] = []
    try:
        result = executor.invoke({
            "input": (
                "Perform a statistical analysis. You MUST use the get_statistical_profile tool. "
                "Based on the results, provide a concise qualitative summary of the series "
                "stationarity, trend strength, and seasonal patterns."
            )
        })
        summary = str(result.get("output", "Statistical analysis complete."))
        reasoning_steps = [
            {"thought": a.log, "observation": str(o)} for a, o in result.get("intermediate_steps", [])
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
