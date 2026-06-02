from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from core.config import GROQ_API_KEY
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
Thought:{agent_scratchpad}"""
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
    def run_adf_test_tool(command: str) -> str:
        """Run the Augmented Dickey-Fuller stationarity test."""
        return adf["interpretation"]

    @tool
    def run_kpss_test_tool(command: str) -> str:
        """Run the KPSS stationarity test (null = stationary)."""
        return kpss_res["interpretation"]

    @tool
    def run_stl_decomposition_tool(command: str) -> str:
        """Run STL decomposition to separate trend, seasonal, and residual components."""
        try:
            stl = run_stl_decomposition(series, period=inferred_period or 12)
            trend_range = max(stl["trend"]) - min(stl["trend"])
            seas_range = max(stl["seasonal"]) - min(stl["seasonal"])
            resid_std = float(np.std(stl["residual"]))
            return (
                f"STL decomposition (period={inferred_period}): "
                f"trend range={trend_range:.2f}, seasonal range={seas_range:.2f}, "
                f"residual std={resid_std:.2f}."
            )
        except Exception as exc:
            return f"STL decomposition failed: {exc}"

    @tool
    def compute_acf_pacf_tool(command: str) -> str:
        """Compute ACF and PACF to identify autocorrelation structure."""
        acf_data = compute_acf_pacf(series)
        # Report first few significant lags
        conf_bound = 1.96 / np.sqrt(len(series))
        sig_acf = [i for i, v in enumerate(acf_data["acf_values"][1:], 1) if abs(v) > conf_bound]
        sig_pacf = [i for i, v in enumerate(acf_data["pacf_values"][1:], 1) if abs(v) > conf_bound]
        return (
            f"ACF significant lags (95% CI): {sig_acf[:10]}. "
            f"PACF significant lags (95% CI): {sig_pacf[:10]}."
        )

    @tool
    def run_periodogram_tool(command: str) -> str:
        """Run spectral periodogram to identify dominant periodic cycle."""
        dp = periodogram["dominant_period"]
        return (
            f"Dominant period from periodogram: {dp:.2f} observations. "
            f"Suggests seasonal cycle of approximately {int(round(dp))} periods."
        )

    @tool
    def detect_trend_tool(command: str) -> str:
        """Detect whether a significant linear trend exists in the series."""
        return trend["interpretation"]

    # ── Run ReAct agent for qualitative summary ───────────────────────────────
    tools_list = [
        run_adf_test_tool, run_kpss_test_tool, run_stl_decomposition_tool,
        compute_acf_pacf_tool, run_periodogram_tool, detect_trend_tool,
    ]
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY, temperature=0)
    agent = create_react_agent(llm, tools_list, _REACT_PROMPT)
    executor = AgentExecutor(
        agent=agent, tools=tools_list, verbose=False,
        max_iterations=14, handle_parsing_errors=True,
    )

    try:
        result = executor.invoke({
            "input": (
                "Use all available tools to perform a comprehensive statistical analysis of "
                "this time series. Run the ADF and KPSS stationarity tests, STL decomposition, "
                "ACF/PACF analysis, periodogram, and trend detection. "
                "Summarise all findings including stationarity, seasonality, and trend."
            )
        })
        summary = str(result.get("output", "Statistical analysis complete."))
    except Exception as exc:
        logger.warning("Statistical agent LLM call failed: %s", exc)
        summary = (
            f"ADF: {'stationary' if adf['is_stationary'] else 'non-stationary'}. "
            f"KPSS: {'stationary' if kpss_res['is_stationary'] else 'non-stationary'}. "
            f"Trend: {'present' if trend['has_trend'] else 'absent'}."
        )

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
    )
