from __future__ import annotations

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from core.config import GROQ_API_KEY
from core.logging_config import get_logger
from rag.knowledge_base import RAGKnowledgeBase
from schemas import ForecastResult, ModelSelectionResult, StatisticalResult, ValidationResult

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

_REPORT_SECTIONS = [
    "executive summary forecasting results",
    "data quality validation time series",
    "statistical analysis stationarity trend seasonality",
    "model selection ARIMA SARIMA Holt-Winters criteria",
    "forecast accuracy metrics RMSE MAE MAPE confidence intervals",
    "methodology assumptions limitations forecasting",
]


def run_report_agent(
    validation: ValidationResult,
    statistical: StatisticalResult,
    model_selection: ModelSelectionResult,
    forecast: ForecastResult,
    rag_kb: RAGKnowledgeBase,
) -> str:
    """Use the LLM with RAG context to write a 6-section analyst report."""

    # ── Build analysis context string ─────────────────────────────────────────
    analysis_context = f"""
ANALYSIS RESULTS SUMMARY
=========================
Data Quality:
  - Rows: {validation.row_count}
  - Missing timestamps: {validation.missing_timestamps}
  - Duplicate timestamps: {validation.duplicate_timestamps}
  - Missing values: {validation.missing_values}
  - Regular intervals: {validation.is_regular}
  - Detected frequency: {validation.frequency}
  - Issues: {'; '.join(validation.issues) if validation.issues else 'None'}

Statistical Analysis:
  - ADF stationary: {statistical.is_stationary_adf} (p={statistical.adf_p_value:.4f})
  - KPSS stationary: {statistical.is_stationary_kpss} (p={statistical.kpss_p_value:.4f})
  - Trend detected: {statistical.has_trend} (slope={statistical.trend_slope:.6f})
  - Seasonal period: {statistical.seasonal_period}
  - Dominant periodogram period: {statistical.dominant_period:.2f}

Model Selection:
  - Selected model: {model_selection.selected_model}
  - Reasoning: {model_selection.explanation[:400]}

Forecast Results:
  - Model used: {forecast.model_used}
  - Horizon: {len(forecast.forecast)} periods
  - RMSE: {forecast.rmse:.4f}
  - MAE: {forecast.mae:.4f}
  - MAPE: {forecast.mape:.2f}%
  - Forecast values (first 5): {[round(v, 2) for v in forecast.forecast[:5]]}
"""

    # ── Tool definitions ──────────────────────────────────────────────────────
    retrieved_context: list[str] = []

    @tool
    def retrieve_from_rag(query: str) -> str:
        """Retrieve relevant methodology documentation from the knowledge base.
        Provide a clear topic query such as 'ARIMA model assumptions' or
        'stationarity testing interpretation'."""
        try:
            chunks = rag_kb.retrieve(query, k=3)
            text = "\n---\n".join(chunks)
            retrieved_context.append(text)
            return text
        except Exception as exc:
            logger.warning("RAG retrieval failed for query '%s': %s", query, exc)
            return f"RAG retrieval unavailable: {exc}"

    # ── Run ReAct agent ───────────────────────────────────────────────────────
    tools_list = [retrieve_from_rag]
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY, temperature=0)
    agent = create_react_agent(llm, tools_list, _REACT_PROMPT)
    executor = AgentExecutor(
        agent=agent, tools=tools_list, verbose=False,
        max_iterations=8, handle_parsing_errors=True,
    )

    rag_queries = " ".join(f"'{q}'," for q in _REPORT_SECTIONS)

    try:
        result = executor.invoke({
            "input": (
                f"You are a professional data analyst. Using the analysis results below and the "
                f"retrieve_from_rag tool, write a comprehensive 6-section forecast report.\n\n"
                f"{analysis_context}\n\n"
                f"Use retrieve_from_rag for each of these topics: {rag_queries}\n\n"
                f"Write the report with these 6 sections using Markdown headings:\n"
                f"1. Executive Summary\n"
                f"2. Data Quality Assessment\n"
                f"3. Statistical Analysis\n"
                f"4. Model Selection & Rationale\n"
                f"5. Forecast Results\n"
                f"6. Methodology & Limitations\n\n"
                f"Incorporate the retrieved methodology context naturally in each section. "
                f"Be specific, professional, and data-driven."
            )
        })
        report = str(result.get("output", ""))
    except Exception as exc:
        logger.warning("Report agent LLM call failed: %s — generating fallback report.", exc)
        report = _fallback_report(validation, statistical, model_selection, forecast)

    if not report.strip():
        report = _fallback_report(validation, statistical, model_selection, forecast)

    logger.info("Report generation complete. Length: %d chars", len(report))
    return report


def _fallback_report(
    validation: ValidationResult,
    statistical: StatisticalResult,
    model_selection: ModelSelectionResult,
    forecast: ForecastResult,
) -> str:
    return f"""## Executive Summary

This report presents the results of an automated time series forecasting analysis.
The {forecast.model_used} model was selected and achieved MAPE={forecast.mape:.2f}%,
RMSE={forecast.rmse:.4f}, MAE={forecast.mae:.4f}.

## Data Quality Assessment

The dataset contains {validation.row_count} observations at {validation.frequency} frequency.
{f"Issues detected: {'; '.join(validation.issues)}" if validation.issues else "No data quality issues detected."}

## Statistical Analysis

{statistical.summary}

ADF test: {'stationary' if statistical.is_stationary_adf else 'non-stationary'} (p={statistical.adf_p_value:.4f}).
KPSS test: {'stationary' if statistical.is_stationary_kpss else 'non-stationary'} (p={statistical.kpss_p_value:.4f}).
Trend: {'detected' if statistical.has_trend else 'not detected'} (slope={statistical.trend_slope:.6f}).
Seasonal period: {statistical.seasonal_period}.

## Model Selection & Rationale

Selected model: **{model_selection.selected_model}**

{model_selection.explanation[:600]}

## Forecast Results

Model: {forecast.model_used}
Forecast horizon: {len(forecast.forecast)} periods
RMSE: {forecast.rmse:.4f} | MAE: {forecast.mae:.4f} | MAPE: {forecast.mape:.2f}%

The 95% prediction intervals widen with the forecast horizon, reflecting accumulating uncertainty.

## Methodology & Limitations

This analysis uses classical statistical forecasting models (ARIMA, SARIMA, Holt-Winters).
Results assume the historical patterns remain stable in the future.
Structural breaks, regime changes, or external shocks are not accounted for.
"""
