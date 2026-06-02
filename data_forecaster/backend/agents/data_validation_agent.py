from __future__ import annotations

from typing import Any
import pandas as pd
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from core.config import GEMINI_MODEL, USE_OLLAMA, OLLAMA_BASE_URL, OLLAMA_MODEL
from core.logging_config import get_logger
from schemas import ValidationResult

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


def run_validation_agent(
    df: pd.DataFrame, date_col: str, value_col: str, freq: str, preflight_options: dict[str, Any] | None = None
) -> ValidationResult:
    """Run the data validation ReAct agent and return a ValidationResult."""

    series = df.set_index(date_col)[value_col]

    # ── Tool definitions (closed over df / series) ───────────────────────────

    @tool
    def get_full_data_quality_report(command: str) -> str:
        """Returns a complete technical report of duplicates, missing values, gaps, and frequency."""
        diffs = series.index.to_series().diff().dropna()
        mode_diff = diffs.mode()[0] if len(diffs) > 0 else "N/A"
        gaps = int((diffs > mode_diff * 1.5).sum()) if mode_diff != "N/A" else 0
        dupes = int(df[date_col].duplicated().sum())
        missing = int(series.isna().sum())
        n = len(series)
        
        return (
            f"DATA QUALITY SNAPSHOT:\n"
            f"- Length: {n} observations\n"
            f"- Frequency: {freq}\n"
            f"- Missing Gaps: {gaps}\n"
            f"- Duplicates: {dupes}\n"
            f"- Null Values: {missing}\n"
            f"- Regularity: {'Regular' if (diffs.nunique() == 1 if n > 1 else True) else 'Irregular'}"
        )

    # ── Compute structured values directly ───────────────────────────────────
    diffs = series.index.to_series().diff().dropna()
    mode_diff = diffs.mode()[0] if len(diffs) > 0 else None
    missing_ts = int((diffs > mode_diff * 1.5).sum()) if mode_diff is not None else 0
    duplicate_ts = int(df[date_col].duplicated().sum())
    missing_vals = int(series.isna().sum())
    is_regular = (diffs.nunique() == 1) if len(diffs) > 0 else True

    issues: list[str] = []
    if missing_ts > 0:
        issues.append(f"{missing_ts} timestamp gap(s) detected.")
    if duplicate_ts > 0:
        issues.append(f"{duplicate_ts} duplicate timestamp(s) detected.")
    if missing_vals > 0:
        issues.append(f"{missing_vals} missing value(s) in '{value_col}'.")
    if not is_regular:
        issues.append("Irregular time intervals detected.")
    if len(series) < 20:
        issues.append("Very short series — forecasting results may be unreliable.")

    # ── Run ReAct agent for qualitative summary ───────────────────────────────
    tools_list = [get_full_data_quality_report]

    if USE_OLLAMA:
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    else:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

    agent = create_react_agent(llm, tools_list, _REACT_PROMPT)
    executor = AgentExecutor(
        agent=agent, tools=tools_list, verbose=False, return_intermediate_steps=True,
        max_iterations=6, handle_parsing_errors=True,
    )

    auto_mode = any(v == "Let AI Decide" for v in (preflight_options or {}).values())
    ai_instruction = (
        "\nNOTE: The user has selected 'Let AI Decide' for data quality remediation. "
        "Evaluate if the current state of the data is optimal for forecasting and "
        "specifically justify why the applied cleaning steps are the best path forward."
        if auto_mode else ""
    )

    reasoning_steps: list[dict[str, Any]] = []
    try:
        result = executor.invoke({
            "input": (
                "Validate the current time series dataset. You MUST use the "
                "get_full_data_quality_report tool to retrieve the technical metrics. "
                "Then, provide a professional summary of the data quality and suitability "
                "for forecasting."
                f"{ai_instruction}"
            )
        })
        summary = str(result.get("output", "Validation complete."))
        reasoning_steps = [
            {"thought": a.log, "observation": str(o)} for a, o in result.get("intermediate_steps", [])
        ]
    except Exception as exc:
        logger.warning("Validation agent LLM call failed: %s", exc)
        summary = f"Validation complete. Issues found: {len(issues)}. " + " ".join(issues)
        reasoning_steps = [{
            "thought": f"Validation agent failed: {str(exc)}",
            "observation": "Proceeding with heuristic-based validation summary."
        }]

    logger.info("Validation complete. Issues: %s", issues)

    return ValidationResult(
        is_valid=len([i for i in issues if "duplicate" in i or "missing value" in i]) == 0,
        row_count=len(series),
        missing_timestamps=missing_ts,
        duplicate_timestamps=duplicate_ts,
        missing_values=missing_vals,
        is_regular=is_regular,
        frequency=freq,
        frequency_alias=freq,
        issues=issues,
        summary=summary,
        reasoning_steps=reasoning_steps,
    )
