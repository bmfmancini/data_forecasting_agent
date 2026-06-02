from __future__ import annotations

import pandas as pd
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from core.config import GROQ_API_KEY
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
Thought:{agent_scratchpad}"""
)


def run_validation_agent(
    df: pd.DataFrame, date_col: str, value_col: str, freq: str
) -> ValidationResult:
    """Run the data validation ReAct agent and return a ValidationResult."""

    series = df.set_index(date_col)[value_col]

    # ── Tool definitions (closed over df / series) ───────────────────────────

    @tool
    def check_missing_timestamps(command: str) -> str:
        """Check for gaps in the time series timestamps."""
        diffs = series.index.to_series().diff().dropna()
        if len(diffs) == 0:
            return "Only one observation — cannot check gaps."
        mode_diff = diffs.mode()[0]
        gaps = int((diffs > mode_diff * 1.5).sum())
        return f"Missing timestamp gaps detected: {gaps}. Expected interval: {mode_diff}."

    @tool
    def check_duplicates(command: str) -> str:
        """Check for duplicate timestamps in the time series."""
        dupes = int(df[date_col].duplicated().sum())
        return f"Duplicate timestamps: {dupes}."

    @tool
    def check_missing_values(command: str) -> str:
        """Check for missing (NaN) values in the value column."""
        missing = int(series.isna().sum())
        pct = missing / len(series) * 100 if len(series) > 0 else 0
        return f"Missing values: {missing} ({pct:.1f}% of {len(series)} observations)."

    @tool
    def check_irregular_intervals(command: str) -> str:
        """Check whether the time series has regular (evenly spaced) intervals."""
        diffs = series.index.to_series().diff().dropna()
        if len(diffs) == 0:
            return "Cannot assess — too few observations."
        unique_diffs = diffs.nunique()
        is_regular = unique_diffs == 1
        return (
            f"Interval regularity: {'regular' if is_regular else 'irregular'}. "
            f"Unique interval counts: {unique_diffs}."
        )

    @tool
    def detect_frequency(command: str) -> str:
        """Detect the inferred pandas frequency alias of the time series."""
        return f"Detected frequency alias: '{freq}'."

    @tool
    def assess_size(command: str) -> str:
        """Assess whether the series is large enough for reliable forecasting."""
        n = len(series)
        if n < 20:
            adequacy = "Too short — results unreliable (< 20 observations)."
        elif n < 50:
            adequacy = "Marginal — some models may be unreliable (< 50 observations)."
        else:
            adequacy = "Adequate for forecasting."
        return f"Series length: {n} observations. Assessment: {adequacy}"

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
    tools_list = [
        check_missing_timestamps, check_duplicates, check_missing_values,
        check_irregular_intervals, detect_frequency, assess_size,
    ]
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY, temperature=0)
    agent = create_react_agent(llm, tools_list, _REACT_PROMPT)
    executor = AgentExecutor(
        agent=agent, tools=tools_list, verbose=False,
        max_iterations=12, handle_parsing_errors=True,
    )

    try:
        result = executor.invoke({
            "input": (
                "Use all available tools to thoroughly validate this time series dataset. "
                "Check for missing timestamps, duplicates, missing values, irregular intervals, "
                "detect the frequency, and assess the dataset size. "
                "Provide a concise summary of the data quality."
            )
        })
        summary = str(result.get("output", "Validation complete."))
    except Exception as exc:
        logger.warning("Validation agent LLM call failed: %s", exc)
        summary = f"Validation complete. Issues found: {len(issues)}. " + " ".join(issues)

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
    )
