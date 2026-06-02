from __future__ import annotations

from typing import Any
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from core.config import GEMINI_MODEL, USE_OLLAMA, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_API_KEY
from core.logging_config import get_logger
from schemas import ValidationResult

logger = get_logger(__name__)


def run_validation_agent(
    df: pd.DataFrame, date_col: str, value_col: str, freq: str, preflight_options: dict[str, Any] | None = None
) -> ValidationResult:
    """Run the data validation ReAct agent and return a ValidationResult."""

    series = df.set_index(date_col)[value_col]

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

    auto_mode = any(v == "Let AI Decide" for v in (preflight_options or {}).values())
    ai_instruction = (
        "\nNOTE: The user has selected 'Let AI Decide' for data quality remediation. "
        "Evaluate if the current state of the data is optimal for forecasting and "
        "specifically justify why the applied cleaning steps are the best path forward."
        if auto_mode else ""
    )

    quality_report = (
        f"DATA QUALITY SNAPSHOT:\n"
        f"- Length: {len(series)} observations\n"
        f"- Frequency: {freq}\n"
        f"- Missing Gaps: {missing_ts}\n"
        f"- Duplicates: {duplicate_ts}\n"
        f"- Null Values: {missing_vals}\n"
        f"- Regularity: {'Regular' if is_regular else 'Irregular'}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data validation expert. Summarize the data quality for forecasting."),
        ("human", (
            "Validate the current time series dataset based on these metrics:\n\n"
            "{report}\n\n"
            "Provide a professional summary of the data quality and suitability for forecasting.{ai_instruction}"
        ))
    ])

    try:
        chain = prompt | llm
        response = chain.invoke({
            "report": quality_report,
            "ai_instruction": ai_instruction
        })
        summary = response.content
        reasoning_steps = [
            {"thought": "Computing quality metrics in Python...", "observation": quality_report},
            {"thought": "Generating qualitative summary...", "observation": "Complete"}
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
