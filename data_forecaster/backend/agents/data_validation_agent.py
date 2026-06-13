from __future__ import annotations

from typing import Any
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from core.config import GEMINI_MODEL, USE_OLLAMA, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_API_KEY
from core.logging_config import get_logger
from schemas import ValidationResult
from prompts.data_validation_prompt import DATA_VALIDATION_PROMPT
from utils.data_cleaning import audit_series, validate_schema

logger = get_logger(__name__)


def run_validation_agent(
    df: pd.DataFrame, date_col: str, value_col: str, freq: str, preflight_options: dict[str, Any] | None = None
) -> ValidationResult:
    """Run the data validation ReAct agent and return a ValidationResult.

    The agent combines three information sources:

    1. **Heuristic metrics** — missing timestamps, duplicate timestamps,
       null values and regularity are computed directly from the series.
    2. **Structured audit** — ``audit_series`` produces a compact
       ``DatetimeIndex``/outlier snapshot for the LLM prompt.
    3. **Schema validation** — ``validate_schema`` checks frequency,
       missing‑rate and value range against a configurable contract.

    Args:
        df: Source DataFrame containing the time series.
        date_col: Name of the datetime column.
        value_col: Name of the numeric value column.
        freq: Inferred pandas frequency alias.
        preflight_options: Optional mapping of user‑selected preflight
            decisions (used to inform the LLM when the user chose
            ``"Let AI Decide"``).

    Returns:
        A populated :class:`ValidationResult` summarising quality issues
        and a natural‑language summary produced by the LLM.
    """

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

    options = preflight_options or {}
    auto_choices = [k for k, v in options.items() if v == "Let AI Decide"]
    
    # Build a more descriptive remediation summary for the LLM to evaluate
    remediation_log = []
    if "duplicate_strategy" in auto_choices:
        remediation_log.append("- Duplicates: Aggregated using 'mean' to preserve central tendency.")
    if "missing_strategy" in auto_choices:
        remediation_log.append("- Missing Values: Filled via 'linear time-interpolation' to maintain trend continuity.")
    if "frequency" in auto_choices:
        remediation_log.append(f"- Frequency: Automatically inferred as '{freq}' based on the median time delta.")

    ai_instruction = (
        "\nNOTE: The user has selected 'Let AI Decide' for data quality remediation. "
        "The system has applied the following treatments:\n"
        + "\n".join(remediation_log)
        + "\n\nAs the Analyst, evaluate if these steps were statistically sound for this specific dataset and justify the reasoning."
        if auto_choices else ""
    )

    audit_info = audit_series(series)
    # Default validation config – can be overridden by caller in the future
    validation_cfg = {
        "expected_freq": freq,
        "max_missing_rate": 0.05,
        "min_value": -1e9,
        "max_value": 1e9,
    }
    schema_report = validate_schema(series, validation_cfg)

    quality_report = (
        f"DATA QUALITY SNAPSHOT:\n"
        f"- Length: {audit_info['length']} observations\n"
        f"- Frequency: {audit_info['freq'] or freq}\n"
        f"- Missing Gaps: {missing_ts}\n"
        f"- Duplicates: {duplicate_ts}\n"
        f"- Null Values: {missing_vals}\n"
        f"- Regularity: {'Regular' if is_regular else 'Irregular'}\n"
        f"- Schema Pass: {schema_report.get('freq_regular', False)} (freq regular)\n"
        f"- Missing Rate OK: {schema_report.get('missing_below_threshold', False)}\n"
        f"- Values In Range: {schema_report.get('values_in_range', True)}"
    )

    prompt = DATA_VALIDATION_PROMPT

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
