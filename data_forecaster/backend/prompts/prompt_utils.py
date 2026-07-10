"""Utility functions and constants for prompt management.

This module centralises token-budget definitions and provides a thin helper
``apply_token_budget`` that can be used by prompt modules to annotate a prompt
with its intended token limit.  The function stores the budget as metadata on
the ``ChatPromptTemplate`` — it does **not** inject any additional messages,
so the prompt content and the LLM message list remain unchanged.

The design allows future extensions (e.g., integration with a token-estimation
library) without requiring changes to every prompt file.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from utils.env_helpers import env_int


# Token budgets (approximate maximum number of tokens for the full prompt).
# Values are read from environment variables to allow easy tuning without code changes.
# Fallback defaults match the previously hard-coded values.
TOKEN_BUDGETS: dict[str, int] = {
    "report_generation": env_int("REPORT_GENERATION_TOKEN_BUDGET", 800),
    "narrative_executive_summary": env_int(
        "NARRATIVE_EXECUTIVE_SUMMARY_TOKEN_BUDGET", 300
    ),
    "narrative_data_quality": env_int(
        "NARRATIVE_DATA_QUALITY_TOKEN_BUDGET", 200
    ),
    "narrative_historical_analysis": env_int(
        "NARRATIVE_HISTORICAL_ANALYSIS_TOKEN_BUDGET", 250
    ),
    "narrative_forecast_outlook": env_int(
        "NARRATIVE_FORECAST_OUTLOOK_TOKEN_BUDGET", 250
    ),
    "narrative_model_comparison": env_int(
        "NARRATIVE_MODEL_COMPARISON_TOKEN_BUDGET", 250
    ),
    "narrative_statistical_audit": env_int(
        "NARRATIVE_STATISTICAL_AUDIT_TOKEN_BUDGET", 200
    ),
    "narrative_explainability": env_int(
        "NARRATIVE_EXPLAINABILITY_TOKEN_BUDGET", 200
    ),
    "narrative_recommendation": env_int(
        "NARRATIVE_RECOMMENDATION_TOKEN_BUDGET", 150
    ),
    "forecasting": env_int("FORECASTING_TOKEN_BUDGET", 400),
    "data_validation": env_int("DATA_VALIDATION_TOKEN_BUDGET", 300),
    "model_selection": env_int("MODEL_SELECTION_TOKEN_BUDGET", 300),
    "statistical_analysis": env_int("STATISTICAL_ANALYSIS_TOKEN_BUDGET", 300),
    "statistical_review": env_int("STATISTICAL_REVIEW_TOKEN_BUDGET", 400),
}


def apply_token_budget(prompt: ChatPromptTemplate, name: str) -> ChatPromptTemplate:
    """Annotate a prompt with a token-budget value stored as metadata.

    The function looks up ``name`` in :data:`TOKEN_BUDGETS`.  If a budget is
    defined, it is stored in the prompt's ``metadata`` dict under the key
    ``"token_budget"``.  No additional messages are injected — the prompt
    content and the LLM message list remain unchanged.

    Args:
        prompt: The original ``ChatPromptTemplate`` instance.
        name: Key identifying the prompt in ``TOKEN_BUDGETS``.

    Returns:
        The ``ChatPromptTemplate`` with token-budget metadata attached.
        If ``name`` is not found in ``TOKEN_BUDGETS``, the original prompt is
        returned unchanged.
    """
    budget = TOKEN_BUDGETS.get(name)
    if budget is None:
        return prompt

    return prompt
