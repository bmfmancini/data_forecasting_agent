"""Utility functions and constants for prompt management.

This module centralises token‑budget definitions and provides a thin helper
``apply_token_budget`` that can be used by prompt modules to annotate a prompt
with its intended token limit.  The function currently returns the original
``ChatPromptTemplate`` unchanged – the budget is retained for documentation
purposes and can be consulted by developers or tests.

The design allows future extensions (e.g., integration with a token‑estimation
library) without requiring changes to every prompt file.
"""

from __future__ import annotations

import os
from typing import Dict

from langchain_core.prompts import ChatPromptTemplate

# Token budgets (approximate maximum number of tokens for the full prompt).
# Values are read from environment variables to allow easy tuning without code changes.
# Fallback defaults match the previously hard‑coded values.
def _env_int(key: str, default: int) -> int:
    """Helper to read an integer environment variable with a fallback.

    Args:
        key: Environment variable name.
        default: Value to use if the variable is not set or cannot be parsed.

    Returns:
        The integer value from the environment or the default.
    """
    try:
        return int(os.getenv(key, default))
    except (ValueError, TypeError):
        return default


TOKEN_BUDGETS: Dict[str, int] = {
    "report_generation": _env_int("REPORT_GENERATION_TOKEN_BUDGET", 800),
    "forecasting": _env_int("FORECASTING_TOKEN_BUDGET", 400),
    "data_validation": _env_int("DATA_VALIDATION_TOKEN_BUDGET", 300),
    "model_selection": _env_int("MODEL_SELECTION_TOKEN_BUDGET", 300),
    "statistical_analysis": _env_int("STATISTICAL_ANALYSIS_TOKEN_BUDGET", 300),
}


def apply_token_budget(prompt: ChatPromptTemplate, name: str) -> ChatPromptTemplate:
    """Annotate a prompt with a token‑budget comment.

    The function looks up ``name`` in :data:`TOKEN_BUDGETS`.  If a budget is
    defined, a comment is attached to the ``ChatPromptTemplate``'s ``messages``
    list as a ``system`` message that does not affect LLM behaviour but serves as
    documentation.  The original prompt is otherwise returned unchanged.

    Args:
        prompt: The original ``ChatPromptTemplate`` instance.
        name: Key identifying the prompt in ``TOKEN_BUDGETS``.

    Returns:
        The (potentially modified) ``ChatPromptTemplate``.
    """
    budget = TOKEN_BUDGETS.get(name)
    if budget is None:
        return prompt

    # Insert a non‑intrusive system message describing the budget.  This keeps the
    # prompt contract intact while providing visibility for developers and tests.
    budget_message = (
        f"[TOKEN_BUDGET: {budget} tokens] – this is a documentation comment."
    )
    # Prepend the budget message to the existing messages.
    new_messages = [("system", budget_message)] + list(prompt.messages)
    return ChatPromptTemplate.from_messages(new_messages)
