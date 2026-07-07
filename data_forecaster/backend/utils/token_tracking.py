"""Utilities for capturing and estimating LLM token usage.

Provides helpers to extract native token usage metadata from LangChain
LLM responses and fall back to a character-based heuristic estimate when
the provider does not report usage metadata.
"""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from core.logging_config import get_logger

logger = get_logger(__name__)

# Heuristic conversion factor: approximately 4 characters per token.
_CHARS_PER_TOKEN: int = 4


def extract_token_usage(
    response: Any,
    input_text: str | None = None,
) -> dict[str, int]:
    """Extract token usage from an LLM response, with heuristic fallback.

    Attempts to read native ``usage_metadata`` from the LangChain response
    object. If unavailable, estimates token counts from the response content
    (and optionally the input text) using a ~4 chars/token heuristic.

    Args:
        response: The LangChain LLM response object (e.g. ``AIMessage``).
            May be ``None`` or any object; this function never raises.
        input_text: Optional formatted prompt string used to estimate
            input tokens when native metadata is unavailable.

    Returns:
        A dict with integer keys ``input_tokens``, ``output_tokens``,
        and ``total_tokens``. Values are never negative and the dict is
        always returned (never raises).
    """
    try:
        usage = getattr(response, "usage_metadata", None)
        if isinstance(usage, dict) and usage.get("total_tokens"):
            return {
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            }
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to read usage_metadata: %s", exc)

    # ── Heuristic fallback ────────────────────────────────────────────────
    content = ""
    try:
        content = getattr(response, "content", "") or ""
        if not isinstance(content, str):
            content = str(content)
    except Exception:  # pragma: no cover - defensive
        content = ""

    output_tokens = max(1, len(content) // _CHARS_PER_TOKEN) if content else 0
    input_tokens = (
        max(1, len(input_text) // _CHARS_PER_TOKEN)
        if input_text
        else 0
    )
    total = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total,
    }


def estimate_input_text(
    prompt_template: ChatPromptTemplate,
    inputs: dict[str, Any],
) -> str:
    """Format a ChatPromptTemplate with inputs and return the text.

    Used to provide input-text context for the heuristic token estimator
    when native usage metadata is unavailable.

    Args:
        prompt_template: The LangChain chat prompt template.
        inputs: Dictionary of variables to format the template with.

    Returns:
        The formatted prompt as a concatenated string. Returns an empty
        string if formatting fails (never raises).
    """
    try:
        messages = prompt_template.format_messages(**inputs)
        return "\n".join(
            msg.content for msg in messages if hasattr(msg, "content")
        )
    except Exception as exc:
        logger.debug("estimate_input_text formatting failed: %s", exc)
        return ""