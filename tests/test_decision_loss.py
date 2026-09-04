"""Tests for business-aware decision-loss selection."""

from __future__ import annotations

import pandas as pd

from agents.forecasting_agent import (
    _loss_recommendation_rationale,
    _resolve_loss_preference,
)
from utils.preflight import run_preflight_checks


def test_explicit_loss_is_preserved() -> None:
    """An explicit business choice must not be overridden by LLM text."""
    assert _resolve_loss_preference(
        "mae", "Recommended decision loss: rmse"
    ) == ("mae", "user_selected")


def test_auto_loss_uses_constrained_llm_recommendation() -> None:
    """Auto accepts only the supported metric on the labelled response line."""
    assert _resolve_loss_preference(
        "auto", "Analysis\nRecommended decision loss: WAPE\n"
    ) == ("wape", "llm_recommended")


def test_auto_loss_falls_back_safely_when_recommendation_is_missing() -> None:
    """An unavailable or malformed recommendation falls back transparently."""
    assert _resolve_loss_preference("auto", "Use asymmetric loss") == (
        "mase",
        "llm_unavailable_fallback",
    )


def test_llm_loss_rationale_is_captured_from_labelled_line() -> None:
    """The evidence retains a concise explanation, not only the chosen metric."""
    text = (
        "Recommended decision loss: rmse\n"
        "Decision-loss rationale: Large misses can cause costly stockouts.\n"
    )
    assert _loss_recommendation_rationale("rmse", "llm_recommended", text) == (
        "Large misses can cause costly stockouts."
    )


def test_preflight_defaults_decision_loss_to_auto() -> None:
    """Users receive assistance by default rather than a technical guess."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=12, freq="MS"),
            "value": range(12),
        }
    )
    result = run_preflight_checks(frame, "date", "value", 3)
    decision = next(item for item in result.decisions if item.key == "loss_metric")

    assert result.defaults["loss_metric"] == "auto"
    assert decision.default == "auto"
    assert decision.options == ["auto", "rmse", "mae", "wape", "mase"]
