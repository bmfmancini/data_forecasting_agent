"""Unit tests for the model selection agent parser.

Tests focus on the LLM output parsing logic, especially the handling
of markdown formatting and unicode hyphens that previously caused the
parser to override the LLM's explicit model choice.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

# Add the backend directory to the path so that backend-internal imports
# (e.g. ``from core.logging_config import ...``) resolve correctly.
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")),
)

from agents.model_selection_agent import run_model_selection_agent  # noqa: E402
from schemas import StatisticalResult  # noqa: E402

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def seasonal_stat_result() -> StatisticalResult:
    """A statistical result with seasonality and trend."""
    return StatisticalResult(
        is_stationary_adf=False,
        adf_statistic=-1.5,
        adf_p_value=0.45,
        is_stationary_kpss=False,
        kpss_statistic=0.8,
        kpss_p_value=0.01,
        has_trend=True,
        trend_slope=2.65,
        outlier_count=2,
        outlier_ratio=0.02,
        is_white_noise=False,
        white_noise_p_value=0.001,
        recommended_remediation=["box_cox"],
        seasonal_period=12,
        dominant_period=12.0,
        summary="Non-stationary seasonal series with trend.",
    )


class _MockPrompt:
    """Mock ChatPromptTemplate that supports the ``|`` operator."""

    def __init__(self, response: SimpleNamespace) -> None:
        self._response = response

    def __or__(self, other: object) -> _MockChain:
        del other  # Unused.
        return _MockChain(self._response)


class _MockChain:
    """Mock LCEL chain that returns a pre-set response on invoke."""

    def __init__(self, response: SimpleNamespace) -> None:
        self._response = response

    def invoke(self, inputs: dict) -> SimpleNamespace:
        del inputs  # Unused.
        return self._response


def _patch_llm(
    monkeypatch: pytest.MonkeyPatch,
    response: SimpleNamespace,
) -> None:
    """Patch get_llm and MODEL_SELECTION_PROMPT for deterministic tests."""
    monkeypatch.setattr(
        "agents.model_selection_agent.get_llm",
        lambda temperature=0: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "agents.model_selection_agent.MODEL_SELECTION_PROMPT",
        _MockPrompt(response),
    )


# ── Parser Tests ──────────────────────────────────────────────────────────────


class TestModelSelectionParser:
    """Tests for the model selection LLM output parser."""

    def test_parses_plain_text_selected_model(
        self,
        seasonal_stat_result: StatisticalResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Plain text 'Selected model: SARIMA' should parse correctly."""
        response = SimpleNamespace(
            content=(
                "Selected model: SARIMA\n\n"
                "## Why this model was chosen\n"
                "SARIMA handles seasonality."
            ),
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )
        _patch_llm(monkeypatch, response)

        result = run_model_selection_agent(seasonal_stat_result)
        assert result.selected_model == "SARIMA"

    def test_parses_markdown_bold_selected_model(
        self,
        seasonal_stat_result: StatisticalResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Markdown bold '**Selected model:**' should parse correctly."""
        response = SimpleNamespace(
            content=(
                "**Selected model:** Holt-Winters\n\n"
                "## Why this model was chosen\n"
                "Holt-Winters natively incorporates seasonality."
            ),
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )
        _patch_llm(monkeypatch, response)

        result = run_model_selection_agent(seasonal_stat_result)
        assert result.selected_model == "Holt-Winters"

    def test_parses_unicode_hyphen_holt_winters(
        self,
        seasonal_stat_result: StatisticalResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unicode non-breaking hyphen (U+2011) should be normalized."""
        # This is the exact bug from production: the LLM used a unicode
        # hyphen in "Holt‑Winters" which caused the parser to miss the
        # match and fall back to scanning the first 100 chars.
        response = SimpleNamespace(
            content=(
                "**Selected model:** Holt\u2011Winters\n\n"
                "## Why this model was chosen\n"
                "Holt\u2011Winters natively incorporates seasonality."
            ),
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )
        _patch_llm(monkeypatch, response)

        result = run_model_selection_agent(seasonal_stat_result)
        assert result.selected_model == "Holt-Winters"

    def test_does_not_override_explicit_choice_with_fallback_scan(
        self,
        seasonal_stat_result: StatisticalResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fallback scan must not override an explicit 'Selected model' line.

        Previously, when the exact match failed (e.g. due to markdown or
        unicode), the fallback scan would search the first 100 chars and
        pick whichever model name appeared first — often SARIMA from the
        suitability text, overriding the LLM's actual choice.
        """
        response = SimpleNamespace(
            content=(
                "**Selected model:** Holt\u2011Winters\n\n"
                "## Why this model was chosen\n"
                "SARIMA Assessment: good for seasonality.\n"
                "Holt\u2011Winters natively incorporates seasonality."
            ),
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )
        _patch_llm(monkeypatch, response)

        result = run_model_selection_agent(seasonal_stat_result)
        assert result.selected_model == "Holt-Winters"

    def test_fallback_scans_selected_model_line_only(
        self,
        seasonal_stat_result: StatisticalResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fallback should scan the 'Selected model' line, not first 100 chars."""
        # No exact "Selected model: X" match, but a line with "selected model"
        # that contains the model name.
        response = SimpleNamespace(
            content=(
                "The selected model is ARIMA for this series.\n\n"
                "## Why this model was chosen\n"
                "SARIMA Assessment: good for seasonality."
            ),
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )
        _patch_llm(monkeypatch, response)

        result = run_model_selection_agent(seasonal_stat_result)
        assert result.selected_model == "ARIMA"

    def test_parses_lowercase_selected_model(
        self,
        seasonal_stat_result: StatisticalResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lowercase 'selected model: arima' should parse via case-insensitive match."""
        response = SimpleNamespace(
            content=(
                "selected model: arima\n\n"
                "## Why this model was chosen\n"
                "ARIMA handles autocorrelation well."
            ),
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )
        _patch_llm(monkeypatch, response)

        result = run_model_selection_agent(seasonal_stat_result)
        assert result.selected_model == "ARIMA"
