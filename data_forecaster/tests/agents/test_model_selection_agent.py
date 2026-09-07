"""Unit tests for the model selection agent parser.

Tests focus on the LLM output parsing logic, especially the handling
of markdown formatting and unicode hyphens that previously caused the
parser to override the LLM's explicit model choice.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.model_selection_agent import run_model_selection_agent
from schemas import StatisticalResult

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


class _FailingPrompt:
    """Mock prompt whose chain fails during invocation."""

    def __or__(self, other: object) -> _FailingChain:
        del other  # Unused.
        return _FailingChain()


class _FailingChain:
    """Mock LCEL chain that raises on invoke."""

    def invoke(self, inputs: dict) -> SimpleNamespace:
        del inputs  # Unused.
        raise RuntimeError("LLM down")


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
        assert "Business-readable selection summary" in result.explanation
        assert result.arima_rejected_reason is not None
        assert "does not model the recurring seasonal cycle" in (
            result.arima_rejected_reason
        )

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


# ── Deterministic Override Tests ──────────────────────────────────────────────


class TestDeterministicMetricOverride:
    """Tests for the deterministic best-metric override during retry."""

    def test_selects_best_metric_model_when_all_metrics_provided(
        self,
        seasonal_stat_result: StatisticalResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When all_metrics is provided, the best RMSE model is selected."""
        # SARIMA has the lowest RMSE, so it should be selected even though
        # the LLM mock would return Holt-Winters.
        response = SimpleNamespace(
            content="Selected model: Holt-Winters\n",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )
        _patch_llm(monkeypatch, response)

        all_metrics = {
            "Holt-Winters": {"RMSE": 0.11, "MAE": 0.09, "MAPE": 0.8},
            "ARIMA": {"RMSE": 0.10, "MAE": 0.08, "MAPE": 0.7},
            "SARIMA": {"RMSE": 0.09, "MAE": 0.07, "MAPE": 0.7},
        }
        result = run_model_selection_agent(
            seasonal_stat_result,
            review_feedback="Previous selection was suboptimal.",
            exclude_model="Holt-Winters",
            all_metrics=all_metrics,
        )
        # SARIMA has the lowest RMSE and is not excluded
        assert result.selected_model == "SARIMA"
        assert "empirical validation metrics" in result.explanation
        assert result.arima_rejected_reason is not None
        assert "Higher forecast error" in result.arima_rejected_reason
        assert "does not model the recurring seasonal cycle" in (
            result.arima_rejected_reason
        )

    def test_excluded_model_not_selected_even_with_best_metrics(
        self,
        seasonal_stat_result: StatisticalResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The excluded model must not be selected even if it has best metrics."""
        response = SimpleNamespace(
            content="Selected model: SARIMA\n",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )
        _patch_llm(monkeypatch, response)

        all_metrics = {
            "Holt-Winters": {"RMSE": 0.08, "MAE": 0.06, "MAPE": 0.5},
            "ARIMA": {"RMSE": 0.10, "MAE": 0.08, "MAPE": 0.7},
            "SARIMA": {"RMSE": 0.09, "MAE": 0.07, "MAPE": 0.7},
        }
        result = run_model_selection_agent(
            seasonal_stat_result,
            exclude_model="Holt-Winters",
            all_metrics=all_metrics,
        )
        # Holt-Winters has best RMSE but is excluded; SARIMA is next best
        assert result.selected_model == "SARIMA"
        assert result.selected_model != "Holt-Winters"

    def test_no_override_when_all_metrics_is_none(
        self,
        seasonal_stat_result: StatisticalResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without all_metrics, the LLM output is used (no deterministic override)."""
        response = SimpleNamespace(
            content="Selected model: Holt-Winters\n",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )
        _patch_llm(monkeypatch, response)

        result = run_model_selection_agent(seasonal_stat_result)
        assert result.selected_model == "Holt-Winters"


# ── Business Explanation Tests ───────────────────────────────────────────────


class TestBusinessModelExplanations:
    """Tests for business-readable selection and rejection explanations."""

    def test_heuristic_fallback_explains_rejected_models(
        self,
        seasonal_stat_result: StatisticalResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """LLM failure should still produce specific rejection reasons."""
        monkeypatch.setattr(
            "agents.model_selection_agent.get_llm",
            lambda temperature=0: SimpleNamespace(),
        )
        monkeypatch.setattr(
            "agents.model_selection_agent.MODEL_SELECTION_PROMPT",
            _FailingPrompt(),
        )

        result = run_model_selection_agent(seasonal_stat_result)

        assert result.selected_model == "SARIMA"
        assert "Heuristic fallback used" in result.explanation
        assert result.arima_rejected_reason is not None
        assert "plain ARIMA ignores seasonality" in result.arima_rejected_reason
        assert result.ewma_rejected_reason is not None
        assert "does not explicitly model seasonality" in result.ewma_rejected_reason


# ── Prophet awareness tests ───────────────────────────────────────────────────


class TestProphetAwareness:
    """Tests asserting the model-selection agent is aware of Prophet."""

    def test_parses_plain_text_prophet(
        self,
        seasonal_stat_result: StatisticalResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Plain text 'Selected model: Prophet' should parse to Prophet."""
        response = SimpleNamespace(
            content=(
                "Selected model: Prophet\n\n"
                "## Why this model was chosen\n"
                "Prophet models trend and seasonality natively."
            ),
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )
        _patch_llm(monkeypatch, response)

        result = run_model_selection_agent(seasonal_stat_result)
        assert result.selected_model == "Prophet"

    def test_parses_meta_prophet_variant(
        self,
        seasonal_stat_result: StatisticalResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """'Meta-Prophet' (unicode hyphen) normalizes to the Prophet model."""
        response = SimpleNamespace(
            content=(
                "**Selected model:** Meta‑Prophet\n\n"
                "## Why this model was chosen\n"
                "Meta-Prophet handles the seasonal cycle and changepoints."
            ),
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )
        _patch_llm(monkeypatch, response)

        result = run_model_selection_agent(seasonal_stat_result)
        assert result.selected_model == "Prophet"

    def test_prophet_in_models_registry(self) -> None:
        """Prophet is a candidate model in the canonical registry."""
        from forecasting import registry

        assert "Prophet" in registry.MODEL_NAMES
        assert "fit_fn" in registry.MODELS["Prophet"]

    def test_deterministic_override_selects_prophet_on_best_metrics(
        self,
        seasonal_stat_result: StatisticalResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Prophet with the lowest MASE is selected by the deterministic override."""
        response = SimpleNamespace(
            content="Selected model: Holt-Winters\n",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )
        _patch_llm(monkeypatch, response)

        all_metrics = {
            "Holt-Winters": {"RMSE": 0.12, "MAE": 0.10, "MAPE": 0.9, "MASE": 1.2},
            "ARIMA": {"RMSE": 0.10, "MAE": 0.08, "MAPE": 0.7, "MASE": 1.0},
            "SARIMA": {"RMSE": 0.09, "MAE": 0.07, "MAPE": 0.7, "MASE": 0.8},
            "Prophet": {"RMSE": 0.07, "MAE": 0.05, "MAPE": 0.5, "MASE": 0.5},
        }
        result = run_model_selection_agent(
            seasonal_stat_result,
            review_feedback="Previous selection was suboptimal.",
            exclude_model="Holt-Winters",
            all_metrics=all_metrics,
        )
        # Prophet has the lowest MASE and is not excluded
        assert result.selected_model == "Prophet"
        assert result.prophet_rejected_reason is None
        # Other models get a rejection reason
        assert result.arima_rejected_reason is not None
        assert "Higher forecast error" in result.arima_rejected_reason

    def test_heuristic_fallback_provides_prophet_rejection_reason(
        self,
        seasonal_stat_result: StatisticalResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Heuristic fallback on a seasonal series rejects Prophet with a reason."""
        monkeypatch.setattr(
            "agents.model_selection_agent.get_llm",
            lambda temperature=0: SimpleNamespace(),
        )
        monkeypatch.setattr(
            "agents.model_selection_agent.MODEL_SELECTION_PROMPT",
            _FailingPrompt(),
        )

        result = run_model_selection_agent(seasonal_stat_result)

        # SARIMA is the heuristic pick for a seasonal series; Prophet is not.
        assert result.selected_model == "SARIMA"
        assert result.prophet_rejected_reason is not None
        assert "lighter to fit than Prophet" in result.prophet_rejected_reason

    def test_suitability_summary_includes_prophet_assessment(
        self,
        seasonal_stat_result: StatisticalResult,
    ) -> None:
        """The suitability summary built for the LLM includes a Prophet section."""
        from agents.model_selection_agent import _build_suitability_summary

        summary = _build_suitability_summary(seasonal_stat_result)
        assert "Prophet Assessment:" in summary
        assert "Prophet models seasonality" in summary


# ── Disabled-model (registry) Tests ──────────────────────────────────────────


class TestDisabledModels:
    """Disabled models must be invisible to selection and heuristics."""

    def test_suitability_summary_omits_disabled(
        self,
        seasonal_stat_result: StatisticalResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Disabled models do not appear in the LLM suitability summary."""
        monkeypatch.setattr(
            "agents.model_selection_agent.get_enabled_models",
            lambda: ("ARIMA", "EWMA"),
        )
        from agents.model_selection_agent import _build_suitability_summary

        summary = _build_suitability_summary(seasonal_stat_result)

        assert "ARIMA Assessment:" in summary
        assert "EWMA Assessment:" in summary
        assert "Prophet Assessment:" not in summary
        assert "SARIMA Assessment:" not in summary

    def test_heuristic_never_selects_disabled(
        self,
        seasonal_stat_result: StatisticalResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """LLM failure falls back to an enabled model, never a disabled one."""
        monkeypatch.setattr(
            "agents.model_selection_agent.get_enabled_models",
            lambda: ("Holt-Winters", "EWMA"),
        )
        monkeypatch.setattr(
            "agents.model_selection_agent.get_llm",
            lambda temperature=0: SimpleNamespace(),
        )
        monkeypatch.setattr(
            "agents.model_selection_agent.MODEL_SELECTION_PROMPT",
            _FailingPrompt(),
        )

        result = run_model_selection_agent(seasonal_stat_result)

        assert result.selected_model in ("Holt-Winters", "EWMA")

    def test_single_enabled_model_is_selected(
        self,
        seasonal_stat_result: StatisticalResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With one enabled model, heuristics return it without error."""
        monkeypatch.setattr(
            "agents.model_selection_agent.get_enabled_models",
            lambda: ("EWMA",),
        )
        monkeypatch.setattr(
            "agents.model_selection_agent.get_llm",
            lambda temperature=0: SimpleNamespace(),
        )
        monkeypatch.setattr(
            "agents.model_selection_agent.MODEL_SELECTION_PROMPT",
            _FailingPrompt(),
        )

        result = run_model_selection_agent(seasonal_stat_result)

        assert result.selected_model == "EWMA"

    def test_llm_selecting_disabled_model_falls_back(
        self,
        seasonal_stat_result: StatisticalResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An LLM naming a disabled model is overridden by the fallback."""
        monkeypatch.setattr(
            "agents.model_selection_agent.get_enabled_models",
            lambda: ("ARIMA", "EWMA"),
        )
        _patch_llm(
            monkeypatch,
            SimpleNamespace(content="Selected model: Prophet\n\nBecause reasons."),
        )

        result = run_model_selection_agent(seasonal_stat_result)

        assert result.selected_model in ("ARIMA", "EWMA")
