"""Traditional Forecasting mode tests — every agent skips the LLM entirely.

Zero-LLM-call assertions patch ``get_llm`` at each *use site* (agents
import it directly) with ``MagicMock`` objects and assert they were never
called.  Raising side effects are insufficient — the agents' broad
``except`` blocks would swallow them, so construction itself must never
happen in traditional mode.

Mid-run deployment-wide disable coverage lives here too: a queued AI
forecast that is running when an administrator disables AI features hits
``LLMDisabledError`` at construction inside the agent's existing fallback
path, so it completes deterministically instead of crashing.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from forecasting.contracts import BacktestEvaluation, ForecastMetrics
from schemas import (
    ForecastResult,
    ModelSelectionResult,
    StatisticalResult,
    ValidationResult,
)

import agents.data_validation_agent as data_validation_agent
import agents.forecasting_agent as forecasting_agent
import agents.report_generation_agent as report_generation_agent
import agents.statistical_analysis_agent as statistical_analysis_agent
from exceptions import LLMDisabledError


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A clean monthly series with no validation issues."""
    n = 48
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="MS"),
            "value": 100.0
            + 0.5 * np.arange(n)
            + 5.0 * np.sin(2 * np.pi * np.arange(n) / 12)
            + rng.normal(scale=1.0, size=n),
        }
    )


@pytest.fixture
def clean_series(clean_df: pd.DataFrame) -> pd.Series:
    return clean_df.set_index("date")["value"]


@pytest.fixture
def validation_result() -> ValidationResult:
    return ValidationResult(
        is_valid=True,
        row_count=48,
        missing_timestamps=0,
        duplicate_timestamps=0,
        missing_values=0,
        is_regular=True,
        frequency="MS",
        frequency_alias="M",
        issues=[],
        summary="Data is clean and regular.",
    )


@pytest.fixture
def statistical_result() -> StatisticalResult:
    return StatisticalResult(
        is_stationary_adf=False,
        adf_statistic=-1.5,
        adf_p_value=0.45,
        is_stationary_kpss=False,
        kpss_statistic=0.8,
        kpss_p_value=0.01,
        has_trend=True,
        trend_slope=0.5,
        seasonal_period=12,
        dominant_period=12.0,
        summary="Non-stationary seasonal series with trend.",
    )


@pytest.fixture
def model_selection() -> ModelSelectionResult:
    return ModelSelectionResult(
        selected_model="ARIMA",
        explanation="ARIMA selected by the user (Traditional Forecasting).",
        holt_winters_rejected_reason="Not selected.",
        arima_rejected_reason=None,
        sarima_rejected_reason="Not selected.",
        ewma_rejected_reason="Not selected.",
        selection_method="forced",
    )


@pytest.fixture
def forecast_result() -> ForecastResult:
    return ForecastResult(
        status="ok",
        model_used="ARIMA",
        forecast=[110.0, 111.0, 112.0],
        lower_ci=[100.0, 101.0, 102.0],
        upper_ci=[120.0, 121.0, 122.0],
        forecast_dates=["2024-01-01", "2024-02-01", "2024-03-01"],
        rmse=5.0,
        mae=4.0,
        mape=4.0,
    )


@pytest.fixture
def all_metrics() -> dict[str, dict[str, float]]:
    return {
        "ARIMA": {"RMSE": 5.0, "MAE": 4.0, "MAPE": 4.0},
        "Holt-Winters": {"RMSE": 6.0, "MAE": 5.0, "MAPE": 5.0},
    }


def _fake_backtests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace the (slow) rolling-origin backtest with instant evaluations.

    The forecasting agent's LLM loss-recommendation block is what is under
    test; the numerical backtesting engine is covered elsewhere.
    """

    def fake_evaluate(
        series: pd.Series,
        candidates: dict[str, Any],
        config: Any,
    ) -> dict[str, BacktestEvaluation]:
        del series, config
        return {
            name: BacktestEvaluation(
                model_name=name,
                pooled_metrics=ForecastMetrics(
                    rmse=5.0, mae=4.0, mape=4.0, mase=1.2
                ),
                n_evaluated=3,
                n_failed_origins=0,
                is_rankable=True,
            )
            for name in candidates
        }

    monkeypatch.setattr(forecasting_agent, "evaluate_candidates", fake_evaluate)


# ── Data validation agent ─────────────────────────────────────────────────────


class TestDataValidationAgentTraditional:
    """``run_validation_agent(use_llm=False)`` never touches the LLM."""

    def test_traditional_mode_never_constructs_llm(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clean_df: pd.DataFrame,
    ) -> None:
        llm = MagicMock()
        monkeypatch.setattr(data_validation_agent, "get_llm", llm)

        result = data_validation_agent.run_validation_agent(
            clean_df, "date", "value", "MS", use_llm=False
        )

        llm.assert_not_called()
        assert result.is_valid is True
        assert result.summary.strip() == "Validation complete. Issues found: 0."
        assert result.token_usage == {}
        assert result.reasoning_steps[-1]["thought"] == (
            "Traditional Forecasting: deterministic validation summary used "
            "(LLM skipped by request)."
        )

    def test_midrun_disable_falls_back_deterministically(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clean_df: pd.DataFrame,
    ) -> None:
        """An AI-mode run whose LLM construction raises LLMDisabledError
        keeps the deterministic heuristic summary — never crashes."""
        monkeypatch.setattr(
            data_validation_agent, "get_llm", MagicMock(side_effect=LLMDisabledError)
        )

        result = data_validation_agent.run_validation_agent(
            clean_df, "date", "value", "MS", use_llm=True
        )

        assert result.is_valid is True
        assert result.summary.strip() == "Validation complete. Issues found: 0."


# ── Statistical analysis agent ─────────────────────────────────────────────────


class TestStatisticalAnalysisAgentTraditional:
    """``run_statistical_agent(use_llm=False)`` never touches the LLM."""

    def test_traditional_mode_never_constructs_llm(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clean_series: pd.Series,
    ) -> None:
        llm = MagicMock()
        monkeypatch.setattr(statistical_analysis_agent, "get_llm", llm)

        result = statistical_analysis_agent.run_statistical_agent(
            clean_series,
            seasonal_period=12,
            user_domain="Skip / Let AI Guess",
            use_llm=False,
        )

        llm.assert_not_called()
        assert result.summary.startswith("Stationarity classification:")
        assert result.domain == "General / Unknown"
        assert result.token_usage == {}
        assert result.reasoning_steps[-1]["thought"] == (
            "Traditional Forecasting: deterministic statistical summary used "
            "(LLM skipped by request)."
        )

    def test_traditional_mode_preserves_user_domain(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clean_series: pd.Series,
    ) -> None:
        monkeypatch.setattr(
            statistical_analysis_agent, "get_llm", MagicMock()
        )

        result = statistical_analysis_agent.run_statistical_agent(
            clean_series, seasonal_period=12, user_domain="Retail", use_llm=False
        )

        assert result.domain == "Retail"

    def test_midrun_disable_falls_back_deterministically(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clean_series: pd.Series,
    ) -> None:
        monkeypatch.setattr(
            statistical_analysis_agent,
            "get_llm",
            MagicMock(side_effect=LLMDisabledError),
        )

        result = statistical_analysis_agent.run_statistical_agent(
            clean_series, seasonal_period=12, use_llm=True
        )

        assert result.summary.startswith("Stationarity classification:")


# ── Forecasting agent ─────────────────────────────────────────────────────────


class TestForecastingAgentTraditional:
    """``run_forecasting_agent(use_llm=False)`` keeps the default loss."""

    def test_traditional_mode_never_constructs_llm(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clean_series: pd.Series,
        statistical_result: StatisticalResult,
        model_selection: ModelSelectionResult,
    ) -> None:
        llm = MagicMock()
        monkeypatch.setattr(forecasting_agent, "get_llm", llm)
        _fake_backtests(monkeypatch)

        result, _ = forecasting_agent.run_forecasting_agent(
            clean_series,
            model_selection,
            statistical_result,
            3,
            "MS",
            loss_preference="auto",
            use_llm=False,
        )

        llm.assert_not_called()
        decision_loss = result.validation_design["decision_loss"]
        assert decision_loss["resolution_source"] != "llm_recommended"
        traditional_steps = [
            step
            for step in result.reasoning_steps
            if "Traditional Forecasting" in step["thought"]
        ]
        assert traditional_steps, "Traditional reasoning step missing"

    def test_ai_mode_requests_loss_recommendation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clean_series: pd.Series,
        statistical_result: StatisticalResult,
        model_selection: ModelSelectionResult,
    ) -> None:
        """AI mode still constructs an LLM for the loss recommendation."""
        # ``FORECASTING_PROMPT | llm`` coerces the mock into a RunnableLambda,
        # so the LLM is *called* (not ``.invoke``d) with the prompt output.
        response = SimpleNamespace(
            content="Recommended decision loss: rmse",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        )
        llm = MagicMock(return_value=response)
        monkeypatch.setattr(forecasting_agent, "get_llm", llm)
        _fake_backtests(monkeypatch)

        result, _ = forecasting_agent.run_forecasting_agent(
            clean_series,
            model_selection,
            statistical_result,
            3,
            "MS",
            loss_preference="auto",
            use_llm=True,
        )

        llm.assert_called_once()
        # The contrast under test: AI mode constructs an LLM for the loss
        # recommendation.  (Which metric wins also depends on the faked
        # backtest evaluations, so only the call is asserted here.)


# ── Report generation agent ────────────────────────────────────────────────────


class TestReportGenerationAgentTraditional:
    """``run_report_agent(use_llm=False)`` uses the deterministic templates."""

    @staticmethod
    def _run(
        monkeypatch: pytest.MonkeyPatch,
        validation_result: ValidationResult,
        statistical_result: StatisticalResult,
        model_selection: ModelSelectionResult,
        forecast_result: ForecastResult,
        use_llm: bool,
    ) -> tuple[Any, list[dict[str, Any]], dict[str, int]]:
        narratives = MagicMock()
        monkeypatch.setattr(
            report_generation_agent, "generate_narratives", narratives
        )
        narrative_llm = MagicMock()
        monkeypatch.setattr(
            "report.narrative.get_llm", narrative_llm, raising=False
        )
        report, reasoning, _strategy, tokens = report_generation_agent.run_report_agent(
            validation_result,
            statistical_result,
            model_selection,
            forecast_result,
            rag_kb=MagicMock(),
            use_llm=use_llm,
        )
        return report, reasoning, tokens, narratives, narrative_llm

    def test_traditional_mode_never_constructs_llm(
        self,
        monkeypatch: pytest.MonkeyPatch,
        validation_result: ValidationResult,
        statistical_result: StatisticalResult,
        model_selection: ModelSelectionResult,
        forecast_result: ForecastResult,
    ) -> None:
        report, reasoning, tokens, narratives, narrative_llm = self._run(
            monkeypatch,
            validation_result,
            statistical_result,
            model_selection,
            forecast_result,
            use_llm=False,
        )

        narratives.assert_not_called()
        narrative_llm.assert_not_called()
        # Deterministic template narratives fill every section...
        assert report.metadata.llm_narrative_fallback is True
        assert report.metadata.llm_fallback_sections
        assert all(
            section.narrative
            for section in [
                report.executive_summary,
                report.data_quality,
                report.historical_analysis,
                report.forecast_outlook,
                report.model_comparison,
                report.statistical_audit,
                report.explainability,
            ]
        )
        # ...but the run is not marked as an LLM failure: zero tokens and
        # a neutral reasoning step with no ``llm_fallback`` key.
        assert tokens == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        final_step = reasoning[-1]
        assert "Traditional Forecasting" in final_step["thought"]
        assert not final_step.get("llm_fallback")

    def test_llm_failure_still_flags_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        validation_result: ValidationResult,
        statistical_result: StatisticalResult,
        model_selection: ModelSelectionResult,
        forecast_result: ForecastResult,
    ) -> None:
        """Genuine AI failures are unchanged: fallback + llm_fallback flag."""
        narratives = MagicMock(side_effect=RuntimeError("provider outage"))
        monkeypatch.setattr(
            report_generation_agent, "generate_narratives", narratives
        )

        report, reasoning, _strategy, tokens = report_generation_agent.run_report_agent(
            validation_result,
            statistical_result,
            model_selection,
            forecast_result,
            rag_kb=MagicMock(),
            use_llm=True,
        )

        narratives.assert_called_once()
        assert report.metadata.llm_narrative_fallback is True
        assert tokens == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        assert reasoning[-1]["llm_fallback"] is True