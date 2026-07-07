"""Unit tests for the statistical review (QA) agent."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

# Add the backend directory to the path so that backend-internal imports
# (e.g. ``from core.logging_config import ...``) resolve correctly.
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "backend")
    ),
)

from agents.statistical_review_agent import (  # noqa: E402
    _compute_verdict,
    _deterministic_pre_check,
    _parse_endorsements,
    _parse_flags,
    _parse_summary,
    _parse_verdict,
    run_statistical_review_agent,
)
from schemas import (  # noqa: E402
    ForecastResult,
    ModelSelectionResult,
    StatisticalResult,
)


# ── Mock helpers ──────────────────────────────────────────────────────────────


class _MockChain:
    """Mock LCEL chain that returns a pre-set response on invoke."""

    def __init__(self, response: SimpleNamespace) -> None:
        self._response = response

    def invoke(self, inputs: dict) -> SimpleNamespace:
        del inputs  # Unused.
        return self._response


class _MockPrompt:
    """Mock ChatPromptTemplate that supports the ``|`` operator."""

    def __init__(self, response: SimpleNamespace) -> None:
        self._response = response

    def __or__(self, other: object) -> _MockChain:
        del other  # Unused.
        return _MockChain(self._response)


def _patch_llm(
    monkeypatch: pytest.MonkeyPatch,
    response: SimpleNamespace,
) -> None:
    """Patch get_llm and STATISTICAL_REVIEW_PROMPT for deterministic tests."""
    monkeypatch.setattr(
        "agents.statistical_review_agent.get_llm",
        lambda temperature=0: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "agents.statistical_review_agent.STATISTICAL_REVIEW_PROMPT",
        _MockPrompt(response),
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def clean_stat_result() -> StatisticalResult:
    """A clean statistical result with no issues."""
    return StatisticalResult(
        is_stationary_adf=True,
        adf_statistic=-5.0,
        adf_p_value=0.001,
        is_stationary_kpss=True,
        kpss_statistic=0.1,
        kpss_p_value=0.5,
        has_trend=False,
        trend_slope=0.0,
        outlier_count=0,
        outlier_ratio=0.0,
        is_white_noise=False,
        white_noise_p_value=0.3,
        recommended_remediation=[],
        seasonal_period=1,
        dominant_period=1.0,
        summary="Series is stationary with no trend or seasonality.",
    )


@pytest.fixture
def seasonal_stat_result() -> StatisticalResult:
    """A statistical result with seasonality and non-stationarity."""
    return StatisticalResult(
        is_stationary_adf=False,
        adf_statistic=-1.5,
        adf_p_value=0.45,
        is_stationary_kpss=False,
        kpss_statistic=0.8,
        kpss_p_value=0.01,
        has_trend=True,
        trend_slope=0.5,
        outlier_count=2,
        outlier_ratio=0.02,
        is_white_noise=False,
        white_noise_p_value=0.001,
        recommended_remediation=["iqr_clip"],
        seasonal_period=12,
        dominant_period=12.0,
        summary="Non-stationary seasonal series with trend.",
    )


@pytest.fixture
def arima_model_selection() -> ModelSelectionResult:
    """Model selection that chose ARIMA."""
    return ModelSelectionResult(
        selected_model="ARIMA",
        explanation="ARIMA selected for its autocorrelation handling.",
        holt_winters_rejected_reason="Not needed.",
        arima_rejected_reason=None,
        sarima_rejected_reason="Overkill.",
        ewma_rejected_reason="Too simple.",
    )


@pytest.fixture
def sarima_model_selection() -> ModelSelectionResult:
    """Model selection that chose SARIMA."""
    return ModelSelectionResult(
        selected_model="SARIMA",
        explanation="SARIMA selected for seasonal data.",
        holt_winters_rejected_reason="Less flexible.",
        arima_rejected_reason="Ignores seasonality.",
        sarima_rejected_reason=None,
        ewma_rejected_reason="Too simple.",
    )


@pytest.fixture
def good_forecast_result() -> ForecastResult:
    """A forecast result with low error."""
    return ForecastResult(
        model_used="SARIMA",
        forecast=[100.0, 101.0, 102.0],
        lower_ci=[95.0, 96.0, 97.0],
        upper_ci=[105.0, 106.0, 107.0],
        forecast_dates=["2024-01-01", "2024-02-01", "2024-03-01"],
        rmse=2.0,
        mae=1.5,
        mape=3.0,
    )


@pytest.fixture
def poor_forecast_result() -> ForecastResult:
    """A forecast result with high MAPE."""
    return ForecastResult(
        model_used="ARIMA",
        forecast=[100.0, 110.0, 130.0],
        lower_ci=[80.0, 85.0, 90.0],
        upper_ci=[120.0, 135.0, 170.0],
        forecast_dates=["2024-01-01", "2024-02-01", "2024-03-01"],
        rmse=15.0,
        mae=12.0,
        mape=25.0,
    )


@pytest.fixture
def all_metrics() -> dict[str, dict[str, float]]:
    """Sample all-metrics dict."""
    return {
        "ARIMA": {"RMSE": 15.0, "MAE": 12.0, "MAPE": 25.0},
        "SARIMA": {"RMSE": 2.0, "MAE": 1.5, "MAPE": 3.0},
        "Holt-Winters": {"RMSE": 5.0, "MAE": 4.0, "MAPE": 8.0},
        "EWMA": {"RMSE": 20.0, "MAE": 18.0, "MAPE": 30.0},
    }


@pytest.fixture
def mock_llm_response_pass() -> SimpleNamespace:
    """A mock LLM response with a PASS verdict."""
    return SimpleNamespace(
        content=(
            "Verdict: PASS\n\n"
            "## Summary\n"
            "The pipeline outputs are consistent and well-supported.\n\n"
            "## Flags\n"
            "None\n\n"
            "## Endorsements\n"
            "- SARIMA correctly handles the detected seasonality.\n"
            "- Forecast error metrics are within acceptable range.\n"
        ),
        usage_metadata={
            "input_tokens": 200,
            "output_tokens": 100,
            "total_tokens": 300,
        },
    )


@pytest.fixture
def mock_llm_response_warn() -> SimpleNamespace:
    """A mock LLM response with a WARN verdict and flags."""
    return SimpleNamespace(
        content=(
            "Verdict: WARN\n\n"
            "## Summary\n"
            "Some concerns about model selection consistency.\n\n"
            "## Flags\n"
            "- [WARNING] [agent: model_selection] ARIMA may not capture "
            "seasonality | Recommendation: Consider SARIMA\n\n"
            "## Endorsements\n"
            "- Statistical tests were thorough.\n"
        ),
        usage_metadata={
            "input_tokens": 200,
            "output_tokens": 100,
            "total_tokens": 300,
        },
    )


# ── Deterministic Pre-Check Tests ─────────────────────────────────────────────


class TestDeterministicPreCheck:
    """Tests for the _deterministic_pre_check function."""

    def test_seasonality_mismatch_flags_arima(
        self,
        seasonal_stat_result: StatisticalResult,
        arima_model_selection: ModelSelectionResult,
        good_forecast_result: ForecastResult,
    ) -> None:
        """Seasonal period > 1 + ARIMA selected should flag critical."""
        flags = _deterministic_pre_check(
            seasonal_stat_result, arima_model_selection, good_forecast_result
        )
        critical_flags = [
            f for f in flags if f["severity"] == "critical"
        ]
        assert any(
            "ARIMA" in f["issue"] and "seasonal" in f["issue"].lower()
            for f in critical_flags
        )

    def test_no_flag_when_sarima_matches_seasonality(
        self,
        seasonal_stat_result: StatisticalResult,
        sarima_model_selection: ModelSelectionResult,
        good_forecast_result: ForecastResult,
    ) -> None:
        """SARIMA with seasonality should not flag seasonality mismatch."""
        flags = _deterministic_pre_check(
            seasonal_stat_result, sarima_model_selection, good_forecast_result
        )
        assert not any("seasonal" in f["issue"].lower() for f in flags)

    def test_high_mape_flags_warning(
        self,
        clean_stat_result: StatisticalResult,
        arima_model_selection: ModelSelectionResult,
        poor_forecast_result: ForecastResult,
    ) -> None:
        """MAPE > 20 should produce a warning flag."""
        flags = _deterministic_pre_check(
            clean_stat_result, arima_model_selection, poor_forecast_result
        )
        assert any(
            f["severity"] == "warning" and "MAPE" in f["issue"]
            for f in flags
        )

    def test_no_flags_for_clean_inputs(
        self,
        clean_stat_result: StatisticalResult,
        sarima_model_selection: ModelSelectionResult,
        good_forecast_result: ForecastResult,
    ) -> None:
        """Clean inputs with no issues should produce no flags."""
        flags = _deterministic_pre_check(
            clean_stat_result, sarima_model_selection, good_forecast_result
        )
        assert flags == []


# ── Parsing Tests ─────────────────────────────────────────────────────────────


class TestParsing:
    """Tests for LLM response parsing functions."""

    def test_parse_verdict_pass(self) -> None:
        """Test parsing a PASS verdict."""
        assert _parse_verdict("Verdict: PASS\n...") == "pass"

    def test_parse_verdict_warn(self) -> None:
        """Test parsing a WARN verdict."""
        assert _parse_verdict("Verdict: WARN\n...") == "warn"

    def test_parse_verdict_fail(self) -> None:
        """Test parsing a FAIL verdict."""
        assert _parse_verdict("Verdict: FAIL\n...") == "fail"

    def test_parse_verdict_default_warn(self) -> None:
        """Test that missing verdict defaults to warn."""
        assert _parse_verdict("No verdict here.") == "warn"

    def test_parse_flags_extracts_structured_flags(self) -> None:
        """Test that structured flags are parsed correctly."""
        text = (
            "## Flags\n"
            "- [CRITICAL] [agent: model_selection] Wrong model | "
            "Recommendation: Use SARIMA\n"
            "- [WARNING] [agent: forecasting] High error | "
            "Recommendation: Tune parameters\n"
        )
        flags = _parse_flags(text)
        assert len(flags) == 2
        assert flags[0]["severity"] == "critical"
        assert flags[0]["agent"] == "model_selection"
        assert "Wrong model" in flags[0]["issue"]
        assert flags[1]["severity"] == "warning"

    def test_parse_flags_none_returns_empty(self) -> None:
        """Test that 'None' flags line returns empty list."""
        text = "## Flags\nNone\n"
        flags = _parse_flags(text)
        assert flags == []

    def test_parse_endorsements_extracts_items(self) -> None:
        """Test that endorsements are parsed correctly."""
        text = (
            "## Endorsements\n"
            "- SARIMA correctly handles seasonality.\n"
            "- Tests were thorough.\n"
        )
        endorsements = _parse_endorsements(text)
        assert len(endorsements) == 2
        assert "SARIMA correctly handles seasonality." in endorsements

    def test_parse_endorsements_none_returns_empty(self) -> None:
        """Test that 'None' endorsements returns empty list."""
        text = "## Endorsements\n- None\n"
        endorsements = _parse_endorsements(text)
        assert endorsements == []

    def test_parse_summary_extracts_text(self) -> None:
        """Test that summary is parsed correctly."""
        text = (
            "## Summary\n"
            "The pipeline is consistent.\n\n"
            "## Flags\n"
        )
        summary = _parse_summary(text)
        assert "consistent" in summary

    def test_parse_summary_default_when_missing(self) -> None:
        """Test default summary when section is missing."""
        summary = _parse_summary("No summary section here.")
        assert "completed" in summary.lower()


# ── Verdict Computation Tests ─────────────────────────────────────────────────


class TestComputeVerdict:
    """Tests for _compute_verdict function."""

    def test_critical_flags_force_warn_from_pass(self) -> None:
        """Critical deterministic flags should force pass → warn."""
        flags = [{"severity": "critical", "issue": "test"}]
        assert _compute_verdict("pass", flags) == "warn"

    def test_critical_flags_preserve_fail(self) -> None:
        """Critical flags should preserve an existing fail verdict."""
        flags = [{"severity": "critical", "issue": "test"}]
        assert _compute_verdict("fail", flags) == "fail"

    def test_no_critical_flags_preserve_pass(self) -> None:
        """No critical flags should preserve pass."""
        flags = [{"severity": "warning", "issue": "test"}]
        assert _compute_verdict("pass", flags) == "pass"


# ── Agent Integration Tests ───────────────────────────────────────────────────


class TestRunStatisticalReviewAgent:
    """Tests for the full run_statistical_review_agent function."""

    def test_returns_all_required_keys(
        self,
        clean_stat_result: StatisticalResult,
        sarima_model_selection: ModelSelectionResult,
        good_forecast_result: ForecastResult,
        all_metrics: dict[str, dict[str, float]],
        mock_llm_response_pass: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Result should contain all required keys with correct types."""
        _patch_llm(monkeypatch, mock_llm_response_pass)

        result = run_statistical_review_agent(
            clean_stat_result,
            sarima_model_selection,
            good_forecast_result,
            all_metrics,
        )

        assert hasattr(result, "verdict")
        assert hasattr(result, "flags")
        assert hasattr(result, "endorsements")
        assert hasattr(result, "summary")
        assert hasattr(result, "reasoning_steps")
        assert hasattr(result, "token_usage")
        assert isinstance(result.verdict, str)
        assert isinstance(result.flags, list)
        assert isinstance(result.endorsements, list)
        assert isinstance(result.summary, str)
        assert isinstance(result.reasoning_steps, list)
        assert isinstance(result.token_usage, dict)

    def test_llm_failure_falls_back_to_precheck(
        self,
        seasonal_stat_result: StatisticalResult,
        arima_model_selection: ModelSelectionResult,
        good_forecast_result: ForecastResult,
        all_metrics: dict[str, dict[str, float]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """LLM failure should fall back to pre-check with verdict warn."""
        def raising_llm(temperature: float = 0):
            raise RuntimeError("LLM unavailable")

        monkeypatch.setattr(
            "agents.statistical_review_agent.get_llm", raising_llm
        )

        result = run_statistical_review_agent(
            seasonal_stat_result,
            arima_model_selection,
            good_forecast_result,
            all_metrics,
        )

        # Should have flags from deterministic pre-check (seasonality mismatch)
        assert len(result.flags) > 0
        assert result.verdict == "warn"
        assert any(
            f["severity"] == "critical" for f in result.flags
        )
        # Summary should mention pre-check fallback
        assert "pre-check" in result.summary.lower()

    def test_pass_verdict_when_no_issues(
        self,
        clean_stat_result: StatisticalResult,
        sarima_model_selection: ModelSelectionResult,
        good_forecast_result: ForecastResult,
        all_metrics: dict[str, dict[str, float]],
        mock_llm_response_pass: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Clean inputs with LLM pass verdict should return pass."""
        _patch_llm(monkeypatch, mock_llm_response_pass)

        result = run_statistical_review_agent(
            clean_stat_result,
            sarima_model_selection,
            good_forecast_result,
            all_metrics,
        )

        assert result.verdict == "pass"
        assert result.flags == []
        assert len(result.endorsements) == 2

    def test_deterministic_flags_merged_with_llm_flags(
        self,
        seasonal_stat_result: StatisticalResult,
        arima_model_selection: ModelSelectionResult,
        good_forecast_result: ForecastResult,
        all_metrics: dict[str, dict[str, float]],
        mock_llm_response_warn: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Deterministic flags should be merged with LLM flags."""
        _patch_llm(monkeypatch, mock_llm_response_warn)

        result = run_statistical_review_agent(
            seasonal_stat_result,
            arima_model_selection,
            good_forecast_result,
            all_metrics,
        )

        # Should have deterministic critical flag (seasonality + ARIMA)
        # plus LLM warning flag
        assert len(result.flags) >= 2
        assert any(f["severity"] == "critical" for f in result.flags)
        assert any(f["severity"] == "warning" for f in result.flags)