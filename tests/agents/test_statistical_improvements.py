"""Statistical regression tests for the complete forecasting procedure."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from agents import forecasting_agent
from forecasting import registry
from forecasting.backtesting import (
    BacktestConfig,
    FoldPrediction,
    evaluate_candidates,
    generate_folds,
)
from forecasting.engine import ForecastEngine, backtest_config
from forecasting.intervals import smoothing_paths
from forecasting.metrics import calculate_forecast_metrics
from forecasting.residual_diagnostics import analyze_backtest_errors
from schemas import ModelSelectionResult, StatisticalResult


def series(n=100, freq="D"):
    return pd.Series(
        30 + np.random.default_rng(4).normal(size=n),
        index=pd.date_range("2020-01-01", periods=n, freq=freq),
    )


def naive(train, fold):
    return FoldPrediction(np.repeat(train.iloc[-1], fold.horizon))


def test_capped_origins_span_history_and_include_latest():
    folds = generate_folds(
        series(1000), BacktestConfig(horizon=1, max_origins=5, final_test_size=1)
    )
    assert len(folds) == 5
    assert folds[0].train_end_index == 499
    assert folds[-1].test_end_index == 999
    assert folds[2].train_end_index > 700


def test_one_origin_cap_keeps_latest():
    folds = generate_folds(series(), BacktestConfig(horizon=3, max_origins=1))
    assert folds[0].test_end_index == 100


def test_partial_failure_cannot_win():
    def fragile(train, fold):
        return naive(train, fold) if fold.fold_index == 0 else None

    results = evaluate_candidates(
        series(),
        {"naive": naive, "fragile": fragile},
        BacktestConfig(horizon=3, max_origins=5),
    )
    assert results["naive"].is_rankable
    assert results["fragile"].n_failed_origins == 4
    assert not results["fragile"].is_rankable


def test_nonfinite_predictions_fail_instead_of_dropping_difficult_targets():
    results = evaluate_candidates(
        series(),
        {"bad": lambda train, fold: FoldPrediction(np.array([np.nan, 30, 30]))},
        BacktestConfig(horizon=3),
    )
    assert not results["bad"].is_rankable
    assert results["bad"].n_evaluated == 0


def test_final_test_not_fitted_during_candidate_selection():
    seen = []

    def capture(train, fold):
        seen.append(fold.test_end_index)
        return naive(train, fold)

    results = evaluate_candidates(
        series(), {"naive": capture}, BacktestConfig(horizon=5, final_test_size=10)
    )
    assert max(seen) <= 90
    assert results["naive"].final_test_metrics.rmse is None


def test_missing_actuals_use_the_same_scoring_mask():
    data = series()
    data.iloc[70:72] = np.nan
    results = evaluate_candidates(
        data, {"a": naive, "b": naive}, BacktestConfig(horizon=5)
    )
    assert results["a"].n_evaluated == results["b"].n_evaluated
    assert results["a"].pooled_metrics.n_missing == 2


def test_mase_does_not_compress_missing_calendar_pairs():
    metrics = calculate_forecast_metrics([10], [11], training=[1, np.nan, 100, 101])
    assert metrics.mase == 1.0  # only the observed adjacent pair 100 -> 101


def test_seasonal_baseline_uses_fixed_calendar_lag(monkeypatch):
    from forecasting import engine as engine_module
    from forecasting.contracts import SeasonalityEvidence

    monkeypatch.setattr(registry, "get_enabled_models", lambda: ("EWMA",))
    monkeypatch.setattr(
        engine_module,
        "detect_seasonality",
        lambda *args, **kwargs: SeasonalityEvidence(selected_period=30),
    )
    data = series(60, freq="MS")
    result = ForecastEngine("MS").candidates["Seasonal Naive"].fit(data, 3)
    assert result.forecast == pytest.approx(data.iloc[-12:-9].tolist())


def test_ses_simulation_matches_multistep_state_variance():
    fit = SimpleExpSmoothing(series(400), initialization_method="estimated").fit(
        smoothing_level=0.5, optimized=False
    )
    paths = smoothing_paths(fit, 12, repetitions=20000)
    ratio = np.var(paths[:, -1]) / np.var(paths[:, 0])
    assert ratio == pytest.approx(1 + 11 * 0.5**2, rel=0.08)


def test_pooled_multistep_errors_are_not_tested_as_white_noise():
    result = analyze_backtest_errors([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    assert result.ljung_box_p_value is None
    assert result.shapiro_p_value is None
    assert result.mean_ci_lower is None
    assert set(result.variance_by_horizon) == {1, 2, 3}


def test_missing_actuals_do_not_count_as_interval_misses():
    result = analyze_backtest_errors(
        [[0, np.nan]],
        fold_actuals=[[10, np.nan]],
        fold_lower=[[9, 9]],
        fold_upper=[[11, 11]],
    )
    assert result.interval_coverage == 1.0
    assert result.winkler_score == 2.0


def test_requested_horizon_preserved_when_history_supports_it():
    config = backtest_config(120, 24, "MS", {})
    assert config.horizon == 24
    folds = generate_folds(series(120, "MS"), config)
    assert len(folds) >= 2


def test_candidate_catalog_does_not_depend_on_future_targets(monkeypatch):
    monkeypatch.setattr(registry, "get_enabled_models", lambda: ("EWMA",))
    engine = ForecastEngine("D")
    data = series()
    first = engine.candidates["EWMA + Auto transform"].fit(data.iloc[:50], 4)
    data.iloc[50:] = -1e9
    second = engine.candidates["EWMA + Auto transform"].fit(data.iloc[:50], 4)
    assert first.forecast == second.forecast
    assert first.fitted_configuration == second.fitted_configuration


def test_recent_window_imputation_does_not_see_older_values(monkeypatch):
    monkeypatch.setattr(registry, "get_enabled_models", lambda: ("EWMA",))
    engine = ForecastEngine("D", {"recent_window": 24})
    data = series()
    first = engine.candidates["EWMA + Recent window"].fit(data, 3)
    data.iloc[:-24] = 1e9
    second = engine.candidates["EWMA + Recent window"].fit(data, 3)
    assert first.forecast == second.forecast


def test_disabled_models_have_no_variants_or_ensemble_members(monkeypatch):
    monkeypatch.setattr(registry, "get_enabled_models", lambda: ("EWMA",))
    engine = ForecastEngine("D")
    assert all(
        "ARIMA" not in name and "Prophet" not in name for name in engine.candidates
    )
    assert "Simple Ensemble" not in engine.candidates


def _stats():
    return StatisticalResult(
        is_stationary_adf=True,
        adf_statistic=0,
        adf_p_value=0.01,
        is_stationary_kpss=True,
        kpss_statistic=0,
        kpss_p_value=0.9,
        has_trend=False,
        trend_slope=0,
        seasonal_period=999,
        summary="Full-history statistics must not configure validation.",
    )


def test_forecast_pipeline_works_without_llm_and_reports_honest_intervals(monkeypatch):
    monkeypatch.setattr(registry, "get_enabled_models", lambda: ("EWMA",))
    monkeypatch.setattr(
        forecasting_agent,
        "get_llm",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    forecast, metrics = forecasting_agent.run_forecasting_agent(
        series(),
        ModelSelectionResult(selected_model="EWMA", explanation="test"),
        _stats(),
        4,
        "D",
        loss_preference="mae",
    )
    assert forecast.model_used in metrics
    assert len(forecast.lower_ci) == 4
    assert "calibrated" not in forecast.interval_label
    assert forecast.final_test_metrics["rmse"] is not None
    assert forecast.validation_design["final_test_used_for_selection"] is False
    assert forecast.validation_design["mase_period"] != 999
    assert all(
        candidate.final_test_metrics["rmse"] is None
        for candidate in forecast.candidate_results
        if candidate.model != forecast.model_used
    )


def test_unestimable_scaled_loss_reports_its_actual_fallback(monkeypatch):
    monkeypatch.setattr(registry, "get_enabled_models", lambda: ("EWMA",))
    monkeypatch.setattr(
        forecasting_agent,
        "get_llm",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    data = series()
    data[:] = 10.0
    forecast, _ = forecasting_agent.run_forecasting_agent(
        data,
        ModelSelectionResult(selected_model="EWMA", explanation="test"),
        _stats(),
        4,
        "D",
        loss_preference="mase",
    )
    decision = forecast.validation_design["decision_loss"]
    assert decision["requested"] == "mase"
    assert decision["resolved"] == "mae"
    assert decision["resolution_source"] == "unavailable_metric_fallback"
    assert forecast.selection_metrics["mae"] == 0.0


def test_gap_forecast_skips_unobserved_lead_periods(monkeypatch):
    monkeypatch.setattr(registry, "get_enabled_models", lambda: ("EWMA",))
    engine = ForecastEngine("D")
    from forecasting.contracts import BacktestFold

    data = pd.Series(
        np.arange(20, dtype=float), index=pd.date_range("2020", periods=20)
    )
    result = engine.candidates["Drift"](
        data,
        BacktestFold(
            fold_index=0,
            train_end_index=20,
            test_start_index=23,
            test_end_index=25,
            horizon=2,
        ),
    )
    assert result.predictions.tolist() == [23.0, 24.0]


def test_pinball_loss_preserves_asymmetric_costs():
    under = calculate_forecast_metrics([10], [8], quantile=0.9)
    over = calculate_forecast_metrics([10], [12], quantile=0.9)
    assert under.pinball == pytest.approx(1.8)
    assert over.pinball == pytest.approx(0.2)


def test_quantile_forecast_uses_the_predictive_distribution(monkeypatch):
    monkeypatch.setattr(registry, "get_enabled_models", lambda: ("EWMA",))
    fit = (
        ForecastEngine("D", {"point_quantile": 0.9}).candidates["EWMA"].fit(series(), 4)
    )
    assert fit.forecast == np.quantile(fit.prediction_samples, 0.9, axis=0).tolist()
    assert fit.fitted_configuration["point_quantile"] == 0.9


def test_tsb_decays_after_observed_zero_demand():
    from forecasting.intermittent import tsb_states

    values = np.array([0, 10, 0, 10, 0, 10], dtype=float)
    p, size, _ = tsb_states(values, 0.1, 0.2)
    later_p, later_size, _ = tsb_states(np.r_[values, np.zeros(20)], 0.1, 0.2)
    assert later_size == size
    assert later_p == pytest.approx(p * 0.8**20)


def test_tsb_is_opt_in_and_rejects_negative_targets(monkeypatch):
    from forecasting.intermittent import fit_intermittent_window

    monkeypatch.setattr(
        registry, "get_enabled_models", lambda: ("EWMA", "Intermittent Demand")
    )
    assert "Intermittent Demand" not in ForecastEngine("D").candidates
    assert (
        "Intermittent Demand"
        in ForecastEngine("D", {"demand_pattern": "intermittent"}).candidates
    )
    with pytest.raises(ValueError, match="nonnegative"):
        fit_intermittent_window(pd.Series([-1, 0, 0, 2, 0, 0, 3, 0, 0, 0]), 3)


def test_hourly_dynamic_regression_has_daily_and_weekly_features():
    from forecasting.dynamic_regression import design_matrix

    train, future, names = design_matrix(series(400, "h"), 24, "h", 2, {})
    assert train.shape == (400, 9)
    assert future.shape == (24, 9)
    assert "sin_24_1" in names and "sin_168_1" in names


def test_covariates_require_known_ahead_declaration_and_complete_values():
    from forecasting.dynamic_regression import design_matrix

    data = series(20)
    covariates = {
        "promotion": {
            date.isoformat(): float(i % 2) for i, date in enumerate(data.index)
        }
    }
    with pytest.raises(ValueError, match="declared known"):
        design_matrix(data, 3, "D", 1, {"known_covariates": covariates})
    with pytest.raises(ValueError, match="every training and forecast timestamp"):
        design_matrix(
            data,
            3,
            "D",
            1,
            {"known_covariates": covariates, "covariates_known_in_advance": True},
        )


def test_changing_future_covariates_does_not_change_training_features():
    from forecasting.dynamic_regression import design_matrix

    data = series(30)
    dates = pd.date_range(data.index[0], periods=34)
    values = {date.isoformat(): float(i) for i, date in enumerate(dates)}
    options = {
        "known_covariates": {"price": values},
        "covariates_known_in_advance": True,
    }
    first, _, _ = design_matrix(data, 4, "D", 1, options)
    for date in dates[-4:]:
        values[date.isoformat()] = 1e9
    second, _, _ = design_matrix(data, 4, "D", 1, options)
    np.testing.assert_array_equal(first, second)


def test_frequency_change_requires_explicit_aggregation():
    from utils.preflight import prepare_series_frame

    frame = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=60), "value": 1.0}
    )
    with pytest.raises(ValueError, match="aggregation"):
        prepare_series_frame(frame, "date", "value", {"frequency": "MS"})
    monthly, _ = prepare_series_frame(
        frame, "date", "value", {"frequency": "MS", "aggregation": "sum"}
    )
    assert monthly["value"].tolist() == [31.0, 29.0]


def test_descriptive_transformation_does_not_change_report_history():
    from services.pipeline_service import _apply_agent_remediation

    data = series()
    stats = _stats().model_copy(
        update={"recommended_remediation": ["box_cox", "iqr_clip"]}
    )
    output = _apply_agent_remediation(
        data, stats, [], {"outlier_strategy": "Let AI Decide"}
    )
    pd.testing.assert_series_equal(output, data)


def test_review_uses_chosen_loss_not_rmse():
    from agents.statistical_review_agent import _check_deterministic_policy_violation

    selection = ModelSelectionResult(
        selected_model="EWMA",
        explanation="Selected EWMA",
        selection_method="deterministic",
        selection_evidence={"decision_loss": {"resolved": "mae"}},
    )
    assert (
        _check_deterministic_policy_violation(
            selection,
            {
                "EWMA": {"RMSE": 10, "MAE": 1},
                "ARIMA": {"RMSE": 2, "MAE": 1.5},
            },
        )
        is None
    )


def test_forced_prophet_reaches_prophet_fitter(monkeypatch):
    from forecasting.contracts import ForecastAdapterResult, ForecastFitStatus

    calls = []

    def fit(train, horizon, **kwargs):
        calls.append((len(train), horizon))
        point = [float(train.mean())] * horizon
        return ForecastAdapterResult(
            status=ForecastFitStatus.OK,
            forecast=point,
            lower_ci=[v - 2 for v in point],
            upper_ci=[v + 2 for v in point],
            prediction_samples=[point, point],
        )

    monkeypatch.setattr(registry, "get_enabled_models", lambda: ("Prophet",))
    monkeypatch.setitem(registry.MODELS["Prophet"], "window_fn", fit)
    monkeypatch.setattr(forecasting_agent, "get_llm", lambda **kwargs: None)
    result, _ = forecasting_agent.run_forecasting_agent(
        series(),
        ModelSelectionResult(
            selected_model="Prophet", explanation="forced", selection_method="forced"
        ),
        _stats(),
        3,
        "D",
        loss_preference="mae",
    )
    assert result.model_used == "Prophet"
    assert (100, 3) in calls
    assert any(length < 100 for length, _ in calls)


def test_production_failure_reranks_using_requested_loss(monkeypatch):
    from types import SimpleNamespace
    from forecasting.contracts import (
        BacktestEvaluation,
        ForecastAdapterResult,
        ForecastFitStatus,
        ForecastMetrics,
    )

    class Procedure:
        def __init__(self, name):
            self.base_name, self.members = name, ()

        def fit(self, train, horizon):
            if self.base_name == "A":
                raise ValueError("production failure")
            return ForecastAdapterResult(
                status=ForecastFitStatus.OK,
                forecast=[30.0] * horizon,
                lower_ci=[28.0] * horizon,
                upper_ci=[32.0] * horizon,
            )

    candidates = {
        name: Procedure(name) for name in ("A", "B", "C", "Naive", "Seasonal Naive")
    }
    monkeypatch.setattr(
        forecasting_agent,
        "ForecastEngine",
        lambda *args: SimpleNamespace(candidates=candidates, preprocessing={}),
    )
    evaluations = {
        name: BacktestEvaluation(
            model_name=name,
            n_origins=2,
            pooled_metrics=ForecastMetrics(mae=mae, rmse=rmse, mase=mae),
        )
        for name, mae, rmse in [("A", 1, 10), ("B", 2, 20), ("C", 3, 3)]
    }
    monkeypatch.setattr(
        forecasting_agent, "evaluate_candidates", lambda *args: evaluations
    )
    monkeypatch.setattr(
        forecasting_agent,
        "evaluate_final_candidate",
        lambda *args: (ForecastMetrics(), None),
    )
    monkeypatch.setattr(forecasting_agent, "get_llm", lambda **kwargs: None)
    result, _ = forecasting_agent.run_forecasting_agent(
        series(),
        ModelSelectionResult(selected_model="A", explanation="test"),
        _stats(),
        3,
        "D",
        loss_preference="mae",
    )
    assert result.model_used == "B"
    assert result.is_fallback
    assert result.validation_design["production_failures"] == ["A"]
