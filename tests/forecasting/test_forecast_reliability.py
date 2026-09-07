"""Time availability, interval audits, and repeated-forecast safeguards."""

import numpy as np
import pandas as pd
import pytest

from forecasting.known_context import options_as_of, prepare_exog_options
from forecasting.calibration import (
    fit_interval_calibration,
    apply_interval_calibration,
    audit_interval_calibration,
)
from forecasting.contracts import (
    BacktestEvaluation,
    BacktestFold,
    BacktestFoldResult,
    ForecastFitStatus,
)


def test_predictors_use_historical_versions_and_reject_future_only_records():
    series = pd.Series([1.0, 2.0], index=pd.date_range("2025-01-01", periods=2))
    values = {
        d: [
            {"value": 3, "available_at": "2024-12-01"},
            {"value": 99, "available_at": "2025-01-03"},
        ]
        for d in ["2025-01-01", "2025-01-02", "2025-01-03"]
    }
    options = prepare_exog_options(
        {"known_covariates": {"price": values}}, series, 1, "D"
    )
    resolved = options_as_of(options, series, 1, "D")
    assert set(resolved["known_covariates"]["price"].values()) == {3}
    values["2025-01-03"] = [{"value": 99, "available_at": "2025-01-03"}]
    with pytest.raises(ValueError, match="unavailable"):
        options_as_of(options, series, 1, "D")


def test_scalar_predictors_need_explicit_declaration():
    series = pd.Series([1.0, 2.0], index=pd.date_range("2025-01-01", periods=2))
    options = {
        "known_covariates": {
            "price": {"2025-01-01": 1, "2025-01-02": 2, "2025-01-03": 3}
        }
    }
    with pytest.raises(ValueError, match="declaration"):
        options_as_of(prepare_exog_options(options, series, 1, "D"), series, 1, "D")
    options["covariates_known_in_advance"] = True
    assert options_as_of(options, series, 1, "D")["predictor_availability"][
        "assumed_known_ahead"
    ] == ["price"]


def test_retrospective_events_do_not_enter_earlier_forecasts():
    series = pd.Series([1.0, 2.0], index=pd.date_range("2025-01-01", periods=2))
    options = {
        "known_events": [
            {"date": "2025-01-03", "type": "promotion", "label": "Later"},
            {
                "date": "2025-01-03",
                "type": "promotion",
                "label": "Planned",
                "available_at": "2024-12-01",
            },
        ]
    }
    out = options_as_of(prepare_exog_options(options, series, 1, "D"), series, 1, "D")
    assert out["prophet_holidays"]["holiday"].tolist() == ["Planned"]


def _fold(index, start, value=0.0):
    return BacktestFoldResult(
        fold=BacktestFold(
            fold_index=index,
            train_end_index=start,
            test_start_index=start,
            test_end_index=start + 1,
            horizon=1,
        ),
        status=ForecastFitStatus.OK,
        predictions=[value],
        lower_ci=[-1.0],
        upper_ci=[1.0],
    )


def test_calibration_uses_only_selection_errors_and_audits_later_actuals():
    series = pd.Series([0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0])
    evaluation = BacktestEvaluation(
        model_name="Naive",
        folds=[_fold(i, i + 1) for i in range(5)],
        validation_design={"requested_horizon": 2},
    )
    calibration = fit_interval_calibration(evaluation, series)
    assert calibration["by_horizon"]["1"]["expansion"] == 6.0
    assert calibration["by_horizon"]["2"]["status"] == "insufficient_evidence"
    assert apply_interval_calibration([-1.0, -1.0], [1.0, 1.0], calibration) == (
        [-7.0, -1.0],
        [7.0, 1.0],
        [1],
    )
    audit = audit_interval_calibration(_fold(6, 6), series, calibration)
    assert audit["by_horizon"]["1"]["adjusted_coverage"] == 0
    series.iloc[-1] = 1000
    assert fit_interval_calibration(evaluation, series) == calibration


def test_calibration_does_not_count_missing_actuals():
    series = pd.Series([0.0, 3.0, 4.0, np.nan, 6.0, 7.0])
    evaluation = BacktestEvaluation(
        model_name="Naive",
        folds=[_fold(i, i + 1) for i in range(5)],
        validation_design={"requested_horizon": 1},
    )
    calibration = fit_interval_calibration(evaluation, series)
    assert calibration["by_horizon"]["1"]["n_origins"] == 4
    assert apply_interval_calibration([-1.0], [1.0], calibration) == ([-1.0], [1.0], [])


def test_subdaily_predictor_versions_keep_distinct_timestamps():
    series = pd.Series(
        [1.0, 2.0], index=pd.date_range("2025-01-01", periods=2, freq="h")
    )
    values = {
        date.isoformat(): {"value": i, "available_at": "2024-12-01"}
        for i, date in enumerate(pd.date_range("2025-01-01", periods=3, freq="h"))
    }
    out = options_as_of({"known_covariates": {"load": values}}, series, 1, "h")
    assert list(out["known_covariates"]["load"].values()) == [0.0, 1.0, 2.0]


def test_calibration_skips_overlapping_windows():
    series = pd.Series([5.0] * 12)
    folds = [
        BacktestFoldResult(
            fold=BacktestFold(
                fold_index=i,
                train_end_index=i + 1,
                test_start_index=i + 1,
                test_end_index=i + 4,
                horizon=3,
            ),
            status=ForecastFitStatus.OK,
            predictions=[0.0] * 3,
            lower_ci=[-1.0] * 3,
            upper_ci=[1.0] * 3,
        )
        for i in range(8)
    ]
    evaluation = BacktestEvaluation(
        model_name="Naive", folds=folds, validation_design={"requested_horizon": 3}
    )
    calibration = fit_interval_calibration(evaluation, series)
    assert calibration["origin_indices"] == [0, 3, 6]
    assert calibration["by_horizon"]["1"]["status"] == "insufficient_evidence"
