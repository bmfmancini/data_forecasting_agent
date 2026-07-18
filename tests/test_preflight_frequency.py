"""Regression tests for intraday frequency handling and quality warnings."""

from __future__ import annotations

import pandas as pd

from utils.data_cleaning import frequency_to_seasonal_period, validate_schema
from utils.preflight import (
    _normalize_frequency,
    prepare_series_frame,
    run_preflight_checks,
)


def _minute_frame(periods: int = 1441) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-07-17 11:17", periods=periods, freq="min"),
            "Load": range(periods),
        }
    )


def test_minute_alias_is_not_normalized_as_monthly() -> None:
    assert _normalize_frequency("min") == "min"
    assert _normalize_frequency("MIN") == "min"
    assert _normalize_frequency("T") == "min"
    assert _normalize_frequency("15min") == "15min"


def test_minute_series_survives_preparation_without_monthly_reindexing() -> None:
    frame = _minute_frame()

    prepared, frequency = prepare_series_frame(frame, "Date", "Load")

    assert frequency == "min"
    assert len(prepared) == len(frame)
    assert prepared["Date"].iloc[0] == frame["Date"].iloc[0]
    assert prepared["Date"].iloc[-1] == frame["Date"].iloc[-1]


def test_one_day_of_minute_data_warns_but_can_continue() -> None:
    result = run_preflight_checks(_minute_frame(), "Date", "Load", 12)

    assert result.status != "error"
    assert result.detected_frequency == "min"
    assert any(
        "spans only 1 day" in warning
        and "you can proceed with caution" in warning
        for warning in result.warnings
    )
    decision = next(
        item for item in result.decisions if item.key == "continue_limited_history"
    )
    assert decision.default == "continue"
    assert decision.options == ["continue", "stop"]


def test_equivalent_minute_aliases_pass_schema_frequency_validation() -> None:
    frame = _minute_frame()
    series = frame.set_index("Date")["Load"]

    report = validate_schema(series, {"expected_freq": "min"})

    assert report["freq_regular"] is True


def test_intraday_frequency_uses_daily_seasonal_period() -> None:
    assert frequency_to_seasonal_period("min") == 1440
    assert frequency_to_seasonal_period("15min") == 96
    assert frequency_to_seasonal_period("h") == 24
    assert frequency_to_seasonal_period("2h") == 12
