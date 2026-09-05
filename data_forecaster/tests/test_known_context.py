"""Unit tests for :mod:`forecasting.known_context` (structured context → models).

Covers the pure conversion layer that translates user-facing preflight context
(holiday country, custom events, covariates) into the shapes Prophet and
Dynamic Regression ingest, plus the report-facing summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.known_context import (
    country_list,
    country_name,
    expand_holidays,
    merge_events,
    prepare_exog_options,
    summarize_context,
    to_event_dummies,
    to_known_covariates,
    to_prophet_holidays,
)


# ── Country helpers ────────────────────────────────────────────────────────────


class TestCountryHelpers:
    def test_country_list_is_sorted_pairs(self) -> None:
        pairs = country_list()
        assert pairs
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
        names = [name for _, name in pairs]
        assert names == sorted(names)
        # US is present and mapped to a display name.
        codes = dict(pairs)
        assert codes["US"] == "United States"

    def test_country_name_lookup(self) -> None:
        assert country_name("US") == "United States"
        assert country_name("us") == "United States"  # case-insensitive
        assert country_name(None) is None
        assert country_name("ZZ") is None  # unknown code


# ── Holiday expansion ─────────────────────────────────────────────────────────


class TestExpandHolidays:
    def test_us_holidays_have_dates_and_labels(self) -> None:
        events = expand_holidays("US", [2024])
        assert events
        for event in events:
            assert event["type"] == "holiday"
            assert event["date"][:4] == "2024"
            assert event["label"]
        # Independence Day is always present.
        dates = {event["date"] for event in events}
        assert "2024-07-04" in dates

    def test_empty_or_unknown_country_yields_no_events(self) -> None:
        assert expand_holidays("", [2024]) == []
        assert expand_holidays(None, [2024]) == []
        assert expand_holidays("ZZ", [2024]) == []

    def test_events_sorted_by_date(self) -> None:
        events = expand_holidays("GB", [2024])
        dates = [event["date"] for event in events]
        assert dates == sorted(dates)


# ── Event merging ──────────────────────────────────────────────────────────────


class TestMergeEvents:
    def test_holidays_plus_custom_events(self) -> None:
        merged = merge_events(
            {
                "holidays_country": "US",
                "known_events": [
                    {"type": "spike", "date": "2024-07-04", "label": "sale"},
                    {"type": "lull", "date": "2024-02-15", "label": "slow"},
                ],
            },
            [2024],
        )
        types = [event["type"] for event in merged]
        assert types.count("holiday") > 0
        assert "spike" in types
        assert "lull" in types

    def test_invalid_custom_dates_are_skipped(self) -> None:
        merged = merge_events(
            {
                "known_events": [
                    {"type": "spike", "date": "2024-13-99", "label": "bad"},
                    {"type": "lull", "date": "2024-02-15", "label": "good"},
                ],
            },
            [2024],
        )
        dates = [event["date"] for event in merged]
        assert "2024-02-15" in dates
        assert all("13-99" not in d for d in dates)


# ── Prophet holiday frame ──────────────────────────────────────────────────────


class TestToProphetHolidays:
    def test_frame_columns_and_dedup(self) -> None:
        events = [
            {"type": "holiday", "date": "2024-07-04", "label": "Independence Day"},
            {"type": "spike", "date": "2024-07-04", "label": "sale"},
            {"type": "holiday", "date": "2024-07-04", "label": "Independence Day"},
        ]
        frame = to_prophet_holidays(events)
        assert list(frame.columns) == ["ds", "holiday"]
        # The duplicate (same ds + label) is dropped; the spike stays distinct.
        assert len(frame) == 2

    def test_empty_events_yields_empty_frame(self) -> None:
        frame = to_prophet_holidays([])
        assert list(frame.columns) == ["ds", "holiday"]
        assert frame.empty


# ── Event dummies ──────────────────────────────────────────────────────────────


class TestToEventDummies:
    def test_indicator_columns_per_type(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        events = [
            {"type": "spike", "date": "2024-01-02", "label": "a"},
            {"type": "lull", "date": "2024-01-04", "label": "b"},
        ]
        dummies = to_event_dummies(events, idx)
        assert set(dummies.columns) == {"event_spike", "event_lull"}
        # Spike fires on Jan 2 only.
        assert dummies["event_spike"].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
        assert dummies["event_lull"].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]

    def test_no_events_yields_empty_frame(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        dummies = to_event_dummies([], idx)
        assert dummies.empty
        assert list(dummies.columns) == []


# ── Known covariates (Dynamic Regression shape) ───────────────────────────────


class TestToKnownCovariates:
    def test_dummies_plus_user_covariate_reindexed(self) -> None:
        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        events = [{"type": "spike", "date": "2024-01-02", "label": "a"}]
        user = {"price": {"2024-01-01": 9.0, "2024-01-03": 11.0}}
        out = to_known_covariates(events, user, idx)
        assert "event_spike" in out
        assert "price" in out
        # Event dummy is 0.0-filled.
        assert out["event_spike"]["2024-01-02"] == 1.0
        assert out["event_spike"]["2024-01-01"] == 0.0
        # User covariate keeps NaN where not supplied.
        assert out["price"]["2024-01-01"] == 9.0
        assert np.isnan(out["price"]["2024-01-02"])

    def test_non_dict_covariate_skipped(self) -> None:
        idx = pd.date_range("2024-01-01", periods=2, freq="D")
        out = to_known_covariates([], {"bad": [1, 2, 3]}, idx)
        assert out == {}


# ── Report summary ─────────────────────────────────────────────────────────────


class TestSummarizeContext:
    def test_counts_by_type_and_country(self) -> None:
        summary = summarize_context(
            {
                "holidays_country": "US",
                "known_events": [
                    {"type": "spike", "date": "2024-07-04", "label": "a"},
                    {"type": "spike", "date": "2024-08-04", "label": "b"},
                    {"type": "lull", "date": "2024-02-15", "label": "c"},
                    {"type": "spike", "date": "bad", "label": "skip"},
                ],
                "known_covariates": {"price": {}},
            }
        )
        assert summary["holidays_country"] == "US"
        assert summary["events_by_type"] == {"spike": 2, "lull": 1}
        assert summary["event_count"] == 3  # invalid date excluded
        assert summary["covariates"] == ["price"]

    def test_empty_context(self) -> None:
        summary = summarize_context(None)
        assert summary == {
            "holidays_country": None,
            "events_by_type": {},
            "event_count": 0,
            "covariates": [],
        }


# ── prepare_exog_options (engine wiring) ──────────────────────────────────────


class TestPrepareExogOptions:
    @staticmethod
    def _series() -> pd.Series:
        idx = pd.date_range("2023-01-01", periods=12, freq="MS")
        return pd.Series(np.arange(12, dtype=float) + 1, index=idx)

    def test_noop_when_nothing_declared(self) -> None:
        out = prepare_exog_options({}, self._series(), 3, "MS")
        assert "known_covariates" not in out
        assert "prophet_holidays" not in out

    def test_populates_all_channels(self) -> None:
        series = self._series()
        out = prepare_exog_options(
            {
                "holidays_country": "US",
                "known_events": [
                    {"type": "spike", "date": "2023-06-04", "label": "sale"},
                ],
                "known_covariates": {
                    "price": {
                        str(d.date()): float(i)
                        for i, d in enumerate(series.index)
                    }
                },
            },
            series,
            3,
            "MS",
        )
        assert out["covariates_known_in_advance"] is True
        assert "event_holiday" in out["known_covariates"]
        assert "event_spike" in out["known_covariates"]
        assert "price" in out["known_covariates"]
        assert not out["prophet_holidays"].empty
        assert out["prophet_regressors"] == out["known_covariates"]

    def test_user_covariates_only_still_set_known_in_advance(self) -> None:
        series = self._series()
        out = prepare_exog_options(
            {
                "known_covariates": {
                    "price": {str(d.date()): float(i) for i, d in enumerate(series.index)}
                }
            },
            series,
            3,
            "MS",
        )
        assert out["covariates_known_in_advance"] is True
        assert "price" in out["known_covariates"]
        # No holidays declared → no prophet_holidays frame.
        assert "prophet_holidays" not in out