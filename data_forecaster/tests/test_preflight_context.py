"""Tests for the structured business-context preflight decisions.

Verifies the new ``kind``-tagged decisions (holiday country, custom events,
covariates) are emitted by :func:`run_preflight_checks`, that their sentinels
are preserved in the defaults, and that the ``PreflightDecision`` schema
carries ``kind`` / ``option_labels`` round-trip.
"""

from __future__ import annotations

import pandas as pd

from forecasting.known_context import country_list
from schemas import PreflightDecision
from utils.preflight import run_preflight_checks


def _df(n: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({"date": idx, "value": range(n)})


class TestPreflightStructuredDecisions:
    def test_emits_country_events_covariates_decisions(self) -> None:
        response = run_preflight_checks(_df(), "date", "value", 5)
        keys = {d.key: d for d in response.decisions}
        assert "holidays_country" in keys
        assert "known_events" in keys
        assert "known_covariates" in keys

    def test_country_decision_has_kind_and_labels(self) -> None:
        response = run_preflight_checks(_df(), "date", "value", 5)
        country = next(d for d in response.decisions if d.key == "holidays_country")
        assert country.kind == "country"
        assert country.allow_custom is True
        # Codes and display names are parallel lists.
        assert len(country.options) == len(country.option_labels)
        assert "US" in country.options
        us_index = country.options.index("US")
        assert country.option_labels[us_index] == "United States"

    def test_events_and_covariates_have_kinds(self) -> None:
        response = run_preflight_checks(_df(), "date", "value", 5)
        events = next(d for d in response.decisions if d.key == "known_events")
        covs = next(d for d in response.decisions if d.key == "known_covariates")
        assert events.kind == "dates"
        assert covs.kind == "covariates"

    def test_defaults_preserve_sentinels(self) -> None:
        response = run_preflight_checks(_df(), "date", "value", 5)
        assert response.defaults["holidays_country"] == ""
        assert response.defaults["known_events"] == []
        assert response.defaults["known_covariates"] == {}
        assert response.defaults["covariates_known_in_advance"] is False

    def test_country_options_match_known_context_list(self) -> None:
        response = run_preflight_checks(_df(), "date", "value", 5)
        country = next(d for d in response.decisions if d.key == "holidays_country")
        expected = dict(country_list())
        assert dict(zip(country.options, country.option_labels)) == expected


class TestPreflightDecisionSchema:
    def test_kind_defaults_to_select(self) -> None:
        decision = PreflightDecision(
            key="x", label="X", message="m", options=["a", "b"], default="a"
        )
        assert decision.kind == "select"
        assert decision.option_labels == []

    def test_kind_and_labels_round_trip(self) -> None:
        decision = PreflightDecision(
            key="holidays_country",
            label="Holiday calendar",
            message="m",
            options=["US", "CA"],
            option_labels=["United States", "Canada"],
            default="",
            kind="country",
        )
        assert decision.kind == "country"
        assert decision.option_labels == ["United States", "Canada"]