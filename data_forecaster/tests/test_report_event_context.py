"""Report-only event interpretation must not alter model forecasts."""
import pandas as pd

from report.event_context import build_event_context


def series(start, values, freq="D"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq))


def test_christmas_matches_history_and_forecast_without_mutation():
    history = series("2024-12-24", [10, 30, 12])
    forecast = series("2025-12-24", [11, 28, 13])
    before = forecast.copy()
    context, historical_notes, forecast_notes = build_event_context(
        {"holidays_country": "CA", "holidays_subdivision": "ON"}, history, forecast
    )
    assert "Christmas" in historical_notes[0]
    assert "Historical local peak of 30" in historical_notes[0]
    assert "Forecast local peak of 28" in forecast_notes[0]
    assert "does not establish" in forecast_notes[0]
    assert {m["series"] for m in context["event_matches"]} == {"Historical", "Forecast"}
    pd.testing.assert_series_equal(forecast, before)


def test_valentines_is_report_observance():
    context, _, notes = build_event_context({"holidays_country": "US"}, None, series("2025-02-13", [5, 20, 6]))
    assert "Valentine's Day (observance)" in notes[0]
    assert context["event_matches"][0]["events"][0]["type"] == "observance"


def test_monthly_flat_and_gapped_series_do_not_claim_daily_peaks():
    options = {"holidays_country": "US"}
    for forecast in [series("2025-01-14", [5, 20, 6], "31D"), series("2025-02-13", [5, 5, 5]), series("2025-02-12", [5, 20, 6], "2D")]:
        _, _, notes = build_event_context(options, None, forecast)
        assert notes == []


def test_custom_events_and_covariate_values_are_retained():
    context, _, notes = build_event_context({
        "known_events": [{"date": "2025-02-14", "type": "promotion", "label": "Half-price sale"}],
        "known_covariates": {"price": {"2025-02-14": 5}},
    }, None, series("2025-02-13", [5, 20, 6]))
    assert "Half-price sale" in notes[0]
    assert context["declared_covariate_values"]["price"]["values"]["2025-02-14"] == 5


def test_regional_calendar_changes_report_matches():
    forecast = series("2025-06-23", [5, 20, 6])
    _, _, qc = build_event_context({"holidays_country": "CA", "holidays_subdivision": "QC"}, None, forecast)
    _, _, on = build_event_context({"holidays_country": "CA", "holidays_subdivision": "ON"}, None, forecast)
    assert qc
    assert not on
