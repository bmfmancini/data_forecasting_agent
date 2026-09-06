"""Dated business context for interpretation, independent of model inputs."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from forecasting.known_context import merge_events


def build_event_context(
    options: dict[str, Any] | None,
    historical: pd.Series | None,
    forecast: pd.Series,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Match daily local extrema to declared events and regional calendars.

    Observances added here are report-only context, never model regressors.
    Exact-day matching avoids treating monthly totals as holiday-day spikes.
    Context is bounded for prompts; notes identify coincidences, not causes.
    """
    options = options or {}
    history = historical if historical is not None else pd.Series(dtype=float)
    dates = list(history.index) + list(forecast.index)
    if not dates:
        return {}, [], []
    dates = pd.DatetimeIndex(dates)
    start, end = dates.min(), dates.max()
    years = range(start.year, end.year + 1)
    events = merge_events(options, years)
    # Commercial observance in these calendars; not a statutory holiday.
    if str(options.get("holidays_country", "")).upper() in {"US", "CA", "GB", "IE", "AU", "NZ"}:
        events.extend({"date": f"{year}-02-14", "type": "observance", "label": "Valentine's Day"} for year in years)
    events = [e for e in events if start.normalize() <= pd.Timestamp(e["date"]) <= end.normalize()]
    events = sorted({(e["date"], e["label"], e["type"]): e for e in events}.values(), key=lambda e: (e["date"], e["label"]))
    by_date: dict[str, list[dict[str, str]]] = {}
    for event in events:
        by_date.setdefault(event["date"], []).append(event)

    def match(series: pd.Series, label: str) -> tuple[list[dict[str, Any]], list[str]]:
        if len(series) < 3:
            return [], []
        series = series.sort_index()
        idx = pd.DatetimeIndex(series.index)
        values = np.asarray(series, dtype=float)
        matches = []
        for i in range(1, len(series) - 1):
            # Only compare consecutive daily observations; do not bridge gaps.
            if idx[i] - idx[i-1] != pd.Timedelta(days=1) or idx[i+1] - idx[i] != pd.Timedelta(days=1):
                continue
            left, value, right = values[i-1:i+2]
            if not np.isfinite([left, value, right]).all():
                continue
            kind = "local peak" if value > max(left, right) else "local trough" if value < min(left, right) else None
            day = idx[i].strftime("%Y-%m-%d")
            if not kind or day not in by_date:
                continue
            matches.append({"series": label, "date": day, "value": float(value), "previous_value": float(left), "next_value": float(right), "pattern": kind, "events": by_date[day]})
        notes = []
        for item in matches[:12]:
            names = "; ".join(f"{e['label']} ({e['type']})" for e in item["events"])
            notes.append(f"{label} {item['pattern']} of {item['value']:g} on {item['date']} coincides with {names}. This timing may be relevant to interpretation; it does not establish that the event caused the movement. Forecast values have not been adjusted for this note.")
        return matches, notes

    history_matches, history_notes = match(history, "Historical")
    forecast_matches, forecast_notes = match(forecast, "Forecast")
    matches = history_matches + forecast_matches
    # Preserve matched events first when the full calendar exceeds the cap.
    matched_events = [e for m in matches for e in m["events"]]
    ordered = list({(e["date"], e["label"], e["type"]): e for e in matched_events + events}.values())
    covariates = options.get("known_covariates") or {}
    covariate_context = {}
    for name, values in list(covariates.items())[:10]:
        if isinstance(values, dict):
            points = sorted((str(d), v) for d, v in values.items())
            covariate_context[name] = {"values": dict(points[:24]), "omitted_values": max(0, len(points)-24)}
    context = {
        "dated_events": ordered[:100],
        "omitted_events": max(0, len(ordered)-100),
        "event_matches": matches[:24],
        "omitted_matches": max(0, len(matches)-24),
        "declared_covariate_values": covariate_context,
        "interpretation_rule": "Use dates and supplied values as contextual evidence regardless of selected model. Historical observations and forecast projections are distinct. Coincidence is not causation; local extrema are not statistically established anomalies. Do not claim an event effect was estimated or modify forecast values. Do not infer a daily spike from an aggregated period. Observances are not necessarily public holidays. Notes are qualitative context, not validated causal explanations.",
    }
    return context if events or covariate_context else {}, history_notes, forecast_notes
