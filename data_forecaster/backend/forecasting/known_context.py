"""Structured business context → forecasting model inputs.

Pure functions (no I/O, no LLM) that translate the user-facing business context
carried in ``preflight_options`` — a holiday country, a list of custom known
events (spikes, lulls, interventions, promotions, outages), and a set of
future-known covariates — into the shapes the forecasting models and the
executive report consume.

The preflight wizard captures:

* ``holidays_country`` — an ISO country code selected from a pre-populated
  dropdown; the backend expands it into dated holidays for the data's year
  range plus the forecast horizon (users do not type holiday dates).
* ``known_events`` — a list of ``{type, date, label}`` dicts for non-standard
  events the user adds by hand. Expanded country holidays are merged into this
  list server-side as ``type == "holiday"`` entries.
* ``known_covariates`` — a ``{name: {date: value}}`` dict captured as inline
  name + date:value rows. This is the exact shape
  :func:`forecasting.dynamic_regression.design_matrix` already ingests.

Models that cannot ingest exogenous regressors (ARIMA, SARIMA, EWMA,
Holt-Winters) receive no model input here; the structured context is used to
enrich the executive report instead (see :mod:`report.builder`).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import holidays as _holidays
import pandas as pd

#: Recognised event types. ``holiday`` is produced by country expansion; the
#: rest are user-declared custom events.
EVENT_TYPES: tuple[str, ...] = (
    "holiday",
    "spike",
    "lull",
    "intervention",
    "promotion",
    "outage",
)

#: Curated list of common countries (ISO alpha-2 code → display name) for the
#: preflight dropdown. The ``holidays`` library supports many more codes; any
#: code it recognises is accepted at expansion time even if not listed here.
_COUNTRIES: dict[str, str] = {
    "US": "United States",
    "CA": "Canada",
    "GB": "United Kingdom",
    "IE": "Ireland",
    "AU": "Australia",
    "NZ": "New Zealand",
    "DE": "Germany",
    "FR": "France",
    "IT": "Italy",
    "ES": "Spain",
    "PT": "Portugal",
    "NL": "Netherlands",
    "BE": "Belgium",
    "CH": "Switzerland",
    "AT": "Austria",
    "SE": "Sweden",
    "NO": "Norway",
    "DK": "Denmark",
    "FI": "Finland",
    "PL": "Poland",
    "CZ": "Czechia",
    "RU": "Russia",
    "UA": "Ukraine",
    "TR": "Turkey",
    "GR": "Greece",
    "MX": "Mexico",
    "BR": "Brazil",
    "AR": "Argentina",
    "CL": "Chile",
    "CO": "Colombia",
    "ZA": "South Africa",
    "EG": "Egypt",
    "NG": "Nigeria",
    "SA": "Saudi Arabia",
    "AE": "United Arab Emirates",
    "IL": "Israel",
    "IN": "India",
    "PK": "Pakistan",
    "BD": "Bangladesh",
    "CN": "China",
    "JP": "Japan",
    "KR": "South Korea",
    "SG": "Singapore",
    "MY": "Malaysia",
    "ID": "Indonesia",
    "PH": "Philippines",
    "TH": "Thailand",
    "VN": "Vietnam",
    "TW": "Taiwan",
    "HK": "Hong Kong",
}


def country_list() -> list[tuple[str, str]]:
    """Return ``(code, display_name)`` pairs for the preflight dropdown.

    Sorted by display name. The frontend renders ``options`` as the codes and
    ``option_labels`` as the display names.
    """
    return sorted(_COUNTRIES.items(), key=lambda kv: kv[1])


def subdivision_options() -> dict[str, list[dict[str, str]]]:
    """Return supported regions and names from the installed holiday calendars."""
    result = {}
    for code, _ in country_list():
        calendar = _holidays.country_holidays(code)
        aliases = getattr(calendar, "subdivisions_aliases", {})
        result[code] = [
            {"code": region, "label": next(
                (name for name, target in aliases.items() if target == region), region
            )}
            for region in calendar.subdivisions
        ]
    return result


def country_name(code: str | None) -> str | None:
    """Return the display name for a country code, or ``None`` if unknown."""
    if not code:
        return None
    return _COUNTRIES.get(str(code).strip().upper())


def _valid_date(value: Any) -> str | None:
    """Return a normalised ``YYYY-MM-DD`` string, or ``None`` if unparseable."""
    if not value:
        return None
    try:
        ts = pd.to_datetime(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d")


def expand_holidays(country: str | None, years: Iterable[int], subdivision: str | None = None) -> list[dict[str, str]]:
    """Expand a country code into dated holiday events.

    Args:
        country: ISO alpha-2 code (e.g. ``"US"``). ``None``/empty → no events.
        years: Year integers to expand (data range plus forecast horizon).

    Returns:
        A list of ``{"type": "holiday", "date": "YYYY-MM-DD", "label": name}``
        dicts, sorted by date. An unknown/unsupported code yields an empty
        list rather than raising.
    """
    if not country:
        return []
    code = str(country).strip().upper()
    subdivision = str(subdivision).strip() if subdivision else None
    if subdivision:
        calendar = _holidays.country_holidays(code)
        if subdivision not in calendar.subdivisions:
            raise ValueError(f"Unsupported state/province {subdivision!r} for {code}.")
    try:
        cal = _holidays.country_holidays(code, years=list(dict.fromkeys(years)), subdiv=subdivision)
    except Exception:
        return []
    return [
        {"type": "holiday", "date": d.isoformat(), "label": str(name)}
        for d, name in sorted(cal.items())
    ]


def merge_events(
    preflight_options: dict[str, Any] | None, years: Iterable[int]
) -> list[dict[str, str]]:
    """Combine expanded country holidays with user-declared custom events.

    Args:
        preflight_options: Raw preflight options dict.
        years: Year integers covering the data range and forecast horizon.

    Returns:
        A merged list of event dicts (holidays first, then custom events).
    """
    options = preflight_options or {}
    events: list[dict[str, str]] = []
    events.extend(expand_holidays(options.get("holidays_country"), years, options.get("holidays_subdivision")))
    custom = options.get("known_events") or []
    for event in custom:
        if not isinstance(event, dict):
            continue
        date = _valid_date(event.get("date"))
        if not date:
            continue
        events.append(
            {
                "type": str(event.get("type") or "event").lower(),
                "date": date,
                "label": str(event.get("label") or event.get("type") or "event"),
            }
        )
    return events


def summarize_context(preflight_options: dict[str, Any] | None) -> dict[str, Any]:
    """Summarise declared context for the report (no date expansion needed).

    Unlike :func:`merge_events`, this does not require the data's year range;
    it counts user-declared custom events by type, records the holiday country
    code, and lists covariate names. Used by the report builder to enrich
    §10–12 narratives regardless of whether the selected model ingests exog.

    Returns:
        A dict with ``holidays_country``, ``events_by_type`` (counts),
        ``event_count``, and ``covariates`` (names). Empty values when nothing
        was declared.
    """
    options = preflight_options or {}
    custom = options.get("known_events") or []
    by_type: dict[str, int] = {}
    valid_count = 0
    for event in custom:
        if not isinstance(event, dict) or not _valid_date(event.get("date")):
            continue
        valid_count += 1
        kind = str(event.get("type") or "event").lower()
        by_type[kind] = by_type.get(kind, 0) + 1
    country = options.get("holidays_country")
    cov_names = list((options.get("known_covariates") or {}).keys())
    return {
        "holidays_country": country if country else None,
        "holidays_subdivision": options.get("holidays_subdivision") or None if country else None,
        "events_by_type": by_type,
        "event_count": valid_count,
        "covariates": cov_names,
    }


def to_prophet_holidays(events: list[dict[str, str]]) -> pd.DataFrame:
    """Build a Prophet ``holidays`` frame (``ds``, ``holiday``) from events.

    All known events — holidays, spikes, lulls, interventions, promotions,
    outages — are treated as Prophet "holidays" (known dated effects). Prophet
    estimates a separate shift for each named holiday.
    """
    rows: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        date = _valid_date(event.get("date"))
        if not date:
            continue
        rows.append(
            {
                "ds": pd.to_datetime(date),
                "holiday": str(event.get("label") or event.get("type") or "event"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["ds", "holiday"])
    frame = pd.DataFrame(rows)
    # Prophet requires unique (ds, holiday) rows; keep the first label per date.
    return frame.drop_duplicates(subset=["ds", "holiday"]).reset_index(drop=True)


def to_event_dummies(
    events: list[dict[str, str]], date_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Build 0/1 indicator columns per event type over ``date_index``.

    One column per event type present (e.g. ``holiday``, ``spike``), value 1.0
    on dates matching any event of that type, 0.0 otherwise. Used to construct
    exogenous regressors for models that accept them.
    """
    idx = pd.DatetimeIndex(date_index)
    # (type, date) pairs with valid dates only.
    pairs: list[tuple[str, pd.Timestamp]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        date = _valid_date(event.get("date"))
        if not date:
            continue
        pairs.append((str(event.get("type") or "event").lower(), pd.to_datetime(date)))
    types_present = sorted({kind for kind, _ in pairs})
    if not types_present:
        return pd.DataFrame(index=idx)
    columns: dict[str, list[float]] = {}
    for kind in types_present:
        dates = {d for k, d in pairs if k == kind}
        columns[f"event_{kind}"] = [1.0 if d in dates else 0.0 for d in idx]
    return pd.DataFrame(columns, index=idx)


def prepare_exog_options(
    options: dict[str, Any] | None,
    series: pd.Series,
    horizon: int,
    freq: str,
) -> dict[str, Any]:
    """Populate engine options with model-ready exogenous context.

    Returns a copy of ``options`` with, when the user declared any holidays,
    events, or covariates:

    * ``known_covariates`` — the ``{name: {date: value}}`` dict consumed by
      :func:`forecasting.dynamic_regression.design_matrix` (event-type
      dummies + user covariates, reindexed onto the full data + forecast
      horizon date range).
    * ``covariates_known_in_advance`` — ``True`` (required by design_matrix).
    * ``prophet_holidays`` — a Prophet ``holidays`` frame (``ds``, ``holiday``).
    * ``prophet_regressors`` — the same covariate dict, for Prophet's
      ``add_regressor`` path.

    The date index covers the full prepared series plus the forecast horizon so
    every rolling-origin fold (and the final production fit) can reindex it.
    """
    out = dict(options or {})
    merged = merge_events(options, _years_for(series, horizon, freq))
    if not merged and not (out.get("known_covariates")):
        return out
    future = pd.date_range(series.index[-1], periods=horizon + 1, freq=freq)[1:]
    date_index = series.index.append(future)
    covariates = to_known_covariates(merged, out.get("known_covariates"), date_index)
    if covariates:
        out["known_covariates"] = covariates
        out["covariates_known_in_advance"] = True
        out["prophet_regressors"] = covariates
    if merged:
        out["prophet_holidays"] = to_prophet_holidays(merged)
    return out


def _years_for(series: pd.Series, horizon: int, freq: str) -> range:
    """Year range spanning the series plus the forecast horizon."""
    start = series.index.min()
    future = pd.date_range(series.index[-1], periods=horizon + 1, freq=freq)[1:]
    end = future[-1] if len(future) else series.index[-1]
    return range(int(start.year), int(end.year) + 1)


def to_known_covariates(
    events: list[dict[str, str]],
    user_covariates: dict[str, dict[str, float]] | None,
    date_index: pd.DatetimeIndex,
) -> dict[str, dict[str, float]]:
    """Build the ``{name: {date: value}}`` dict that ``design_matrix`` ingests.

    Event-type dummies (1.0 on event dates, 0.0 elsewhere) are combined with
    user-supplied covariates, all reindexed onto ``date_index``. Event dummies
    are 0.0-filled; user covariates keep NaN where the user did not supply a
    value, so :func:`forecasting.dynamic_regression.design_matrix` raises a
    clear "needs finite values at every timestamp" error rather than silently
    imputing zero.

    Args:
        events: Merged event list (holidays + custom).
        user_covariates: ``{name: {date: value}}`` from preflight, or ``None``.
        date_index: Full timestamp index (data dates + forecast horizon dates).

    Returns:
        A dict suitable for ``options["known_covariates"]``.
    """
    idx = pd.DatetimeIndex(date_index)
    out: dict[str, dict[str, float]] = {}
    dummies = to_event_dummies(events, idx)
    for column in dummies.columns:
        out[column] = {
            d.strftime("%Y-%m-%d"): float(v) for d, v in zip(idx, dummies[column])
        }
    for name, series in (user_covariates or {}).items():
        if not isinstance(series, dict):
            continue
        cleaned: dict[str, float] = {}
        for date, value in series.items():
            normalised = _valid_date(date)
            if not normalised:
                continue
            try:
                cleaned[normalised] = float(value)
            except (TypeError, ValueError):
                continue
        if not cleaned:
            continue
        values = pd.Series(cleaned, dtype=float)
        values.index = pd.to_datetime(values.index)
        aligned = values.reindex(idx)
        out[str(name)] = {
            d.strftime("%Y-%m-%d"): (float(v) if pd.notna(v) else float("nan"))
            for d, v in aligned.items()
        }
    return out