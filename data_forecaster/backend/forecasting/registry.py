"""Canonical registry of forecasting models and their enable/disable state.

The ``MODELS`` map is the single source of truth for which forecasting
models exist; the ``model_config`` database table holds the mutable
enabled/disabled state (managed via the admin panel). All agents must
consume :func:`get_enabled_models` rather than hardcoding model lists so
that disabled models are never fitted, assessed, or selected.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

import pandas as pd

from core.database import get_connection
from core.logging_config import get_logger
from forecasting.arima_model import fit_arima
from forecasting.ewma_model import fit_ewma
from forecasting.holt_winters import fit_holt_winters
from forecasting.prophet_model import fit_prophet
from forecasting.sarima_model import fit_sarima

logger = get_logger(__name__)

# Fit functions accept (series, forecast_horizon, **kwargs) and return the
# standard metrics dict (forecast, lower_ci, upper_ci, rmse, mae, mape).
FitFn = Callable[..., dict[str, Any]]

#: Canonical model catalog.  ``extra_kwargs`` maps model-specific keyword
#: arguments to keys of the fit context passed by the caller (e.g. the
#: forecasting agent supplies ``seasonal_period`` and ``freq``).
MODELS: dict[str, dict[str, Any]] = {
    "Holt-Winters": {
        "fit_fn": fit_holt_winters,
        "display_name": "Holt-Winters",
        "extra_kwargs": {},
    },
    "ARIMA": {
        "fit_fn": fit_arima,
        "display_name": "ARIMA",
        "extra_kwargs": {},
    },
    "SARIMA": {
        "fit_fn": fit_sarima,
        "display_name": "SARIMA",
        "extra_kwargs": {"seasonal_period": "seasonal_period"},
    },
    "EWMA": {
        "fit_fn": fit_ewma,
        "display_name": "EWMA",
        "extra_kwargs": {},
    },
    "Prophet": {
        "fit_fn": fit_prophet,
        "display_name": "Prophet",
        "extra_kwargs": {"freq": "freq"},
    },
}

#: Names in canonical (priority) order — used for heuristic fallbacks.
MODEL_NAMES: tuple[str, ...] = tuple(MODELS)


def _read_enabled_names(db_path: str | None = None) -> list[str]:
    """Return enabled model names from ``model_config`` (canonical order)."""
    with get_connection(db_path) as connection:
        rows: list[sqlite3.Row] = connection.execute(
            "SELECT name FROM model_config WHERE enabled = 1 ORDER BY priority"
        ).fetchall()
    enabled = {row["name"] for row in rows}
    return [name for name in MODEL_NAMES if name in enabled]


def get_enabled_models(db_path: str | None = None) -> tuple[str, ...]:
    """Return the names of currently enabled models.

    Defends the at-least-one-enabled invariant: if the table is missing,
    unreadable, or somehow all-disabled, all models are treated as enabled
    and a warning is logged.

    Args:
        db_path: Optional database path override (testing).

    Returns:
        A tuple of enabled model names in canonical order.
    """
    try:
        enabled = _read_enabled_names(db_path)
    except sqlite3.Error as exc:
        logger.warning(
            "model_config unreadable (%s) — treating all models as enabled.",
            exc,
        )
        return MODEL_NAMES
    if not enabled:
        logger.warning(
            "model_config has no enabled models — treating all as enabled."
        )
        return MODEL_NAMES
    return tuple(enabled)


def set_model_enabled(
    name: str, enabled: bool, db_path: str | None = None
) -> None:
    """Enable or disable a model, enforcing the at-least-one invariant.

    Args:
        name: Model name (must exist in :data:`MODELS`).
        enabled: ``True`` to enable, ``False`` to disable.
        db_path: Optional database path override (testing).

    Raises:
        ValueError: When the model name is unknown, or the change would
            leave zero models enabled.
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name=}")
    currently_enabled = set(get_enabled_models(db_path))
    if not enabled and currently_enabled == {name}:
        raise ValueError(
            "Cannot disable the last enabled model — at least one model "
            "must remain enabled."
        )
    with get_connection(db_path) as connection:
        connection.execute(
            "UPDATE model_config SET enabled = ? WHERE name = ?",
            (1 if enabled else 0, name),
        )
        connection.commit()
    logger.info("Model '%s' %s.", name, "enabled" if enabled else "disabled")


def list_model_states(db_path: str | None = None) -> list[dict[str, Any]]:
    """Return all models with their enabled state (for the admin UI).

    Args:
        db_path: Optional database path override (testing).

    Returns:
        A list of dicts with ``name``, ``display_name``, and ``enabled``
        keys, in canonical order.
    """
    enabled = set(get_enabled_models(db_path))
    return [
        {
            "name": name,
            "display_name": MODELS[name]["display_name"],
            "enabled": name in enabled,
        }
        for name in MODEL_NAMES
    ]


def get_fit_functions(
    context: dict[str, Any], db_path: str | None = None
) -> list[tuple[str, FitFn, dict[str, Any]]]:
    """Return ``(name, fit_fn, kwargs)`` triples for enabled models.

    Args:
        context: Fit context supplying values referenced by each model's
            ``extra_kwargs`` mapping (e.g. ``seasonal_period``, ``freq``).
        db_path: Optional database path override (testing).

    Returns:
        A list of triples ready to iterate in the forecasting fit loop.
    """
    triples: list[tuple[str, FitFn, dict[str, Any]]] = []
    for name in get_enabled_models(db_path):
        entry = MODELS[name]
        kwargs = {
            kwarg: context[source]
            for kwarg, source in entry["extra_kwargs"].items()
            if source in context
        }
        triples.append((name, entry["fit_fn"], kwargs))
    return triples
