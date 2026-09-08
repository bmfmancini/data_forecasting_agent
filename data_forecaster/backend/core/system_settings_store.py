"""Leaf module for the deployment-wide "Enable AI features" switch.

The flag lives in the ``system_settings`` singleton table so it is readable
on every install — including Traditional Forecasting-only deployments that
never write an ``llm_config`` row.  It is kept in this leaf module (rather
than in :mod:`services.setup_service`) because :mod:`core.llm_factory`
must consult it on every LLM construction, and putting the accessor in a
service would create an import cycle.

:mod:`services.setup_service` re-exports these helpers so callers can use
either module; new code should prefer this one.
"""

from __future__ import annotations

import sqlite3

from core.database import get_connection

_SYSTEM_SETTINGS_SINGLETON_WHERE = " WHERE singleton = 1"


def is_llm_enabled(db_path: str | None = None) -> bool:
    """Return whether AI features are enabled deployment-wide.

    Defaults to ``True`` when the singleton row or table is missing so a
    partially initialised database never silently disables AI.

    Args:
        db_path: Optional database path override (testing).

    Returns:
        ``True`` when LLM features are enabled.
    """
    connection = get_connection(db_path)
    try:
        row: sqlite3.Row | None = connection.execute(
            "SELECT llm_enabled FROM system_settings" + _SYSTEM_SETTINGS_SINGLETON_WHERE
        ).fetchone()
        return bool(row["llm_enabled"]) if row else True
    except sqlite3.OperationalError:
        # Pre-migration database (no system_settings table yet): AI stays
        # on.  init_database() adds the column on every startup.
        return True
    finally:
        connection.close()


def set_llm_enabled(enabled: bool, db_path: str | None = None) -> None:
    """Set the deployment-wide AI features switch.

    Args:
        enabled: ``True`` to enable AI features, ``False`` to force
            Traditional Forecasting deployment-wide.
        db_path: Optional database path override (testing).
    """
    connection = get_connection(db_path)
    try:
        connection.execute(
            "UPDATE system_settings SET llm_enabled = ?,"
            " updated_at = datetime('now')" + _SYSTEM_SETTINGS_SINGLETON_WHERE,
            (int(bool(enabled)),),
        )
        connection.commit()
    finally:
        connection.close()