"""Compatibility bootstrap for importing prophet (Meta Prophet).

Prophet is a heavy optional dependency backed by the cmdstanpy toolchain.
To keep the rest of the forecasting package importable when prophet is not
installed, model adapters import it lazily through :func:`import_prophet`
rather than at module load time.  This mirrors the
:mod:`forecasting.pmdarima_compat` pattern.
"""

from __future__ import annotations

import logging
from types import ModuleType


def _silence_chatty_loggers() -> None:
    """Quiet the verbose cmdstanpy/prophet loggers before importing prophet.

    Prophet and its cmdstanpy backend log heavily to stdout/stderr while
    fitting.  Silencing them here (rather than in every caller) keeps the
    forecasting pipeline output readable.  Idempotent.
    """
    for name in ("cmdstanpy", "prophet", "prophet.models"):
        logging.getLogger(name).setLevel(logging.WARNING)


def import_prophet() -> ModuleType:
    """Import and return the ``prophet`` module.

    Returns:
        The imported ``prophet`` module (callers use ``prophet.Prophet``).

    Raises:
        ImportError: If prophet is not installed.  Callers in the forecasting
            pipeline wrap this in try/except so the model is skipped rather
            than crashing the whole run.
    """
    _silence_chatty_loggers()
    import prophet  # pylint: disable=import-outside-toplevel

    return prophet
