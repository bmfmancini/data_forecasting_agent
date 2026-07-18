"""Compatibility bootstrap for importing pmdarima with newer scikit-learn."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import sklearn.utils.validation as sk_validation

from core.logging_config import get_logger
from utils.memory import estimate_arima_workspace_mb, memory_snapshot

logger = get_logger(__name__)


def patch_sklearn_check_array() -> None:
    """Patch scikit-learn's ``check_array`` for pmdarima 2.0.x compatibility.

    pmdarima 2.0.x still passes the removed ``force_all_finite`` keyword.
    scikit-learn 1.6 replaced it with ``ensure_all_finite``. This patch keeps
    the translation in one documented import bootstrap instead of scattering
    ordering-sensitive monkey patches through model modules.
    """
    if hasattr(sk_validation, "_patched_for_pmdarima"):
        return

    original_check_array = sk_validation.check_array

    def patched_check_array(*args: Any, **kwargs: Any) -> Any:
        if "force_all_finite" in kwargs:
            kwargs.setdefault("ensure_all_finite", kwargs.pop("force_all_finite"))
        return original_check_array(*args, **kwargs)

    sk_validation.check_array = patched_check_array
    sk_validation._patched_for_pmdarima = True


def import_pmdarima() -> ModuleType:
    """Patch scikit-learn, then import and return the pmdarima module."""
    patch_sklearn_check_array()

    import pmdarima as pm  # pylint: disable=import-outside-toplevel

    return pm


def fit_auto_arima_memory_aware(
    series: Any,
    *,
    seasonal_period: int = 1,
    **auto_kwargs: Any,
) -> Any:
    """Discover an order with bounded memory, then return a normal fitted model.

    The low-memory discovery path alone can lose prediction intervals. It is
    therefore followed by a conventional fixed-order refit, preserving the
    selected specification and normal forecast/residual behavior.
    """
    import core.config as settings  # local to keep configuration testable

    pm = import_pmdarima()
    estimated_mb = estimate_arima_workspace_mb(len(series), seasonal_period)
    use_low_memory = estimated_mb > settings.ARIMA_LOW_MEMORY_THRESHOLD_MB
    before = memory_snapshot()
    discovery_kwargs = dict(auto_kwargs)
    if use_low_memory:
        discovery_kwargs["low_memory"] = True
    discovered = pm.auto_arima(series, **discovery_kwargs)
    model = discovered
    if use_low_memory:
        model = pm.ARIMA(
            order=discovered.order,
            seasonal_order=getattr(discovered, "seasonal_order", (0, 0, 0, 0)),
            with_intercept=getattr(discovered, "with_intercept", None),
            suppress_warnings=True,
        ).fit(series)
    after = memory_snapshot()
    logger.info(
        "Memory model=auto_arima observations=%d seasonal_period=%d "
        "estimated_mb=%d low_memory_discovery=%s rss_mb=%.1f "
        "peak_rss_mb=%.1f rss_delta_mb=%.1f",
        len(series),
        seasonal_period,
        estimated_mb,
        use_low_memory,
        after.current_rss_mb,
        after.peak_rss_mb,
        after.current_rss_mb - before.current_rss_mb,
    )
    return model
