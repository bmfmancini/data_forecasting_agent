"""Index normalization for forecasting libraries with strict index support."""

from __future__ import annotations

import pandas as pd

from core.logging_config import get_logger

logger = get_logger(__name__)


def normalize_forecast_index(series: pd.Series) -> pd.Series:
    """Return a series with an index supported by Statsmodels forecasting.

    Statsmodels cannot generate out-of-sample timestamps for a non-unique or
    irregular ``DatetimeIndex``. Forecast adapters operate on ordered values,
    so use a positional index in those cases while retaining regular datetime
    indexes for models that can safely extend them.

    Args:
        series: Clean numeric observations in chronological order.

    Returns:
        The original series when its index is regular and unique; otherwise a
        value-identical series using a ``RangeIndex``.
    """
    index = series.index
    if (
        isinstance(index, pd.DatetimeIndex)
        and index.is_unique
        and index.freq is not None
    ):
        return series

    logger.warning(
        "Unsupported forecasting index detected; using a positional index."
    )
    return pd.Series(series.to_numpy(dtype=float), name=series.name)
