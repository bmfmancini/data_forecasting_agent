"""Shared terminal-holdout evaluation used by models and baselines.

This module owns split generation and metric scoring. Model adapters provide
predictions; they do not define metric formulas or missing-value conventions.
The rolling-origin backtesting service will replace the single split with
multiple origins while preserving this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting.contracts import ForecastMetrics
from forecasting.metrics import calculate_forecast_metrics


@dataclass(frozen=True)
class TerminalHoldout:
    """One auditable terminal train/test split."""

    train: pd.Series
    test: pd.Series


def make_terminal_holdout(
    series: pd.Series,
    forecast_horizon: int,
) -> TerminalHoldout:
    """Create the common terminal holdout used by every candidate."""
    if forecast_horizon < 1 or len(series) < 2:
        return TerminalHoldout(series.iloc[:0], series.iloc[:0])
    split = max(
        1,
        min(
            len(series) - 1,
            max(int(len(series) * 0.8), len(series) - forecast_horizon),
        ),
    )
    return TerminalHoldout(series.iloc[:split], series.iloc[split:])


def evaluate_predictions(
    split: TerminalHoldout,
    predicted: np.ndarray | pd.Series,
    *,
    mase_period: int,
) -> ForecastMetrics:
    """Score aligned holdout predictions using central metric conventions."""
    return calculate_forecast_metrics(
        split.test,
        predicted,
        training=split.train,
        mase_period=mase_period,
    )
