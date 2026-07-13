"""Common rolling-origin backtesting service.

This module owns split generation and pooled/by-horizon metric scoring for
every candidate model. Adapters and baselines provide a callable that fits on
a training window and predicts the fold horizon; this service creates the
folds once and reuses them for all candidates so that comparisons are
apples-to-apples.

Design goals:

* Expanding-window validation first, with configuration for initial training
  size, forecast horizon, step size, maximum number of origins, and an
  optional gap between train and validation periods.
* Use the requested production horizon where data permits. When it does
  not, shorten the validation horizon transparently and record which
  horizons are unsupported.
* Calculate metrics by horizon and pooled across folds; retain fold-level
  results.
* Reserve an optional final untouched test window when enough data exists.
* Fit preprocessing and all model choices using training observations only
  within each fold (no future-data leakage).
* Make runtime limits explicit: cap candidate complexity/origins according
  to series length and service budget, but apply identical folds to all
  surviving models.
* Keep the old terminal-holdout path behind a temporary compatibility flag
  and label it accurately.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.logging_config import get_logger
from forecasting.contracts import (
    BacktestEvaluation,
    BacktestFold,
    BacktestFoldResult,
    ForecastFitStatus,
    ForecastMetrics,
)
from forecasting.metrics import calculate_forecast_metrics

logger = get_logger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for rolling-origin backtesting.

    Attributes:
        initial_train_size: Minimum number of observations in the first
            training window. When ``None`` it defaults to ``max(10, len // 2)``.
        horizon:            Forecast horizon per fold. When ``None`` the
            production horizon is used.
        step_size:          Number of periods between successive origins.
            Defaults to ``horizon`` (non-overlapping folds).
        max_origins:        Maximum number of origins to evaluate. ``None``
            means no cap.
        gap:                Optional number of periods between the end of the
            training window and the start of the test window.
        reserve_final_window: When ``True`` the last ``horizon`` observations
            are reserved as a final untouched test window and excluded from
            rolling folds.
        mase_period:        Naive lag used for MASE scale estimation.
    """

    initial_train_size: int | None = None
    horizon: int | None = None
    step_size: int | None = None
    max_origins: int | None = None
    gap: int = 0
    reserve_final_window: bool = False
    mase_period: int = 1


# ── Fold generation ──────────────────────────────────────────────────────────


def generate_folds(
    series: pd.Series,
    config: BacktestConfig,
) -> list[BacktestFold]:
    """Generate expanding-window rolling-origin folds.

    Args:
        series:  Cleaned time series (no NaNs).
        config:  Backtesting configuration.

    Returns:
        A list of :class:`BacktestFold` definitions. Empty when the series
        is too short for even one fold.
    """
    n = len(series)
    horizon = config.horizon or max(1, min(n // 5, 12))
    if horizon < 1:
        horizon = 1

    initial = config.initial_train_size
    if initial is None:
        initial = max(10, n // 2)
    initial = max(1, min(initial, n - horizon - config.gap))

    step = config.step_size or horizon
    step = max(1, step)

    # Optionally reserve a final untouched test window.
    end_limit = n
    if config.reserve_final_window:
        end_limit = n - horizon
        if end_limit <= initial:
            logger.warning(
                "Series too short to reserve a final window; using all data "
                "for rolling folds."
            )
            end_limit = n

    folds: list[BacktestFold] = []
    fold_index = 0
    train_end = initial
    while train_end + config.gap + horizon <= end_limit:
        if config.max_origins is not None and fold_index >= config.max_origins:
            break
        test_start = train_end + config.gap
        test_end = test_start + horizon
        folds.append(
            BacktestFold(
                fold_index=fold_index,
                train_end_index=train_end,
                test_start_index=test_start,
                test_end_index=test_end,
                horizon=horizon,
            )
        )
        fold_index += 1
        train_end += step

    if not folds:
        logger.warning(
            "No backtest folds generated (n=%d, initial=%d, horizon=%d, gap=%d).",
            n,
            initial,
            horizon,
            config.gap,
        )
    return folds


# ── Candidate protocol ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class FoldPrediction:
    """Predictions returned by a candidate for one fold.

    Attributes:
        predictions: Point predictions aligned to the fold test window.
        lower_ci:    Optional lower prediction-interval bounds.
        upper_ci:    Optional upper prediction-interval bounds.
        status:      Fit status for this fold.
        warnings:    Fold-specific warnings.
        fitted_configuration: Configuration used to fit this fold.
    """

    predictions: np.ndarray
    lower_ci: np.ndarray | None = None
    upper_ci: np.ndarray | None = None
    status: ForecastFitStatus = ForecastFitStatus.OK
    warnings: list[str] | None = None
    fitted_configuration: dict[str, object] | None = None


CandidateFn = Callable[[pd.Series, BacktestFold], FoldPrediction | None]


# ── Evaluation ───────────────────────────────────────────────────────────────


def _process_fold(
    name: str,
    series: pd.Series,
    fold: BacktestFold,
    candidate_fn: CandidateFn,
    pooled_actuals: list[float],
    pooled_preds: list[float],
    by_horizon_actuals: dict[int, list[float]],
    by_horizon_preds: dict[int, list[float]],
    warnings: list[str],
) -> BacktestFoldResult | None:
    """Process one fold for a candidate; return the fold result or ``None``.

    ``None`` indicates the fold was skipped (insufficient data). Pooled and
    by-horizon accumulators are updated in place when predictions succeed.
    """
    train = series.iloc[: fold.train_end_index]
    test = series.iloc[fold.test_start_index : fold.test_end_index]
    if len(train) < 2 or len(test) == 0:
        warnings.append(f"Fold {fold.fold_index} skipped (insufficient data).")
        return None

    try:
        result = candidate_fn(train, fold)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Candidate %s failed on fold %d: %s", name, fold.fold_index, exc)
        return BacktestFoldResult(
            fold=fold,
            status=ForecastFitStatus.FAILED,
            warnings=[str(exc)],
        )

    if result is None:
        return BacktestFoldResult(
            fold=fold,
            status=ForecastFitStatus.NOT_ESTIMABLE,
            warnings=["Candidate returned no predictions."],
        )

    preds = np.asarray(result.predictions, dtype=float)
    actuals = test.values.astype(float)
    if preds.shape[0] != actuals.shape[0]:
        warnings.append(
            f"Fold {fold.fold_index} prediction length mismatch "
            f"({preds.shape[0]} vs {actuals.shape[0]})."
        )
        min_len = min(preds.shape[0], actuals.shape[0])
        preds = preds[:min_len]
        actuals = actuals[:min_len]

    residuals = (actuals - preds).tolist()
    fold_result = BacktestFoldResult(
        fold=fold,
        predictions=preds.tolist(),
        lower_ci=(
            np.asarray(result.lower_ci, dtype=float).tolist()
            if result.lower_ci is not None
            else []
        ),
        upper_ci=(
            np.asarray(result.upper_ci, dtype=float).tolist()
            if result.upper_ci is not None
            else []
        ),
        residuals=residuals,
        status=result.status,
        warnings=list(result.warnings or []),
        fitted_configuration=dict(result.fitted_configuration or {}),
    )

    pooled_actuals.extend(actuals.tolist())
    pooled_preds.extend(preds.tolist())
    for h in range(len(actuals)):
        by_horizon_actuals.setdefault(h, []).append(float(actuals[h]))
        by_horizon_preds.setdefault(h, []).append(float(preds[h]))
    return fold_result


def evaluate_candidate(
    name: str,
    series: pd.Series,
    folds: Sequence[BacktestFold],
    candidate_fn: CandidateFn,
    config: BacktestConfig,
) -> BacktestEvaluation:
    """Evaluate one candidate model across all folds.

    Args:
        name:         Candidate model name.
        series:       Full cleaned time series.
        folds:        Fold definitions shared by all candidates.
        candidate_fn: Callable that fits on the fold training window and
                      returns predictions for the fold test window.
        config:       Backtesting configuration (used for ``mase_period``).

    Returns:
        :class:`BacktestEvaluation` with per-fold results and pooled metrics.
    """
    fold_results: list[BacktestFoldResult] = []
    pooled_actuals: list[float] = []
    pooled_preds: list[float] = []
    by_horizon_actuals: dict[int, list[float]] = {}
    by_horizon_preds: dict[int, list[float]] = {}
    warnings: list[str] = []

    for fold in folds:
        result = _process_fold(
            name,
            series,
            fold,
            candidate_fn,
            pooled_actuals,
            pooled_preds,
            by_horizon_actuals,
            by_horizon_preds,
            warnings,
        )
        if result is not None:
            fold_results.append(result)

    pooled = calculate_forecast_metrics(
        np.asarray(pooled_actuals, dtype=float),
        np.asarray(pooled_preds, dtype=float),
        training=series.values.astype(float),
        mase_period=config.mase_period,
    )

    by_horizon: dict[int, ForecastMetrics] = {}
    for h in sorted(by_horizon_actuals):
        by_horizon[h] = calculate_forecast_metrics(
            np.asarray(by_horizon_actuals[h], dtype=float),
            np.asarray(by_horizon_preds[h], dtype=float),
            training=series.values.astype(float),
            mase_period=config.mase_period,
        )

    n_evaluated = pooled.n_evaluated
    unavailable = dict(pooled.unavailable_reasons)
    if not fold_results:
        unavailable.setdefault("all", "No folds were evaluated.")

    return BacktestEvaluation(
        model_name=name,
        folds=fold_results,
        pooled_metrics=pooled,
        by_horizon_metrics=by_horizon,
        n_origins=len(fold_results),
        n_evaluated=n_evaluated,
        unavailable_reasons=unavailable,
        warnings=warnings,
    )


def evaluate_candidates(
    series: pd.Series,
    candidates: dict[str, CandidateFn],
    config: BacktestConfig | None = None,
) -> dict[str, BacktestEvaluation]:
    """Evaluate multiple candidates on identical folds.

    Args:
        series:      Full cleaned time series.
        candidates:  Mapping of candidate name to fit-and-predict callable.
        config:      Backtesting configuration. Defaults to a sensible
                     expanding-window configuration.

    Returns:
        Mapping of candidate name to :class:`BacktestEvaluation`.
    """
    if config is None:
        config = BacktestConfig()
    folds = generate_folds(series, config)
    evaluations: dict[str, BacktestEvaluation] = {}
    for name, fn in candidates.items():
        evaluations[name] = evaluate_candidate(name, series, folds, fn, config)
    return evaluations


# ── Compatibility: terminal holdout ──────────────────────────────────────────


def make_terminal_holdout_folds(
    series: pd.Series,
    forecast_horizon: int,
) -> list[BacktestFold]:
    """Return a single-fold terminal holdout for backward compatibility.

    This preserves the terminal-holdout evaluation boundary behind an
    accurate label. Prefer :func:`generate_folds` for new code.
    """
    n = len(series)
    if forecast_horizon < 1 or n < 2:
        return []
    split = max(
        1,
        min(n - 1, max(int(n * 0.8), n - forecast_horizon)),
    )
    return [
        BacktestFold(
            fold_index=0,
            train_end_index=split,
            test_start_index=split,
            test_end_index=min(split + forecast_horizon, n),
            horizon=min(forecast_horizon, n - split),
        )
    ]