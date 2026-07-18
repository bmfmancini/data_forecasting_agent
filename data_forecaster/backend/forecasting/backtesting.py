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
from time import perf_counter

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
from forecasting.preprocessing import prepare_training_series

logger = get_logger(__name__)


def _bootstrap_metric_intervals(
    actual: np.ndarray,
    predicted: np.ndarray,
    training: np.ndarray,
    mase_period: int,
    *,
    repetitions: int = 500,
) -> dict[str, list[float]]:
    """Return deterministic percentile intervals for aggregate metrics."""
    if actual.size < 3:
        return {}
    rng = np.random.default_rng(42)
    samples: dict[str, list[float]] = {name: [] for name in ("rmse", "mae", "mase")}
    for _ in range(repetitions):
        indices = rng.integers(0, actual.size, actual.size)
        metrics = calculate_forecast_metrics(
            actual[indices],
            predicted[indices],
            training=training,
            mase_period=mase_period,
        )
        for name in samples:
            value = getattr(metrics, name)
            if value is not None and np.isfinite(value):
                samples[name].append(float(value))
    return {
        name: np.quantile(values, [0.025, 0.975]).astype(float).tolist()
        for name, values in samples.items()
        if values
    }


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
        mase_period:        Naive lag used for MASE scale estimation.
        requested_horizon:  Original production horizon before a transparent
            runtime/data-support reduction.
    """

    initial_train_size: int | None = None
    horizon: int | None = None
    step_size: int | None = None
    max_origins: int | None = None
    gap: int = 0
    mase_period: int = 1
    requested_horizon: int | None = None
    apply_iqr_clip: bool = False
    outlier_strategy: str = "none"
    imputation_method: str = "interpolate"
    smoothing_method: str = "none"
    final_test_size: int = 0


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

    reserve = max(0, min(config.final_test_size, max(0, n - 2)))
    end_limit = n - reserve

    initial = config.initial_train_size
    if initial is None:
        initial = max(10, end_limit // 2)
    initial = max(1, min(initial, end_limit - horizon - config.gap))

    step = config.step_size or horizon
    step = max(1, step)

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
    config: BacktestConfig,
) -> BacktestFoldResult | None:
    """Process one fold for a candidate; return the fold result or ``None``.

    ``None`` indicates the fold was skipped (insufficient data). Pooled and
    by-horizon accumulators are updated in place when predictions succeed.
    """
    train = series.iloc[: fold.train_end_index].copy()
    strategy = "clip" if config.apply_iqr_clip else config.outlier_strategy
    train = prepare_training_series(
        train,
        outlier_strategy=strategy,
        imputation_method=config.imputation_method,
        smoothing_method=config.smoothing_method,
        apply_iqr_clip=config.apply_iqr_clip,
    )
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

    if result.status != ForecastFitStatus.OK:
        return BacktestFoldResult(
            fold=fold,
            status=result.status,
            warnings=list(result.warnings or []),
            fitted_configuration=dict(result.fitted_configuration or {}),
        )

    preds = np.asarray(result.predictions, dtype=float)
    actuals = test.values.astype(float)
    if preds.shape[0] != actuals.shape[0]:
        warning = (
            f"Fold {fold.fold_index} prediction length mismatch "
            f"({preds.shape[0]} vs {actuals.shape[0]})."
        )
        warnings.append(warning)
        return BacktestFoldResult(
            fold=fold,
            status=ForecastFitStatus.FAILED,
            warnings=[warning],
            fitted_configuration=dict(result.fitted_configuration or {}),
        )

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
    for h in range(1, len(actuals) + 1):
        by_horizon_actuals.setdefault(h, []).append(float(actuals[h - 1]))
        by_horizon_preds.setdefault(h, []).append(float(preds[h - 1]))
    return fold_result


def evaluate_candidate(
    name: str,
    series: pd.Series,
    folds: Sequence[BacktestFold],
    candidate_fn: CandidateFn,
    config: BacktestConfig,
    activity_callback: Callable[[str], None] | None = None,
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

    for fold_number, fold in enumerate(folds, start=1):
        if activity_callback is not None:
            activity_callback(
                f"Evaluating {name}: validation fold {fold_number} of {len(folds)}…"
            )
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
            config,
        )
        if result is not None:
            fold_results.append(result)

    if folds:
        strategy = "clip" if config.apply_iqr_clip else config.outlier_strategy
        initial_series = prepare_training_series(
            series.iloc[: folds[0].train_end_index].copy(),
            outlier_strategy=strategy,
            imputation_method=config.imputation_method,
            smoothing_method=config.smoothing_method,
        )
        initial_training = initial_series.to_numpy(dtype=float)
    else:
        initial_training = np.asarray([], dtype=float)
    pooled = calculate_forecast_metrics(
        np.asarray(pooled_actuals, dtype=float),
        np.asarray(pooled_preds, dtype=float),
        training=initial_training,
        mase_period=config.mase_period,
    )

    by_horizon: dict[int, ForecastMetrics] = {}
    for h in sorted(by_horizon_actuals):
        by_horizon[h] = calculate_forecast_metrics(
            np.asarray(by_horizon_actuals[h], dtype=float),
            np.asarray(by_horizon_preds[h], dtype=float),
            training=initial_training,
            mase_period=config.mase_period,
        )

    n_evaluated = pooled.n_evaluated
    unavailable = dict(pooled.unavailable_reasons)
    if not fold_results:
        unavailable.setdefault("all", "No folds were evaluated.")

    successful_origins = sum(
        fold.status == ForecastFitStatus.OK for fold in fold_results
    )
    final_test_metrics = ForecastMetrics(
        unavailable_reasons={"all": "No untouched final test window was reserved."}
    )
    final_test_fitted_configuration: dict[str, object] = {}
    # Reuse the same clamped final_test_size as generate_folds so at least
    # two training observations are preserved.
    final_test_size = max(0, min(config.final_test_size, max(0, len(series) - 2)))
    if final_test_size > 0 and len(series) > final_test_size:
        if activity_callback is not None:
            activity_callback(f"Evaluating {name}: final holdout test…")
        final_start = len(series) - final_test_size
        final_fold = BacktestFold(
            fold_index=len(folds),
            train_end_index=final_start,
            test_start_index=final_start,
            test_end_index=len(series),
            horizon=final_test_size,
        )
        final_actuals: list[float] = []
        final_predictions: list[float] = []
        final_result = _process_fold(
            name,
            series,
            final_fold,
            candidate_fn,
            final_actuals,
            final_predictions,
            {},
            {},
            warnings,
            config,
        )
        if final_result is not None and final_result.status == ForecastFitStatus.OK:
            final_test_fitted_configuration = dict(final_result.fitted_configuration)
        if final_result is not None and final_result.status == ForecastFitStatus.OK:
            strategy = "clip" if config.apply_iqr_clip else config.outlier_strategy
            final_training = prepare_training_series(
                series.iloc[:final_start].copy(),
                outlier_strategy=strategy,
                imputation_method=config.imputation_method,
                smoothing_method=config.smoothing_method,
            )
            final_test_metrics = calculate_forecast_metrics(
                np.asarray(final_actuals, dtype=float),
                np.asarray(final_predictions, dtype=float),
                training=final_training,
                mase_period=config.mase_period,
            )
        else:
            final_test_metrics = ForecastMetrics(
                unavailable_reasons={
                    "all": "Candidate failed on the untouched final test window."
                }
            )
    evaluated_horizon = folds[0].horizon if folds else 0
    requested_horizon = config.requested_horizon or config.horizon or evaluated_horizon
    return BacktestEvaluation(
        model_name=name,
        folds=fold_results,
        pooled_metrics=pooled,
        final_test_metrics=final_test_metrics,
        final_test_fitted_configuration=final_test_fitted_configuration,
        by_horizon_metrics=by_horizon,
        n_origins=successful_origins,
        n_failed_origins=len(fold_results) - successful_origins,
        n_evaluated=n_evaluated,
        validation_design={
            "method": "expanding_window",
            "initial_train_size": folds[0].train_end_index if folds else 0,
            "requested_horizon": requested_horizon,
            "evaluated_horizon": evaluated_horizon,
            "unsupported_horizons": list(
                range(evaluated_horizon + 1, requested_horizon + 1)
            ),
            "step_size": config.step_size or evaluated_horizon,
            "gap": config.gap,
            "max_origins": config.max_origins,
            "successful_origins": successful_origins,
            "failed_origins": len(fold_results) - successful_origins,
            "n_evaluated": n_evaluated,
            "mase_period": config.mase_period,
            "apply_iqr_clip": config.apply_iqr_clip,
            "outlier_strategy": config.outlier_strategy,
            "imputation_method": config.imputation_method,
            "smoothing_method": config.smoothing_method,
            "final_test_size": final_test_size,
            "selection_end_index": len(series) - final_test_size,
        },
        metric_intervals=_bootstrap_metric_intervals(
            np.asarray(pooled_actuals, dtype=float),
            np.asarray(pooled_preds, dtype=float),
            initial_training,
            config.mase_period,
        ),
        unavailable_reasons=unavailable,
        warnings=warnings,
    )


def evaluate_candidates(
    series: pd.Series,
    candidates: dict[str, CandidateFn],
    config: BacktestConfig | None = None,
    activity_callback: Callable[[str], None] | None = None,
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
    candidate_total = len(candidates)
    for candidate_number, (name, fn) in enumerate(candidates.items(), start=1):
        if activity_callback is not None:
            activity_callback(
                f"Evaluating {name} candidate {candidate_number} of {candidate_total}…"
            )
        started = perf_counter()
        evaluations[name] = evaluate_candidate(
            name,
            series,
            folds,
            fn,
            config,
            activity_callback=activity_callback,
        )
        logger.info(
            "Performance candidate_evaluation model=%s wall_seconds=%.3f "
            "origins=%d failed_origins=%d",
            name,
            perf_counter() - started,
            evaluations[name].n_origins,
            evaluations[name].n_failed_origins,
        )
    references = [
        evaluations[name]
        for name in ("Seasonal Naive", "Naive")
        if name in evaluations and evaluations[name].is_rankable
    ]
    if references:
        reference = min(
            references,
            key=lambda item: item.pooled_metrics.mae or float("inf"),
        )
        for name, evaluation in list(evaluations.items()):
            skill: dict[str, float] = {}
            for metric in ("mae", "rmse"):
                candidate_value = getattr(evaluation.pooled_metrics, metric)
                reference_value = getattr(reference.pooled_metrics, metric)
                if candidate_value is not None and reference_value not in (None, 0):
                    skill[f"{metric}_skill_vs_{reference.model_name}"] = float(
                        1.0 - candidate_value / reference_value
                    )
            evaluations[name] = evaluation.model_copy(update={"skill_scores": skill})
    return evaluations
