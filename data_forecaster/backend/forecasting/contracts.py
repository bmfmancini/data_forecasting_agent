"""Typed contracts shared by forecast adapters and evaluation services."""

from __future__ import annotations

from enum import StrEnum
import math

from pydantic import BaseModel, Field


class ForecastFitStatus(StrEnum):
    """Outcome of fitting and evaluating a forecasting model."""

    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"
    NOT_ESTIMABLE = "not_estimable"


class ForecastMetrics(BaseModel):
    """Central forecast metrics and their evaluation metadata."""

    rmse: float | None = None
    mae: float | None = None
    mape: float | None = None
    wape: float | None = None
    mase: float | None = None
    smape: float | None = None
    rmsse: float | None = None
    n_evaluated: int = Field(default=0, ge=0)
    n_missing: int = Field(default=0, ge=0)
    unavailable_reasons: dict[str, str] = Field(default_factory=dict)


class ForecastAdapterResult(BaseModel):
    """Result emitted by every model adapter.

    Attributes:
        status:               Fit/evaluation outcome.
        forecast:             Point predictions for the production horizon.
        lower_ci:             Lower prediction-interval bounds.
        upper_ci:             Upper prediction-interval bounds.
        metrics:              Holdout/backtest evaluation metrics.
        fitted_configuration: Provenance of the fitted model (order, trend,
                              alpha, seasonal type, etc.).
        failure_reason:       Human-readable failure explanation (when not ok).
        is_fallback:          Whether this result is a fallback/persistence
                              forecast.
        warnings:             Adapter-specific warnings.
        innovations:          Fitted one-step-ahead innovations (residuals)
                              from the full-series fit. Used by residual
                              diagnostics. Empty when unavailable.
        interval_label:       Label for the prediction intervals —
                              ``"prediction_interval"`` (calibrated or
                              model-based) or ``"experimental"`` when
                              coverage cannot be evaluated.
    """

    status: ForecastFitStatus
    forecast: list[float] = Field(default_factory=list)
    lower_ci: list[float] = Field(default_factory=list)
    upper_ci: list[float] = Field(default_factory=list)
    metrics: ForecastMetrics = Field(default_factory=ForecastMetrics)
    fitted_configuration: dict[str, object] = Field(default_factory=dict)
    failure_reason: str | None = None
    is_fallback: bool = False
    warnings: list[str] = Field(default_factory=list)
    innovations: list[float] = Field(default_factory=list)
    interval_label: str = "prediction_interval"

    @property
    def is_rankable(self) -> bool:
        """Return whether this result has valid point-error evidence."""
        complete_evaluation = self.metrics.n_missing == 0
        return (
            self.status == ForecastFitStatus.OK
            and complete_evaluation
            and all(
                value is not None and math.isfinite(value)
                for value in (self.metrics.rmse, self.metrics.mae)
            )
        )


# ── Rolling-origin backtesting contracts ─────────────────────────────────────


class BacktestFold(BaseModel):
    """One auditable rolling-origin fold.

    Attributes:
        fold_index:        Zero-based fold ordinal.
        train_end_index:   Exclusive end index of the training window.
        test_start_index:  Inclusive start index of the test window.
        test_end_index:    Exclusive end index of the test window.
        horizon:           Number of periods forecast in this fold.
    """

    fold_index: int
    train_end_index: int
    test_start_index: int
    test_end_index: int
    horizon: int


class BacktestFoldResult(BaseModel):
    """Per-fold predictions and errors for one candidate model.

    Attributes:
        fold:           The fold boundary definition.
        predictions:    Point predictions aligned to the fold test window.
        lower_ci:       Lower prediction-interval bounds (when available).
        upper_ci:       Upper prediction-interval bounds (when available).
        residuals:      Actuals minus predictions for the fold test window.
        status:         Fit status for this fold.
        warnings:       Fold-specific warnings (e.g. short window).
        fitted_configuration: Configuration used to fit this fold.
    """

    fold: BacktestFold
    predictions: list[float] = Field(default_factory=list)
    lower_ci: list[float] = Field(default_factory=list)
    upper_ci: list[float] = Field(default_factory=list)
    residuals: list[float] = Field(default_factory=list)
    status: ForecastFitStatus = ForecastFitStatus.OK
    warnings: list[str] = Field(default_factory=list)
    fitted_configuration: dict[str, object] = Field(default_factory=dict)


class BacktestEvaluation(BaseModel):
    """Aggregate rolling-origin evaluation for one candidate model.

    Attributes:
        model_name:   Name of the evaluated candidate.
        folds:         Per-fold results.
        pooled_metrics: Metrics pooled across all fold test windows.
        by_horizon_metrics: Optional metrics keyed by horizon step.
        n_origins:    Number of rolling origins evaluated.
        n_evaluated:  Total number of aligned observations scored.
        unavailable_reasons: Reasons any metric is unavailable.
        warnings:     Cross-fold warnings.
    """

    model_name: str
    folds: list[BacktestFoldResult] = Field(default_factory=list)
    pooled_metrics: ForecastMetrics = Field(default_factory=ForecastMetrics)
    final_test_metrics: ForecastMetrics = Field(default_factory=ForecastMetrics)
    by_horizon_metrics: dict[int, ForecastMetrics] = Field(default_factory=dict)
    n_origins: int = 0
    n_failed_origins: int = 0
    n_evaluated: int = 0
    validation_design: dict[str, object] = Field(default_factory=dict)
    metric_intervals: dict[str, list[float]] = Field(default_factory=dict)
    skill_scores: dict[str, float] = Field(default_factory=dict)
    unavailable_reasons: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @property
    def is_rankable(self) -> bool:
        """Return whether pooled evidence supports ranking."""
        rmse = self.pooled_metrics.rmse
        return (
            self.n_origins > 0
            and self.pooled_metrics.n_missing == 0
            and rmse is not None
            and math.isfinite(rmse)
        )


# ── Residual diagnostics contracts ───────────────────────────────────────────


class ResidualDiagnosticsResult(BaseModel):
    """Typed residual diagnostics for one fitted model.

    Distinguishes fitted innovations from pooled backtest errors. The
    ``error_type`` field records which kind of error was analysed.

    Attributes:
        error_type:           ``"innovations"`` or ``"backtest_errors"``.
        n_errors:             Number of errors analysed.
        mean:                 Mean error (bias estimate).
        mean_ci_lower:        95% CI lower bound for the mean error.
        mean_ci_upper:        95% CI upper bound for the mean error.
        is_zero_mean:         Whether the mean is statistically indistinguishable
                              from zero at the 0.05 level.
        ljung_box_p_value:    p-value of the Ljung-Box test.
        ljung_box_lag:        Lag used for the Ljung-Box test.
        ljung_box_df_adjust:  Degrees-of-freedom adjustment applied for
                              ARIMA-family innovations (fitted AR+MA order).
        is_uncorrelated:      Whether residuals show no significant
                              autocorrelation at the 0.05 level.
        shapiro_p_value:      p-value of the Shapiro-Wilk normality test.
        is_normal:            Whether residuals are consistent with normality.
        variance_by_horizon:  Variance of backtest errors keyed by horizon
                              step (empty for innovations).
        interval_coverage:    Empirical coverage of prediction intervals
                              (fraction of actuals inside the interval).
        interval_mean_width:  Average width of prediction intervals.
        winkler_score:        Mean Winkler interval score at the nominal level.
        nominal_coverage:     Nominal coverage level (e.g. 0.95).
        coverage_estimable:   Whether coverage could be estimated from data.
        warnings:             Diagnostics-specific warnings.
    """

    error_type: str = "innovations"
    n_errors: int = 0
    mean: float = 0.0
    mean_ci_lower: float | None = None
    mean_ci_upper: float | None = None
    is_zero_mean: bool | None = None
    ljung_box_p_value: float | None = None
    ljung_box_lag: int | None = None
    ljung_box_df_adjust: int = 0
    is_uncorrelated: bool | None = None
    shapiro_p_value: float | None = None
    is_normal: bool | None = None
    variance_by_horizon: dict[int, float] = Field(default_factory=dict)
    interval_coverage: float | None = None
    interval_mean_width: float | None = None
    winkler_score: float | None = None
    weighted_interval_score: float | None = None
    interval_coverage_by_horizon: dict[int, float] = Field(default_factory=dict)
    interval_width_by_horizon: dict[int, float] = Field(default_factory=dict)
    winkler_score_by_horizon: dict[int, float] = Field(default_factory=dict)
    nominal_coverage: float = 0.95
    coverage_estimable: bool = False
    warnings: list[str] = Field(default_factory=list)


# ── Evidence-state contracts for statistical diagnostics ────────────────────


class DiagnosticStatus(StrEnum):
    """Outcome of a single statistical diagnostic.

    Every diagnostic returns one of these statuses so callers can
    distinguish real evidence from assumptions, disabled tests, and
    failures.

    Attributes:
        OK:            The diagnostic ran and produced valid evidence.
        NOT_ESTIMABLE: The series is too short or otherwise unsuitable for
                       the diagnostic; no evidence is available.
        DISABLED:      The user explicitly disabled this diagnostic.
        FAILED:        The diagnostic raised an exception; no evidence.
    """

    OK = "ok"
    NOT_ESTIMABLE = "not_estimable"
    DISABLED = "disabled"
    FAILED = "failed"


class SeasonalityEvidence(BaseModel):
    """Typed evidence for seasonality detection.

    Replaces the single ``seasonal_period`` int with a structured record
    that distinguishes observed frequency, candidate periods, data-derived
    evidence, and the selected model period with its provenance.

    Attributes:
        status:               Diagnostic outcome.
        observed_frequency:   Pandas-inferred frequency string (or None).
        frequency_period:     Period implied by the frequency (e.g. 12 for
                               monthly), or None when unknown.
        candidate_periods:    Data-derived candidate periods from the
                               periodogram, sorted by power descending.
        selected_period:       The period chosen for modelling (may be 1 for
                               no seasonality).
        selection_provenance:  How ``selected_period`` was chosen —
                               ``"frequency"``, ``"periodogram"``,
                               ``"metadata"``, or ``"default"``.
        seasonal_strength:    STL-based seasonal strength in [0, 1]; higher
                               is stronger. None when not estimable.
        dominant_period:      Strongest periodogram period (float), or None.
        warnings:              Diagnostic-specific warnings.
    """

    status: DiagnosticStatus = DiagnosticStatus.OK
    observed_frequency: str | None = None
    frequency_period: int | None = None
    candidate_periods: list[int] = Field(default_factory=list)
    selected_period: int = 1
    selection_provenance: str = "default"
    seasonal_strength: float | None = None
    dominant_period: float | None = None
    warnings: list[str] = Field(default_factory=list)


class StationarityEvidence(BaseModel):
    """Typed evidence for stationarity testing with a decision matrix.

    Combines ADF and KPSS results into a single classification:
    ``stationary``, ``trend_stationary``, ``difference_stationary``,
    ``conflicting``, or ``not_estimable``.

    Attributes:
        status:            Diagnostic outcome.
        adf_p_value:       ADF p-value (constant-only specification).
        adf_trend_p_value: ADF p-value (trend specification).
        kpss_p_value:      KPSS p-value (constant specification).
        kpss_trend_p_value: KPSS p-value (trend specification).
        classification:    One of ``"stationary"``, ``"trend_stationary"``,
                           ``"difference_stationary"``, ``"conflicting"``,
                           ``"not_estimable"``.
        is_stationary:     Convenience boolean — True only when
                           classification is ``"stationary"`` or
                           ``"trend_stationary"``.
        warnings:          Diagnostic-specific warnings.
    """

    status: DiagnosticStatus = DiagnosticStatus.OK
    adf_statistic: float | None = None
    adf_p_value: float | None = None
    adf_trend_statistic: float | None = None
    adf_trend_p_value: float | None = None
    kpss_statistic: float | None = None
    kpss_p_value: float | None = None
    kpss_trend_statistic: float | None = None
    kpss_trend_p_value: float | None = None
    classification: str = "not_estimable"
    is_stationary: bool = False
    warnings: list[str] = Field(default_factory=list)


class AnomalyEvidence(BaseModel):
    """Typed evidence for anomaly detection on adjusted residuals.

    Anomalies are detected on detrended/seasonally-adjusted residuals using
    a robust MAD/Hampel-style rule rather than raw-value IQR/z-score.

    Attributes:
        status:        Diagnostic outcome.
        anomaly_count:  Number of anomalies detected.
        anomaly_ratio:  Fraction of observations flagged as anomalies.
        anomaly_indices: Integer positions of flagged anomalies.
        method:         Detection method label (e.g. ``"mad_hampel"``).
        threshold:      Threshold used for detection (in MAD units).
        warnings:       Diagnostic-specific warnings.
    """

    status: DiagnosticStatus = DiagnosticStatus.OK
    anomaly_count: int = 0
    anomaly_ratio: float = 0.0
    anomaly_indices: list[int] = Field(default_factory=list)
    method: str = "mad_hampel"
    threshold: float = 3.5
    warnings: list[str] = Field(default_factory=list)
    classifications: dict[str, list[int]] = Field(default_factory=dict)


class ChangePointEvidence(BaseModel):
    """Typed evidence for calibrated change-point detection.

    Replaces the uncalibrated CUSUM threshold-crossing list with a
    calibrated binary-segmentation change-point method and minimum
    segment/spacing rules. Variance breaks are analyzed separately.

    Attributes:
        status:            Diagnostic outcome.
        change_points:     Integer positions of detected change points.
        n_change_points:   Number of change points.
        method:            Detection method label (e.g. ``"binary_segmentation"``).
        min_segment:       Minimum segment length enforced.
        variance_breaks:   Integer positions of detected variance breaks.
        warnings:          Diagnostic-specific warnings.
    """

    status: DiagnosticStatus = DiagnosticStatus.OK
    change_points: list[int] = Field(default_factory=list)
    n_change_points: int = 0
    method: str = "binary_segmentation"
    min_segment: int = 5
    variance_breaks: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TrendEvidence(BaseModel):
    """Typed evidence for trend detection with effect size.

    Replaces iid OLS trend significance with effect size plus
    autocorrelation-robust inference.

    Attributes:
        status:        Diagnostic outcome.
        has_trend:     Whether a statistically significant trend was detected.
        slope:         Estimated slope (units per period).
        effect_size:   R-squared effect size (fraction of variance explained).
        p_value:       p-value from autocorrelation-robust inference.
        warnings:      Diagnostic-specific warnings.
    """

    status: DiagnosticStatus = DiagnosticStatus.OK
    has_trend: bool = False
    slope: float = 0.0
    effect_size: float = 0.0
    p_value: float | None = None
    warnings: list[str] = Field(default_factory=list)


class PreprocessingTransform(BaseModel):
    """A fold-safe preprocessing transformation with inverse support.

    Transformations are fitted on training data only and can be inverted
    on predictions to return to the original scale.

    Attributes:
        name:           Transform label (e.g. ``"boxcox"``, ``"log"``).
        lambda_value:   Box-Cox lambda (when applicable).
        shift:          Additive shift applied before transformation.
        is_fitted:      Whether the transform has been fitted.
    """

    name: str = "none"
    lambda_value: float | None = None
    shift: float = 0.0
    is_fitted: bool = False
