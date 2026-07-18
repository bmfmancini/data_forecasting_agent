"""Evidence-based statistical diagnostics for the forecasting pipeline.

Replaces assumed/overinterpreted diagnostics with explicit evidence states
and fold-safe transformations. Every diagnostic returns a typed contract from
:mod:`forecasting.contracts` with a ``DiagnosticStatus`` so callers can
distinguish real evidence from assumptions, disabled tests, and failures.

The functions here are pure-Python and do not depend on LLM availability.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import periodogram as scipy_periodogram
from scipy.stats import linregress
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from core.logging_config import get_logger
from forecasting.contracts import (
    AnomalyEvidence,
    ChangePointEvidence,
    DiagnosticStatus,
    SeasonalityEvidence,
    StationarityEvidence,
    TrendEvidence,
)
from utils.data_cleaning import frequency_to_seasonal_period

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_SIGNIFICANCE_LEVEL = 0.05
_MIN_STL_CYCLES = 2
_MIN_PERIODOGRAM_LENGTH = 10
_MIN_STATIONARITY_LENGTH = 10
_MIN_CHANGEPOINT_LENGTH = 20
_DEFAULT_MIN_SEGMENT = 5
_MAD_THRESHOLD = 3.5  # Hampel identifier threshold (in MAD units)
_MAX_CANDIDATE_PERIODS = 5
_HARMONIC_TOLERANCE = 0.15  # 15% tolerance for harmonic matching


# ── Frequency → period mapping ──────────────────────────────────────────────

def _freq_to_period(freq: str | None) -> int | None:
    """Map a pandas frequency string to an integer seasonal period.

    Args:
        freq: Pandas frequency alias (e.g. ``"MS"``, ``"W-SUN"``).

    Returns:
        The integer period, or ``None`` when the frequency is unknown.
    """
    return frequency_to_seasonal_period(freq, default=None)


# ── Seasonality ──────────────────────────────────────────────────────────────


def detect_seasonality(
    series: pd.Series,
    *,
    metadata_period: int | None = None,
    disabled: bool = False,
) -> SeasonalityEvidence:
    """Detect seasonality with explicit evidence states.

    Combines frequency-implied candidate periods with data-derived
    periodogram evidence and robust STL seasonal strength. Harmonics are
    accounted for rather than treating the largest periodogram peak as
    definitive.

    Args:
        series:          Cleaned historical series.
        metadata_period:  Period supplied by metadata/preflight (may be 12
                          for monthly data). Used as a candidate prior.
        disabled:        When True, returns a DISABLED status.

    Returns:
        :class:`SeasonalityEvidence` with the selected period and provenance.
    """
    if disabled:
        return SeasonalityEvidence(
            status=DiagnosticStatus.DISABLED,
            warnings=["Seasonality detection disabled by user."],
        )

    values = series.dropna().astype(float)
    n = len(values)
    if n < _MIN_PERIODOGRAM_LENGTH:
        return SeasonalityEvidence(
            status=DiagnosticStatus.NOT_ESTIMABLE,
            warnings=[f"Series too short for seasonality detection (n={n})."],
        )

    observed_freq = None
    if isinstance(series.index, pd.DatetimeIndex):
        try:
            observed_freq = pd.infer_freq(series.index)
        except Exception:  # pylint: disable=broad-except
            observed_freq = None
    freq_period = _freq_to_period(observed_freq)

    # ── Periodogram on detrended values ──────────────────────────────────────
    detrended = _detrend(values)
    candidate_periods = _periodogram_candidates(detrended)

    # ── STL seasonal strength ─────────────────────────────────────────────────
    seasonal_strength: float | None = None
    stl_period = _select_stl_period(freq_period, metadata_period, candidate_periods, n)
    if stl_period is not None and n >= _MIN_STL_CYCLES * stl_period:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stl = STL(values, period=stl_period, robust=True).fit()
            seasonal_strength = _compute_seasonal_strength(stl)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("STL seasonal strength failed: %s", exc)

    # ── Select the model period ───────────────────────────────────────────────
    selected_period, provenance = _select_seasonal_period(
        freq_period=freq_period,
        metadata_period=metadata_period,
        candidate_periods=candidate_periods,
        seasonal_strength=seasonal_strength,
        stl_period=stl_period,
    )

    dominant_period = candidate_periods[0] if candidate_periods else None
    warnings_list: list[str] = []
    if seasonal_strength is not None and seasonal_strength < 0.1:
        warnings_list.append(
            f"Seasonal strength is low ({seasonal_strength:.3f}); "
            "seasonality may be negligible."
        )
    if not candidate_periods:
        warnings_list.append("No data-derived candidate periods found.")

    return SeasonalityEvidence(
        status=DiagnosticStatus.OK,
        observed_frequency=observed_freq,
        frequency_period=freq_period,
        candidate_periods=candidate_periods[:_MAX_CANDIDATE_PERIODS],
        selected_period=selected_period,
        selection_provenance=provenance,
        seasonal_strength=seasonal_strength,
        dominant_period=float(dominant_period) if dominant_period else None,
        warnings=warnings_list,
    )


def _detrend(values: pd.Series) -> pd.Series:
    """Remove a linear trend from the series before spectral analysis.

    Args:
        values: Cleaned numeric series.

    Returns:
        Detrended series (residuals from OLS on time index).
    """
    x = np.arange(len(values), dtype=float)
    if len(values) < 3:
        return values - values.mean()
    try:
        slope, intercept, _, _, _ = linregress(x, values.values)
        trend = slope * x + intercept
        return pd.Series(values.values - trend, index=values.index)
    except Exception:  # pylint: disable=broad-except
        return values - values.mean()


def _periodogram_candidates(detrended: pd.Series) -> list[int]:
    """Extract candidate seasonal periods from the periodogram.

    Returns integer periods sorted by spectral power descending. Harmonics
    are grouped so that a fundamental and its multiples do not both appear
    unless they are independently strong.

    Args:
        detrended: Detrended numeric series.

    Returns:
        List of integer candidate periods.
    """
    values = detrended.dropna().astype(float).values
    n = len(values)
    if n < _MIN_PERIODOGRAM_LENGTH:
        return []
    try:
        freqs, power = scipy_periodogram(values, detrend=False)
    except Exception:  # pylint: disable=broad-except
        return []

    # Skip DC component
    if len(freqs) <= 1:
        return []
    freqs = freqs[1:]
    power = power[1:]

    # Sort by power descending
    order = np.argsort(power)[::-1]
    candidates: list[int] = []
    for idx in order:
        freq = freqs[idx]
        if freq <= 0:
            continue
        period = 1.0 / freq  # scipy frequencies are cycles per observation
        if period < 2 or period > n / 2:
            continue
        int_period = int(round(period))
        if int_period < 2:
            continue
        # Check if this is a harmonic of an already-selected candidate
        if _is_harmonic_of_existing(int_period, candidates):
            continue
        candidates.append(int_period)
        if len(candidates) >= _MAX_CANDIDATE_PERIODS:
            break
    return candidates


def _is_harmonic_of_existing(period: int, existing: list[int]) -> bool:
    """Check if ``period`` is a harmonic (multiple/divisor) of an existing candidate.

    Args:
        period:   Candidate period to check.
        existing: Already-selected candidate periods.

    Returns:
        True if ``period`` is a harmonic of any existing candidate.
    """
    for existing_period in existing:
        if existing_period <= 0 or period <= 0:
            continue
        ratio = max(period, existing_period) / min(period, existing_period)
        nearest = round(ratio)
        if nearest >= 2 and abs(ratio - nearest) < _HARMONIC_TOLERANCE:
            return True
    return False


def _select_stl_period(
    freq_period: int | None,
    metadata_period: int | None,
    candidate_periods: list[int],
    n: int,
) -> int | None:
    """Select a period for STL decomposition.

    Prefers frequency-derived period, then metadata, then the strongest
    data-derived candidate. Returns None when no period has enough data for
    at least two full cycles.

    Args:
        freq_period:       Period implied by the frequency.
        metadata_period:   Period supplied by metadata/preflight.
        candidate_periods: Data-derived candidate periods.
        n:                 Series length.

    Returns:
        Integer period for STL, or None.
    """
    for period in [freq_period, metadata_period]:
        if period and period >= 2 and n >= _MIN_STL_CYCLES * period:
            return period
    for period in candidate_periods:
        if period >= 2 and n >= _MIN_STL_CYCLES * period:
            return period
    return None


def _compute_seasonal_strength(stl_result: Any) -> float:
    """Compute STL-based seasonal strength in [0, 1].

    Seasonal strength = max(0, 1 - Var(residual) / Var(residual + seasonal)).

    Args:
        stl_result: Fitted STL result object.

    Returns:
        Seasonal strength float in [0, 1].
    """
    seasonal = np.asarray(stl_result.seasonal, dtype=float)
    resid = np.asarray(stl_result.resid, dtype=float)
    var_resid = float(np.var(resid))
    var_combined = float(np.var(resid + seasonal))
    if var_combined == 0:
        return 0.0
    strength = max(0.0, 1.0 - var_resid / var_combined)
    return min(1.0, strength)


def _select_seasonal_period(
    freq_period: int | None,
    metadata_period: int | None,
    candidate_periods: list[int],
    seasonal_strength: float | None,
    stl_period: int | None,
) -> tuple[int, str]:
    """Select the model period and record its provenance.

    Selection priority:
      1. Frequency-derived period (when STL strength supports it).
      2. Metadata period (when STL strength supports it).
      3. Strongest data-derived candidate (when STL strength supports it).
      4. Default period 1 (no seasonality).

    Args:
        freq_period:        Period implied by the frequency.
        metadata_period:    Period supplied by metadata/preflight.
        candidate_periods:  Data-derived candidate periods.
        seasonal_strength:  STL seasonal strength (or None).
        stl_period:         Period used for STL (or None).

    Returns:
        Tuple of (selected_period, provenance_string).
    """
    has_evidence = seasonal_strength is not None and seasonal_strength >= 0.1

    # When STL strength is available and weak, return no seasonality
    if seasonal_strength is not None and seasonal_strength < 0.1:
        return 1, "default"

    if freq_period and freq_period >= 2 and has_evidence:
        return freq_period, "frequency"
    if metadata_period and metadata_period >= 2 and has_evidence:
        return metadata_period, "metadata"
    if candidate_periods and has_evidence:
        return candidate_periods[0], "periodogram"
    if stl_period and has_evidence:
        return stl_period, "periodogram"
    return 1, "default"


# ── Stationarity ─────────────────────────────────────────────────────────────


def assess_stationarity(
    series: pd.Series,
    *,
    disabled: bool = False,
) -> StationarityEvidence:
    """Assess stationarity with ADF/KPSS constant and trend specifications.

    Combines ADF (constant and trend) and KPSS (constant and trend) into a
    decision matrix that can return ``stationary``, ``trend_stationary``,
    ``difference_stationary``, ``conflicting``, or ``not_estimable``.

    Args:
        series:   Cleaned historical series.
        disabled: When True, returns a DISABLED status.

    Returns:
        :class:`StationarityEvidence` with the classification.
    """
    if disabled:
        return StationarityEvidence(
            status=DiagnosticStatus.DISABLED,
            warnings=["Stationarity testing disabled by user."],
        )

    values = series.dropna().astype(float).values
    n = len(values)
    if n < _MIN_STATIONARITY_LENGTH:
        return StationarityEvidence(
            status=DiagnosticStatus.NOT_ESTIMABLE,
            warnings=[f"Series too short for stationarity testing (n={n})."],
        )

    try:
        adf_const_stat, adf_const_p = _run_adf(values, regression="c")
        adf_trend_stat, adf_trend_p = _run_adf(values, regression="ct")
        kpss_const_stat, kpss_const_p = _run_kpss(values, regression="c")
        kpss_trend_stat, kpss_trend_p = _run_kpss(values, regression="ct")
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Stationarity testing failed: %s", exc)
        return StationarityEvidence(
            status=DiagnosticStatus.FAILED,
            warnings=[f"Stationarity tests failed: {exc}"],
        )

    alpha = _SIGNIFICANCE_LEVEL
    adf_const_reject = adf_const_p is not None and adf_const_p < alpha
    adf_trend_reject = adf_trend_p is not None and adf_trend_p < alpha
    kpss_const_reject = kpss_const_p is not None and kpss_const_p < alpha
    kpss_trend_reject = kpss_trend_p is not None and kpss_trend_p < alpha

    classification = _classify_stationarity(
        adf_const_reject, adf_trend_reject, kpss_const_reject, kpss_trend_reject
    )
    is_stationary = classification in ("stationary", "trend_stationary")

    return StationarityEvidence(
        status=DiagnosticStatus.OK,
        adf_statistic=adf_const_stat,
        adf_p_value=adf_const_p,
        adf_trend_statistic=adf_trend_stat,
        adf_trend_p_value=adf_trend_p,
        kpss_statistic=kpss_const_stat,
        kpss_p_value=kpss_const_p,
        kpss_trend_statistic=kpss_trend_stat,
        kpss_trend_p_value=kpss_trend_p,
        classification=classification,
        is_stationary=is_stationary,
    )


def _run_adf(
    values: np.ndarray, regression: str = "c"
) -> tuple[float | None, float | None]:
    """Run the ADF test with the specified regression specification.

    Args:
        values:     Numeric array.
        regression:  ``"c"`` for constant, ``"ct"`` for constant+trend.

    Returns:
        p-value, or None on failure.
    """
    try:
        result = adfuller(values, regression=regression, autolag="AIC")
        return float(result[0]), float(result[1])
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("ADF (%s) failed: %s", regression, exc)
        return None, None


def _run_kpss(
    values: np.ndarray, regression: str = "c"
) -> tuple[float | None, float | None]:
    """Run the KPSS test with the specified regression specification.

    Args:
        values:     Numeric array.
        regression:  ``"c"`` for constant, ``"ct"`` for constant+trend.

    Returns:
        p-value, or None on failure.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = kpss(values, regression=regression, nlags="auto")
        return float(result[0]), float(result[1])
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("KPSS (%s) failed: %s", regression, exc)
        return None, None


def _classify_stationarity(
    adf_const_reject: bool,
    adf_trend_reject: bool,
    kpss_const_reject: bool,
    kpss_trend_reject: bool,
) -> str:
    """Classify stationarity from ADF/KPSS test outcomes.

    Decision matrix:
      - ADF rejects unit root (constant) + KPSS does not reject (constant)
        → ``"stationary"``
      - ADF rejects unit root (trend) + KPSS does not reject (trend) but
        KPSS rejects (constant) → ``"trend_stationary"``
      - ADF does not reject (constant) + KPSS rejects (constant)
        → ``"difference_stationary"``
      - ADF and KPSS disagree → ``"conflicting"``
      - Otherwise → ``"not_estimable"``

    Args:
        adf_const_reject:  ADF constant rejects unit root.
        adf_trend_reject:  ADF trend rejects unit root.
        kpss_const_reject: KPSS constant rejects stationarity.
        kpss_trend_reject: KPSS trend rejects stationarity.

    Returns:
        Classification string.
    """
    if adf_const_reject and not kpss_const_reject:
        return "stationary"
    if adf_trend_reject and not kpss_trend_reject and kpss_const_reject:
        return "trend_stationary"
    if not adf_const_reject and kpss_const_reject:
        return "difference_stationary"
    if adf_const_reject != (not kpss_const_reject):
        return "conflicting"
    return "not_estimable"


# ── Trend ────────────────────────────────────────────────────────────────────


def assess_trend(
    series: pd.Series,
    *,
    disabled: bool = False,
) -> TrendEvidence:
    """Assess trend with effect size and autocorrelation-robust inference.

    Uses OLS to estimate the slope and R-squared effect size, then applies
    a Newey-West HAC covariance to get an autocorrelation-robust p-value.

    Args:
        series:   Cleaned historical series.
        disabled: When True, returns a DISABLED status.

    Returns:
        :class:`TrendEvidence` with slope, effect size, and p-value.
    """
    if disabled:
        return TrendEvidence(
            status=DiagnosticStatus.DISABLED,
            warnings=["Trend detection disabled by user."],
        )

    values = series.dropna().astype(float)
    n = len(values)
    if n < 5:
        return TrendEvidence(
            status=DiagnosticStatus.NOT_ESTIMABLE,
            warnings=[f"Series too short for trend detection (n={n})."],
        )

    x = np.arange(n, dtype=float)
    y = values.values

    try:
        # OLS with Newey-West HAC standard errors
        x_with_const = np.column_stack([np.ones(n), x])
        model = OLS(y, x_with_const).fit(
            cov_type="HAC", cov_kwds={"maxlags": max(1, n // 5)}
        )
        slope = float(model.params[1])
        p_value = float(model.pvalues[1])
        effect_size = float(model.rsquared)
        has_trend = p_value < _SIGNIFICANCE_LEVEL and effect_size > 0.01
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Trend assessment failed: %s", exc)
        return TrendEvidence(
            status=DiagnosticStatus.FAILED,
            warnings=[f"Trend assessment failed: {exc}"],
        )

    return TrendEvidence(
        status=DiagnosticStatus.OK,
        has_trend=has_trend,
        slope=slope,
        effect_size=effect_size,
        p_value=p_value,
    )


# ── Anomalies ────────────────────────────────────────────────────────────────


def detect_anomalies(
    series: pd.Series,
    *,
    seasonal_period: int = 1,
    disabled: bool = False,
) -> AnomalyEvidence:
    """Detect anomalies on detrended/seasonally-adjusted residuals.

    Uses a robust MAD/Hampel-style rule on the residuals after removing
    trend and seasonal components. This distinguishes true anomalies from
    seasonal peaks.

    Args:
        series:          Cleaned historical series.
        seasonal_period: Seasonal period for decomposition (1 = no seasonality).
        disabled:        When True, returns a DISABLED status.

    Returns:
        :class:`AnomalyEvidence` with anomaly count, ratio, and indices.
    """
    if disabled:
        return AnomalyEvidence(
            status=DiagnosticStatus.DISABLED,
            warnings=["Anomaly detection disabled by user."],
        )

    values = series.dropna().astype(float)
    n = len(values)
    if n < 5:
        return AnomalyEvidence(
            status=DiagnosticStatus.NOT_ESTIMABLE,
            warnings=[f"Series too short for anomaly detection (n={n})."],
        )

    residuals = _compute_adjusted_residuals(values, seasonal_period)
    if residuals is None or len(residuals) == 0:
        return AnomalyEvidence(
            status=DiagnosticStatus.NOT_ESTIMABLE,
            warnings=["Could not compute adjusted residuals for anomaly detection."],
        )

    # MAD/Hampel identifier
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    # Scale MAD to approximate standard deviation
    mad_scaled = mad * 1.4826 if mad > 0 else 0.0
    if mad_scaled == 0:
        tolerance = np.finfo(float).eps * max(1.0, abs(median)) * 10.0
        anomaly_indices = [
            int(index)
            for index, value in enumerate(residuals)
            if abs(float(value) - median) > tolerance
        ]
        positive_indices = [
            index for index in anomaly_indices if residuals[index] > median
        ]
        negative_indices = [
            index for index in anomaly_indices if residuals[index] < median
        ]
        return AnomalyEvidence(
            status=DiagnosticStatus.OK,
            anomaly_count=len(anomaly_indices),
            anomaly_ratio=len(anomaly_indices) / n,
            anomaly_indices=anomaly_indices,
            method="mad_hampel",
            threshold=_MAD_THRESHOLD,
            warnings=[
                "MAD is zero; observations differing from the residual median "
                "were classified using numerical tolerance."
            ],
            classifications={
                "positive_spike": positive_indices,
                "negative_spike": negative_indices,
            },
        )

    deviations = np.abs(residuals - median) / mad_scaled
    anomaly_mask = deviations > _MAD_THRESHOLD
    anomaly_indices = [int(i) for i in np.nonzero(anomaly_mask)[0]]
    positive_indices = [index for index in anomaly_indices if residuals[index] > median]
    negative_indices = [index for index in anomaly_indices if residuals[index] < median]

    return AnomalyEvidence(
        status=DiagnosticStatus.OK,
        anomaly_count=len(anomaly_indices),
        anomaly_ratio=len(anomaly_indices) / n,
        anomaly_indices=anomaly_indices,
        method="mad_hampel",
        threshold=_MAD_THRESHOLD,
        classifications={
            "positive_spike": positive_indices,
            "negative_spike": negative_indices,
        },
    )


def _compute_adjusted_residuals(
    values: pd.Series,
    seasonal_period: int,
) -> np.ndarray | None:
    """Compute detrended/seasonally-adjusted residuals.

    Args:
        values:          Cleaned numeric series.
        seasonal_period: Seasonal period for STL (1 = no seasonality).

    Returns:
        Residual array, or None on failure.
    """
    n = len(values)
    try:
        if seasonal_period >= 2 and n >= _MIN_STL_CYCLES * seasonal_period:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stl = STL(values, period=seasonal_period, robust=True).fit()
            return np.asarray(stl.resid, dtype=float)
        # No seasonality — just detrend
        x = np.arange(n, dtype=float)
        slope, intercept, _, _, _ = linregress(x, values.values)
        trend = slope * x + intercept
        return values.values - trend
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Adjusted residual computation failed: %s", exc)
        return None


# ── Change points ────────────────────────────────────────────────────────────


def detect_change_points_calibrated(
    series: pd.Series,
    *,
    min_segment: int = _DEFAULT_MIN_SEGMENT,
    disabled: bool = False,
) -> ChangePointEvidence:
    """Detect change points using calibrated binary segmentation.

    Replaces the uncalibrated CUSUM threshold-crossing list with a
    calibrated binary-segmentation method and minimum segment/spacing rules.
    Variance breaks are analyzed separately.

    Args:
        series:       Cleaned historical series.
        min_segment:  Minimum segment length (enforced).
        disabled:     When True, returns a DISABLED status.

    Returns:
        :class:`ChangePointEvidence` with change points and variance breaks.
    """
    if disabled:
        return ChangePointEvidence(
            status=DiagnosticStatus.DISABLED,
            warnings=["Change-point detection disabled by user."],
        )

    values = series.dropna().astype(float).values
    n = len(values)
    if n < _MIN_CHANGEPOINT_LENGTH:
        return ChangePointEvidence(
            status=DiagnosticStatus.NOT_ESTIMABLE,
            min_segment=min_segment,
            warnings=[f"Series too short for change-point detection (n={n})."],
        )

    change_points = _binary_segmentation(values, min_segment, 0, n)
    change_points = _enforce_min_spacing(change_points, min_segment)
    variance_breaks = _detect_variance_breaks(values, min_segment)

    return ChangePointEvidence(
        status=DiagnosticStatus.OK,
        change_points=change_points,
        n_change_points=len(change_points),
        method="binary_segmentation",
        min_segment=min_segment,
        variance_breaks=variance_breaks,
    )


def _binary_segmentation(
    values: np.ndarray,
    min_segment: int,
    start: int,
    end: int,
) -> list[int]:
    """Recursive binary segmentation for mean-shift detection.

    Uses a CUSUM-based statistic with a permutation-derived threshold to
    avoid uncalibrated threshold crossing.

    Args:
        values:       Numeric array.
        min_segment:  Minimum segment length.
        start:        Start index (inclusive).
        end:          End index (exclusive).

    Returns:
        List of change-point indices (absolute positions).
    """
    segment = values[start:end]
    if len(segment) < 2 * min_segment:
        return []

    # CUSUM-based statistic
    cusum = np.cumsum(segment - segment.mean())
    max_stat = float(np.max(np.abs(cusum)))
    if max_stat == 0:
        return []

    # Permutation-based threshold calibration
    threshold = _calibrate_threshold(segment, n_permutations=100)
    if threshold is None or max_stat < threshold:
        return []

    # Find the split point (max CUSUM location)
    split_rel = int(np.argmax(np.abs(cusum)))
    split_abs = start + split_rel

    # Recurse on both sides
    left_cps = _binary_segmentation(values, min_segment, start, split_abs + 1)
    right_cps = _binary_segmentation(values, min_segment, split_abs + 1, end)
    return left_cps + [split_abs] + right_cps


def _calibrate_threshold(
    segment: np.ndarray,
    n_permutations: int = 100,
) -> float | None:
    """Calibrate a CUSUM threshold via permutation.

    Generates ``n_permutations`` random permutations of the segment and
    computes the max CUSUM statistic for each. The threshold is the 95th
    percentile of the permutation distribution.

    Args:
        segment:        Numeric array.
        n_permutations: Number of permutations for calibration.

    Returns:
        Calibrated threshold, or None on failure.
    """
    if len(segment) < 4:
        return None
    rng = np.random.default_rng(42)
    stats = np.empty(n_permutations)
    for i in range(n_permutations):
        permuted = rng.permutation(segment)
        cusum = np.cumsum(permuted - permuted.mean())
        stats[i] = np.max(np.abs(cusum))
    return float(np.percentile(stats, 95))


def _enforce_min_spacing(
    change_points: list[int],
    min_segment: int,
) -> list[int]:
    """Enforce minimum spacing between change points.

    Args:
        change_points: Raw change-point indices.
        min_segment:   Minimum spacing.

    Returns:
        Filtered change-point list.
    """
    if not change_points:
        return []
    sorted_cps = sorted(change_points)
    filtered = [sorted_cps[0]]
    for cp in sorted_cps[1:]:
        if cp - filtered[-1] >= min_segment:
            filtered.append(cp)
    return filtered


def _detect_variance_breaks(
    values: np.ndarray,
    min_segment: int,
) -> list[int]:
    """Detect variance breaks using rolling variance comparison.

    Args:
        values:       Numeric array.
        min_segment:  Minimum segment length.

    Returns:
        List of variance-break indices.
    """
    n = len(values)
    if n < 3 * min_segment:
        return []
    window = max(min_segment, n // 10)
    rolling_var = pd.Series(values).rolling(window=window, center=True).var().values
    valid_var = rolling_var[np.isfinite(rolling_var)]
    if len(valid_var) == 0:
        return []
    mean_var = float(np.mean(valid_var))
    std_var = float(np.std(valid_var))
    if std_var == 0:
        return []
    # Flag points where variance exceeds mean + 3*std
    threshold = mean_var + 3 * std_var
    breaks = [
        int(i)
        for i in range(n)
        if np.isfinite(rolling_var[i]) and rolling_var[i] > threshold
    ]
    return _enforce_min_spacing(breaks, min_segment)


# ── White noise test (re-exported for convenience) ───────────────────────────


def test_white_noise(series: pd.Series, lags: int = 10) -> dict[str, Any]:
    """Run the Ljung-Box test for white noise.

    Args:
        series: Time series to test.
        lags:   Number of lags for the test.

    Returns:
        Dict with ``p_value``, ``is_white_noise``, and ``interpretation``.
    """
    values = series.dropna()
    n = len(values)
    actual_lags = min(lags, max(1, n // 5))
    try:
        res = acorr_ljungbox(values, lags=[actual_lags], return_df=True)
        p_value = float(res.lb_pvalue.iloc[0])
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("White noise test failed: %s", exc)
        return {
            "p_value": 1.0,
            "is_white_noise": False,
            "interpretation": "White noise test failed.",
        }
    is_white_noise = p_value > _SIGNIFICANCE_LEVEL
    interpretation = (
        f"Ljung-Box p-value: {p_value:.4f}. "
        f"{'Series is white noise (random).' if is_white_noise else 'Series contains significant signal.'}"
    )
    return {
        "p_value": p_value,
        "is_white_noise": is_white_noise,
        "interpretation": interpretation,
    }


def assess_arch_effects(series: pd.Series, lags: int = 5) -> dict[str, object]:
    """Test adjusted residuals for conditional heteroskedasticity."""
    from statsmodels.stats.diagnostic import het_arch

    residuals = _compute_adjusted_residuals(series.dropna().astype(float), 1)
    if residuals is None or len(residuals) < max(12, 2 * lags + 1):
        return {"status": "not_estimable", "p_value": None, "has_arch": None}
    try:
        _, p_value, _, _ = het_arch(np.asarray(residuals, dtype=float), nlags=lags)
        return {
            "status": "ok",
            "p_value": float(p_value),
            "has_arch": bool(p_value < 0.05),
        }
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "status": "failed",
            "p_value": None,
            "has_arch": None,
            "warning": str(exc),
        }


def assess_sen_trend(series: pd.Series) -> dict[str, object]:
    """Return Kendall monotonic-trend evidence and a robust Sen slope."""
    from scipy.stats import kendalltau, theilslopes

    values = series.dropna().astype(float).to_numpy()
    if values.size < 8:
        return {"status": "not_estimable", "p_value": None, "sen_slope": None}
    time = np.arange(values.size, dtype=float)
    tau, p_value = kendalltau(time, values)
    slope, _, lower, upper = theilslopes(values, time, alpha=0.95)
    return {
        "status": "ok",
        "kendall_tau": float(tau),
        "p_value": float(p_value),
        "sen_slope": float(slope),
        "sen_slope_ci": [float(lower), float(upper)],
    }


def assess_intermittency(series: pd.Series) -> dict[str, object]:
    """Characterize zero-heavy nonnegative demand."""
    values = series.dropna().astype(float).to_numpy()
    if values.size == 0 or np.any(values < 0):
        return {"status": "not_applicable", "is_intermittent": False}
    nonzero = np.flatnonzero(values > 0)
    zero_ratio = float(np.mean(values == 0))
    mean_interval = float(np.mean(np.diff(nonzero))) if nonzero.size >= 2 else None
    return {
        "status": "ok",
        "zero_ratio": zero_ratio,
        "mean_nonzero_interval": mean_interval,
        "is_intermittent": bool(zero_ratio >= 0.4 or (mean_interval or 0) >= 2.0),
    }
