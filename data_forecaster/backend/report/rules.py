"""Configurable business rules for the executive report.

All thresholds, classifications, and status mappings used by the
:class:`ExecutiveReportBuilder` are defined here as module-level constants.
Values are configurable via environment variables with sensible defaults so
they can be tuned without code changes and are fully unit-testable.

Rules are organised into the following groups:
- :data:`CONFIDENCE_DEDUCTIONS` — point deductions for the confidence score.
- :data:`CONFIDENCE_LABELS` — score-to-label boundaries.
- :data:`DATA_QUALITY_THRESHOLDS` — data quality rating thresholds.
- :data:`HEALTH_STATUS_RULES` — health indicator status mappings.
- :data:`RECOMMENDATION_PRIORITIES` — recommendation priority rules.
- :data:`DASHBOARD_STATUS_COLORS` — status-to-Bootstrap-colour mapping.
- :data:`MAPE_QUALITY_BANDS` — MAPE-to-quality-description bands.
"""

from __future__ import annotations

from utils.env_helpers import env_float, env_int

# ── Confidence Score Deductions ──────────────────────────────────────────────
# Each entry maps a signal to the points deducted from the starting score
# of 100.  The builder checks conditions in order and applies deductions.

CONFIDENCE_DEDUCTIONS: dict[str, int] = {
    "mape_above_20": env_int("CONF_DEDUCT_MAPE_ABOVE_20", 20),
    "mape_above_10": env_int("CONF_DEDUCT_MAPE_ABOVE_10", 10),
    "mape_above_5": env_int("CONF_DEDUCT_MAPE_ABOVE_5", 5),
    "non_stationary_adf": env_int("CONF_DEDUCT_NON_STATIONARY", 10),
    "white_noise": env_int("CONF_DEDUCT_WHITE_NOISE", 15),
    "outlier_ratio_high": env_int("CONF_DEDUCT_OUTLIERS", 5),
    "missing_data": env_int("CONF_DEDUCT_MISSING_DATA", 5),
    "review_warn": env_int("CONF_DEDUCT_REVIEW_WARN", 10),
    "review_fail": env_int("CONF_DEDUCT_REVIEW_FAIL", 20),
    "structural_breaks": env_int("CONF_DEDUCT_STRUCTURAL_BREAKS", 5),
}

# ── Confidence Labels ────────────────────────────────────────────────────────
# Score boundaries for High / Medium / Low labels.

CONFIDENCE_LABELS: dict[str, int] = {
    "high_min": env_int("CONF_LABEL_HIGH_MIN", 75),
    "medium_min": env_int("CONF_LABEL_MEDIUM_MIN", 50),
}

# ── Data Quality Thresholds ──────────────────────────────────────────────────
# Maximum allowed counts for each rating level.  If any threshold is
# exceeded, the rating drops to the next lower level.

DATA_QUALITY_THRESHOLDS: dict[str, dict[str, int]] = {
    "good": {
        "max_missing": env_int("DQ_GOOD_MAX_MISSING", 0),
        "max_duplicates": env_int("DQ_GOOD_MAX_DUPLICATES", 0),
        "max_gaps": env_int("DQ_GOOD_MAX_GAPS", 0),
        "max_issues": env_int("DQ_GOOD_MAX_ISSUES", 0),
    },
    "fair": {
        "max_missing": env_int("DQ_FAIR_MAX_MISSING", 5),
        "max_duplicates": env_int("DQ_FAIR_MAX_DUPLICATES", 2),
        "max_gaps": env_int("DQ_FAIR_MAX_GAPS", 3),
        "max_issues": env_int("DQ_FAIR_MAX_ISSUES", 3),
    },
    # "poor" is the fallback when fair thresholds are exceeded.
}

# ── MAPE Quality Bands ───────────────────────────────────────────────────────
# Maps MAPE ranges to a human-readable quality description.

MAPE_QUALITY_BANDS: list[tuple[float, str]] = [
    (10.0, "high (MAPE < 10%)"),
    (20.0, "acceptable (MAPE 10–20%)"),
    (50.0, "moderate (MAPE 20–50%)"),
    (float("inf"), "low (MAPE > 50% — treat with caution)"),
]


def mape_quality(mape: float) -> str:
    """Return the quality description for a given MAPE value.

    Args:
        mape: Mean absolute percentage error.

    Returns:
        Human-readable quality band string.
    """
    for threshold, label in MAPE_QUALITY_BANDS:
        if mape < threshold:
            return label
    return MAPE_QUALITY_BANDS[-1][1]


# ── Visual Strategy Thresholds ────────────────────────────────────────────────
# Thresholds used by the report-generation agent's visual strategy recommender.
# Centralised here so the visual strategy stays aligned with the confidence
# and health-indicator bands defined elsewhere in this module.

VISUAL_STRATEGY_THRESHOLDS: dict[str, float] = {
    "mape_high": env_float("VISUAL_MAPE_HIGH", 15.0),
    "mape_moderate": env_float("VISUAL_MAPE_MODERATE", 10.0),
    "outlier_ratio_high": env_float("VISUAL_OUTLIER_RATIO_HIGH", 0.05),
    "trend_slope_min": env_float("VISUAL_TREND_SLOPE_MIN", 0.01),
}


# ── Health Indicator Status Rules ────────────────────────────────────────────
# Status strings used in the health indicators table.  These are the
# possible values; the builder selects the appropriate one based on the
# statistical signals.

HEALTH_STATUS: dict[str, dict[str, str]] = {
    "trend_stability": {
        "stable": "Stable",
        "changing": "Changing",
    },
    "seasonality": {
        "strong": "Strong",
        "weak": "Weak",
        "none": "None",
    },
    "structural_breaks": {
        "monitor": "Monitor",
        "none": "None detected",
    },
    "residual_diagnostics": {
        "acceptable": "Acceptable",
        "concerning": "Concerning",
    },
}

# ── Recommendation Priorities ────────────────────────────────────────────────
# Maps (verdict, confidence_label) to a default priority for the primary
# recommendation.  Used by the builder's recommendation engine.

RECOMMENDATION_PRIORITIES: dict[str, dict[str, str]] = {
    "fail": {"High": "High", "Medium": "High", "Low": "High"},
    "warn": {"High": "Medium", "Medium": "Medium", "Low": "Medium"},
    "pass": {"High": "Medium", "Medium": "Low", "Low": "Low"},
}

# ── Dashboard Status Colours ─────────────────────────────────────────────────
# Maps a status token to a Bootstrap border/context colour class for the
# frontend dashboard cards.

DASHBOARD_STATUS_COLORS: dict[str, str] = {
    "positive": "success",
    "negative": "danger",
    "warning": "warning",
    "neutral": "primary",
    "info": "info",
}

# ── Forecast Direction Labels ────────────────────────────────────────────────

FORECAST_DIRECTIONS: dict[str, str] = {
    "upward": "Upward",
    "downward": "Downward",
    "flat": "Flat",
}

# ── Confidence Label Helper ──────────────────────────────────────────────────


def confidence_label(score: int) -> str:
    """Return the confidence label for a given score.

    Args:
        score: Numeric confidence score (0–100).

    Returns:
        "High" (≥75), "Medium" (50–74), or "Low" (<50).
    """
    if score >= CONFIDENCE_LABELS["high_min"]:
        return "High"
    if score >= CONFIDENCE_LABELS["medium_min"]:
        return "Medium"
    return "Low"


# ── Data Quality Rating Helper ───────────────────────────────────────────────


def data_quality_rating(
    missing: int,
    duplicates: int,
    gaps: int,
    issues_count: int,
    is_regular: bool,
) -> str:
    """Determine the data quality rating from validation counts.

    Args:
        missing:       Number of missing values.
        duplicates:    Number of duplicate timestamps.
        gaps:          Number of missing timestamps (gaps).
        issues_count:  Number of validation issues.
        is_regular:    Whether the series has regular intervals.

    Returns:
        "Good", "Fair", or "Poor".
    """
    good = DATA_QUALITY_THRESHOLDS["good"]
    fair = DATA_QUALITY_THRESHOLDS["fair"]
    if (
        missing <= good["max_missing"]
        and duplicates <= good["max_duplicates"]
        and gaps <= good["max_gaps"]
        and issues_count <= good["max_issues"]
        and is_regular
    ):
        return "Good"
    if (
        missing <= fair["max_missing"]
        and duplicates <= fair["max_duplicates"]
        and gaps <= fair["max_gaps"]
        and issues_count <= fair["max_issues"]
    ):
        return "Fair"
    return "Poor"
