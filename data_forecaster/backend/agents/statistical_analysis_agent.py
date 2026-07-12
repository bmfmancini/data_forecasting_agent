"""Statistical analysis agent for interpreting time series diagnostics."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from core.llm_factory import get_llm
from core.logging_config import get_logger
from prompts.statistical_analysis_prompt import STATISTICAL_ANALYSIS_PROMPT
from schemas import StatisticalResult
from utils.data_cleaning import detect_outliers_iqr, detect_outliers_zscore
from utils.token_tracking import estimate_input_text, extract_token_usage
from utils.statistical import (
    compute_acf_pacf,
    detect_trend,
    run_adf_test,
    run_kpss_test,
    run_periodogram,
    run_stl_decomposition,
    run_white_noise_test,
    check_variance_stability,
    detect_change_points,
)

logger = get_logger(__name__)


def run_statistical_agent(
    series: pd.Series,
    seasonal_period: int = 12,
    user_domain: str = "General",
    disabled_tests: list[str] | None = None,
) -> StatisticalResult:
    """Run statistical analysis ReAct agent and return a StatisticalResult."""
    disabled = set(disabled_tests or [])

    # ── Compute all stats directly ────────────────────────────────────────────
    # Handle constant series to avoid ValueError in statsmodels (adfuller)
    if series.nunique() <= 1:
        logger.warning("Input series is constant. Skipping statistical tests.")
        return StatisticalResult(
            is_stationary_adf=True,
            adf_statistic=0.0,
            adf_p_value=0.0,
            is_stationary_kpss=True,
            kpss_statistic=0.0,
            kpss_p_value=1.0,
            has_trend=False,
            trend_slope=0.0,
            seasonal_period=seasonal_period,
            dominant_period=0.0,
            disabled_tests=sorted(disabled),
            summary="The provided time series is constant (all values are identical). It is statistically stationary with no detectable trend or seasonal patterns.",
            reasoning_steps=[
                {
                    "thought": "Checking series variance...",
                    "observation": "Series is constant. Bypassing ADF/KPSS tests.",
                }
            ],
        )

    adf = (
        {
            "statistic": 0.0,
            "p_value": 1.0,
            "is_stationary": True,
            "interpretation": "ADF stationarity test skipped by user request.",
        }
        if "adf" in disabled
        else run_adf_test(series)
    )
    kpss_res = (
        {
            "statistic": 0.0,
            "p_value": 1.0,
            "is_stationary": True,
            "interpretation": "KPSS stationarity test skipped by user request.",
        }
        if "kpss" in disabled
        else run_kpss_test(series)
    )
    trend = (
        {
            "has_trend": False,
            "slope": 0.0,
            "interpretation": "Trend detection skipped by user request.",
        }
        if "trend" in disabled
        else detect_trend(series)
    )
    periodogram = (
        {
            "dominant_period": 0.0,
            "frequencies": [],
            "power": [],
            "interpretation": "Periodogram skipped by user request.",
        }
        if "periodogram" in disabled
        else run_periodogram(series)
    )
    outliers_iqr = detect_outliers_iqr(series)
    outliers_zscore = detect_outliers_zscore(series)
    white_noise = (
        {
            "p_value": 1.0,
            "is_white_noise": False,
            "interpretation": "White-noise test skipped by user request.",
        }
        if "white_noise" in disabled
        else run_white_noise_test(series)
    )
    var_stability = (
        {
            "is_unstable": False,
            "correlation": 0.0,
            "interpretation": "Variance stability check skipped by user request.",
        }
        if "variance_stability" in disabled
        else check_variance_stability(series)
    )

    # Determine which outlier detection method to recommend based on data characteristics
    # Z-score is better for normally distributed data, IQR for skewed distributions
    # We'll use a simple heuristic: if the data is relatively symmetric and not heavily skewed,
    # z-score might be more appropriate
    skewness = series.dropna().skew()
    kurtosis = series.dropna().kurtosis()

    # Prefer z-score for more normal distributions (low skewness and kurtosis close to 0)
    # and when z-score detects fewer outliers than IQR (indicating IQR might be too aggressive)
    use_zscore = (
        abs(skewness) < 1.0
        and abs(kurtosis) < 3.0
        and outliers_zscore["count"] <= outliers_iqr["count"]
    )

    # Use the selected outlier detection method for reporting
    outliers = outliers_zscore if use_zscore else outliers_iqr

    # Infer seasonal period: prefer explicit arg, validate against periodogram
    dom_period = periodogram["dominant_period"]
    inferred_period: int | None = seasonal_period
    if dom_period < 100:
        pg_period = int(round(dom_period))
        if pg_period > 1:
            # Prefer frequency-derived period but log any mismatch
            if abs(pg_period - seasonal_period) > 2:
                logger.info(
                    "Periodogram period %d differs from freq-derived period %d; using %d",
                    pg_period,
                    seasonal_period,
                    seasonal_period,
                )

    # ── Build Statistical Profile ─────────────────────────────────────────────
    stl = (
        None
        if "stl" in disabled
        else run_stl_decomposition(series, period=inferred_period or 12)
    )
    change_points = (
        {
            "change_points": [],
            "method_used": "disabled",
            "threshold": None,
            "interpretation": "Change-point detection skipped by user request.",
        }
        if "change_points" in disabled
        else detect_change_points(series)
    )
    acf_data = None if "acf_pacf" in disabled else compute_acf_pacf(series)
    conf_bound = 1.96 / np.sqrt(len(series))
    sig_acf = []
    if acf_data is not None:
        sig_acf = [
            i
            for i, v in enumerate(acf_data["acf_values"][1:], 1)
            if abs(v) > conf_bound
        ]

    # Treat 'Skip' or the generic 'Other' as a trigger for AI inference
    is_inferred = user_domain in ["Skip / Let AI Guess", "Other (Custom)"]
    domain_info = (
        f"USER-SPECIFIED DOMAIN: {user_domain}"
        if not is_inferred
        else "DOMAIN: User skipped or requested inference (AI must infer domain from stats)"
    )

    # Add information about which outlier detection method was used
    outlier_method_info = (
        f"Outliers ({'Z-score' if use_zscore else 'IQR'}): {outliers['interpretation']}"
    )
    outlier_comparison = f"Outlier Comparison: IQR found {outliers_iqr['count']} outliers, Z-score found {outliers_zscore['count']} outliers"
    seasonal_range = (
        max(stl["seasonal"]) - min(stl["seasonal"]) if stl is not None else 0.0
    )
    disabled_info = (
        f"Disabled statistical tests for this forecast: {sorted(disabled)}\n"
        if disabled
        else ""
    )

    profile = (
        f"{domain_info}\n"
        f"{disabled_info}"
        f"STATISTICAL PROFILE:\n"
        f"- ADF: {adf['interpretation']}\n"
        f"- KPSS: {kpss_res['interpretation']}\n"
        f"- Trend: {trend['interpretation']}\n"
        f"- {outlier_method_info}\n"
        f"- {outlier_comparison}\n"
        f"- Skewness: {skewness:.2f} (Z-score preferred for values near 0)\n"
        f"- Kurtosis: {kurtosis:.2f} (Z-score preferred for values near 0)\n"
        f"- Randomness: {white_noise['interpretation']}\n"
        f"- Variance Stability: {var_stability['interpretation']}\n"
        f"- Dominant Period: {periodogram['dominant_period']:.2f}\n"
        f"- STL Seasonal Range: {seasonal_range:.2f}\n"
        f"- Change Points: {change_points['interpretation']}\n"
        f"- Significant ACF Lags: {sig_acf[:5]}"
    )

    # ── LLM Setup ────────────────────────────────────────────────────────────
    llm = get_llm(temperature=0)

    prompt = STATISTICAL_ANALYSIS_PROMPT

    recommended_remediation = []
    domain_guess = user_domain if not is_inferred else "General / Unknown"
    token_usage: dict[str, int] = {}
    try:
        chain = prompt | llm
        inputs = {"profile": profile}
        response = chain.invoke(inputs)
        summary = response.content
        token_usage = extract_token_usage(
            response, input_text=estimate_input_text(prompt, inputs)
        )

        if match := re.search(r"DOMAIN:\s*([^\n\.]+)", summary, re.IGNORECASE):
            domain_guess = match.group(1).strip()

        if "APPLY_IQR" in summary or ("APPLY_OUTLIER" in summary and not use_zscore):
            recommended_remediation.append("iqr_clip")
        if "APPLY_ZSCORE" in summary or ("APPLY_OUTLIER" in summary and use_zscore):
            recommended_remediation.append("zscore_clip")
        if "APPLY_BOXCOX" in summary and "box_cox" not in disabled:
            recommended_remediation.append("box_cox")
        if "CHANGE_POINTS_DETECTED" in summary and "change_points" not in disabled:
            recommended_remediation.append("change_point_analysis")

        reasoning_steps = [
            {
                "thought": "Running ADF, KPSS, and STL in Python...",
                "observation": profile,
            },
            {
                "thought": "Generating qualitative interpretation...",
                "observation": "Complete",
            },
        ]
    except Exception as exc:
        logger.warning("Statistical agent LLM call failed: %s", exc)
        summary = (
            f"ADF: {'stationary' if adf['is_stationary'] else 'non-stationary'}. "
            f"KPSS: {'stationary' if kpss_res['is_stationary'] else 'non-stationary'}. "
            f"Trend: {'present' if trend['has_trend'] else 'absent'}."
        )
        if disabled:
            summary += (
                " Disabled by user for this forecast: "
                f"{', '.join(sorted(disabled))}."
            )
        reasoning_steps = [
            {
                "thought": f"Statistical agent failed: {str(exc)}",
                "observation": "Falling back to raw statistical test results.",
            }
        ]

    if disabled and "Disabled by user for this forecast" not in summary:
        summary += (
            "\n\nDisabled by user for this forecast: "
            f"{', '.join(sorted(disabled))}."
        )

    logger.info(
        "Statistical analysis complete. stationary_adf=%s seasonal_period=%s",
        adf["is_stationary"],
        inferred_period,
    )

    return StatisticalResult(
        is_stationary_adf=adf["is_stationary"],
        adf_statistic=adf["statistic"],
        adf_p_value=adf["p_value"],
        is_stationary_kpss=kpss_res["is_stationary"],
        kpss_statistic=kpss_res["statistic"],
        kpss_p_value=kpss_res["p_value"],
        has_trend=trend["has_trend"],
        trend_slope=trend["slope"],
        outlier_count=outliers["count"],
        outlier_ratio=outliers["ratio"],
        is_white_noise=white_noise["is_white_noise"],
        white_noise_p_value=white_noise["p_value"],
        recommended_remediation=recommended_remediation,
        domain=domain_guess,
        seasonal_period=inferred_period,
        dominant_period=periodogram["dominant_period"],
        disabled_tests=sorted(disabled),
        summary=summary,
        reasoning_steps=reasoning_steps,
        token_usage=token_usage,
    )
