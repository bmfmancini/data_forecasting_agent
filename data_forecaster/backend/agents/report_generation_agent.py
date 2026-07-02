from __future__ import annotations

from typing import Any

from core.config import GEMINI_TEMPERATURE
from core.llm_factory import get_llm
from core.logging_config import get_logger
from rag.knowledge_base import RAGKnowledgeBase
from schemas import (
    ForecastResult,
    ModelSelectionResult,
    StatisticalResult,
    ValidationResult,
)
from prompts.report_generation_prompt import REPORT_GENERATION_PROMPT

logger = get_logger(__name__)

_REPORT_SECTIONS = [
    "forecasting executive summary and business implications",
    "time series data quality and validation best practices",
    "statistical stationarity and seasonality interpretation",
    "ARIMA SARIMA and Holt-Winters model selection criteria",
]


def run_report_agent(
    validation: ValidationResult,
    statistical: StatisticalResult,
    model_selection: ModelSelectionResult,
    forecast: ForecastResult,
    rag_kb: RAGKnowledgeBase,
    user_prompt: str | None = None,
    preflight_options: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]], list[dict[str, str]]]:
    """Use the LLM with RAG context to write a 6-section analyst report."""

    # ── Build analysis context string ─────────────────────────────────────────
    # Derive useful derived stats for richer context
    first_forecast = forecast.forecast[0] if forecast.forecast else None
    last_forecast = forecast.forecast[-1] if forecast.forecast else None
    pct_change = (
        ((last_forecast - first_forecast) / abs(first_forecast)) * 100
        if first_forecast and first_forecast != 0
        else None
    )
    first_lower = forecast.lower_ci[0] if forecast.lower_ci else None
    last_upper = forecast.upper_ci[-1] if forecast.upper_ci else None
    first_date = forecast.forecast_dates[0] if forecast.forecast_dates else "N/A"
    last_date = forecast.forecast_dates[-1] if forecast.forecast_dates else "N/A"
    forecast_values_sample = [round(v, 2) for v in forecast.forecast[:10]]
    if statistical.trend_slope > 0:
        trend_direction = "upward"
    elif statistical.trend_slope < 0:
        trend_direction = "downward"
    else:
        trend_direction = "flat"

    # ── Logic-Driven Visual Strategy ─────────────────────────────────────────
    visual_strategy = []
    if statistical.seasonal_period and statistical.seasonal_period > 1:
        visual_strategy.append(
            {
                "chart": "STL Decomposition",
                "reason": "Strong seasonal patterns detected; decomposition is essential to isolate recurring cycles from underlying growth.",
            }
        )
    if forecast.mape > 15:
        visual_strategy.append(
            {
                "chart": "Forecast Confidence Intervals",
                "reason": "High variance in data requires emphasis on the 95% CI ribbon to communicate risk and uncertainty to the C-suite.",
            }
        )
    if model_selection.selected_model == "SARIMA":
        visual_strategy.append(
            {
                "chart": "ACF/PACF",
                "reason": "Used to validate the seasonal autoregressive components of the selected model.",
            }
        )

    # Additional visualization suggestions based on data characteristics
    if statistical.outlier_ratio > 0.05:  # More than 5% outliers
        visual_strategy.append(
            {
                "chart": "Box Plot",
                "reason": "Significant outliers detected; a box plot would effectively display the distribution and highlight extreme values.",
            }
        )

    if statistical.has_trend and abs(statistical.trend_slope) > 0.01:
        visual_strategy.append(
            {
                "chart": "Trend Analysis",
                "reason": "Clear trend detected; a trend analysis visualization would help illustrate the direction and magnitude of change over time.",
            }
        )

    if forecast.mape > 10:  # High error
        visual_strategy.append(
            {
                "chart": "Forecast Error Plot",
                "reason": "Forecast errors are significant; an error plot would help diagnose model performance and identify patterns in prediction accuracy.",
            }
        )

    # General statistical visualizations
    visual_strategy.append(
        {
            "chart": "Histogram",
            "reason": "A histogram of the data distribution provides insights into central tendency, spread, and shape of the time series.",
        }
    )

    # Identify where the AI was asked to make the decision
    auto_choices = [
        k for k, v in (preflight_options or {}).items() if v == "Let AI Decide"
    ]
    ai_decision_context = ""
    if auto_choices:
        remediation_details = []
        if "duplicate_strategy" in auto_choices:
            remediation_details.append("Aggregating duplicate timestamps using 'mean'")
        if "missing_strategy" in auto_choices:
            remediation_details.append(
                "Handling missing values via 'linear time-interpolation'"
            )
        if "frequency" in auto_choices:
            remediation_details.append(
                f"Automatically inferring frequency as '{validation.frequency}'"
            )

        ai_decision_context = (
            "\nAI REMEDIATION ACTIONS APPLIED:\n"
            + "\n".join(f"- {d}" for d in remediation_details)
            + "\n"
        )

    analysis_context = f"""
ANALYSIS RESULTS SUMMARY
=========================
Domain: {statistical.domain}
Data Quality:
  - Rows: {validation.row_count} | Freq: {validation.frequency} | Regular: {validation.is_regular}
  - Missing/Dupes/Gaps: {validation.missing_values}/{validation.duplicate_timestamps}/{validation.missing_timestamps}
  - Issues: {'; '.join(validation.issues) if validation.issues else 'None'}"""

    if validation.summary:
        analysis_context += f"\n  - Summary: {validation.summary[:250]}..."

    analysis_context += f"""
  {ai_decision_context}

Statistical Analysis:
  - ADF/KPSS Stat: {statistical.is_stationary_adf}/{statistical.is_stationary_kpss}
  - Trend detected: {statistical.has_trend} (slope={statistical.trend_slope:.6f}, direction={trend_direction})"""

    if statistical.summary:
        analysis_context += f"\n  - Statistical summary: {statistical.summary[:400]}..."

    analysis_context += f"""
  - Seasonal period: {statistical.seasonal_period}
  - Dominant periodogram period: {f'{statistical.dominant_period:.2f}' if statistical.dominant_period else 'N/A'}

Model Selection:
  - Selected model: {model_selection.selected_model}
  - Reasoning: {model_selection.explanation[:400]}...
  - Holt-Winters: {model_selection.holt_winters_rejected_reason or 'Selected'}
  - ARIMA: {model_selection.arima_rejected_reason or 'Selected'}
  - SARIMA: {model_selection.sarima_rejected_reason or 'Selected'}

Forecast Results:
  - Model used: {forecast.model_used}
  - Forecast horizon: {len(forecast.forecast)} periods ({first_date} → {last_date})
  - RMSE/MAE/MAPE: {forecast.rmse:.2f}/{forecast.mae:.2f}/{forecast.mape:.1f}%
  - Start/End Values: {round(first_forecast, 2) if first_forecast is not None else 'N/A'} / {round(last_forecast, 2) if last_forecast is not None else 'N/A'}
  - Projected change over horizon: {f'{pct_change:+.1f}%' if pct_change is not None else 'N/A'}
  - First 10 forecast values: {forecast_values_sample}
Visual Strategy Recommendations: {visual_strategy}
"""

    # ── Pre-fetch RAG context in one batch to reduce LLM hits ────────────────
    methodology_context = ""
    try:
        # Consolidate sections into a single retrieval query for efficiency
        combined_query = " ".join(_REPORT_SECTIONS)
        chunks = rag_kb.retrieve(combined_query, k=6)
        methodology_context = "\n".join(chunks)
    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)

    llm = get_llm(temperature=GEMINI_TEMPERATURE)

    reasoning_steps: list[dict[str, Any]] = [
        {"thought": "Pre-retrieving methodology from RAG...", "observation": "Success"}
    ]

    extra_instructions = (
        f"\n\nADDITIONAL USER INSTRUCTIONS:\n{user_prompt.strip()}\n"
        "Incorporate these instructions throughout the report where relevant."
        if user_prompt and user_prompt.strip()
        else ""
    )

    ai_logic_instruction = (
        "\nWhere AI Remediation was applied, you must assume full responsibility for the decision. "
        "Explain the statistical rationale for why the chosen data treatment (aggregation, interpolation, etc.) "
        "was the most robust choice to preserve signal and minimize forecast bias."
        if auto_choices
        else ""
    )

    prompt_template = REPORT_GENERATION_PROMPT

    try:
        # Invoke the LLM once with all the context needed
        chain = prompt_template | llm
        response = chain.invoke(
            {
                "data_context": analysis_context,
                "rag_context": methodology_context,
                "ai_logic_instruction": ai_logic_instruction,
                "extra_instructions": extra_instructions,
                "visual_strategy": str(visual_strategy),
            }
        )
        report = response.content
        reasoning_steps.append(
            {
                "thought": "Generating final report in a single pass...",
                "observation": "Complete",
            }
        )

    except Exception as exc:
        logger.warning(
            "Report agent LLM call failed: %s — generating fallback report.", exc
        )
        report = _fallback_report(validation, statistical, model_selection, forecast)
        reasoning_steps.append(
            {"thought": f"Error: {exc}", "observation": "Fallback generated"}
        )

    if not report.strip():
        report = _fallback_report(validation, statistical, model_selection, forecast)

    logger.info("Report generation complete. Length: %d chars", len(report))
    return report, reasoning_steps, visual_strategy


def _fallback_report(
    validation: ValidationResult,
    statistical: StatisticalResult,
    model_selection: ModelSelectionResult,
    forecast: ForecastResult,
) -> str:
    first_forecast = forecast.forecast[0] if forecast.forecast else None
    last_forecast = forecast.forecast[-1] if forecast.forecast else None
    pct_change = (
        ((last_forecast - first_forecast) / abs(first_forecast)) * 100
        if first_forecast and first_forecast != 0
        else None
    )
    first_date = forecast.forecast_dates[0] if forecast.forecast_dates else "N/A"
    last_date = forecast.forecast_dates[-1] if forecast.forecast_dates else "N/A"
    if statistical.trend_slope > 0:
        trend_direction = "upward"
        trend_verb = "growing"
        trend_action = (
            "capacity and resource planning should account for increasing demand"
        )
        trend_plan = "growth"
    elif statistical.trend_slope < 0:
        trend_direction = "downward"
        trend_verb = "declining"
        trend_action = "cost optimisation and efficiency measures are advisable"
        trend_plan = "contraction"
    else:
        trend_direction = "flat"
        trend_verb = "broadly flat"
        trend_action = (
            "the current operational baseline is broadly appropriate for the near term"
        )
        trend_plan = "stable conditions"
    if forecast.mape < 10:
        mape_quality = "high (MAPE < 10%)"
    elif forecast.mape < 20:
        mape_quality = "acceptable (MAPE 10\u201320%)"
    elif forecast.mape < 50:
        mape_quality = "moderate (MAPE 20\u201350%)"
    else:
        mape_quality = "low (MAPE > 50% \u2014 treat with caution)"

    stationarity_note = (
        "The series is stationary by both ADF and KPSS tests, indicating a stable statistical structure."
        if statistical.is_stationary_adf and statistical.is_stationary_kpss
        else "The series required differencing or transformation to achieve stationarity before modelling."
    )

    seasonality_note = (
        f"A dominant seasonal cycle of **{statistical.seasonal_period} periods** was identified."
        if statistical.seasonal_period
        else "No strong seasonality was detected in this series."
    )

    return f"""## 1. Strategic Overview

This strategic brief outlines the results of a quantitative forecasting analysis conducted on **{validation.row_count} historical observations**
at **{validation.frequency}** frequency. Utilizing the **{forecast.model_used}** methodology, we have identified a high-fidelity path forward
for the core business metrics. Over the {len(forecast.forecast)}-period outlook, the metric is projected to move from
**{round(first_forecast, 2) if first_forecast is not None else 'N/A'}** to
**{round(last_forecast, 2) if last_forecast is not None else 'N/A'}**
({f'{pct_change:+.1f}%' if pct_change is not None else 'net change pending'}).
Overall forecast reliability is rated as **{mape_quality}** (MAPE: {forecast.mape:.2f}%).
Executive attention is directed to the assumption of pattern persistence, which remains the primary sensitivity in this model.

---

## 2. Historical Performance & Trend Analysis

Analysis of the **{validation.row_count} data points** indicates a complex historical structure. For visual confirmation, 
refer to the **Historical Line Chart** and the **STL Decomposition** panels.

**Growth Trajectory:** A **{trend_direction}** trend is {'robustly present' if statistical.has_trend else 'statistically insignificant'}
(slope: {statistical.trend_slope:.6f}), indicating that the baseline is {trend_verb} over the observed timeframe.

**Cyclical Drivers:** {seasonality_note}

**Data Integrity:** {'The source data meets all quality benchmarks for high-precision modeling.' if not validation.issues else 'The following data quality constraints were identified: ' + '; '.join(validation.issues)}.
Gaps detected: {validation.missing_timestamps}. Missing values: {validation.missing_values}.
Duplicate records: {validation.duplicate_timestamps}. Sampling regularity: {'Verified' if validation.is_regular else 'Irregular intervals observed'}.

{statistical.summary}

---

## 3. Future Growth & Market Outlook

The **{len(forecast.forecast)}-period projections** (from **{first_date}** to **{last_date}**) are visualized in the 
**Forecast & Confidence Intervals** chart.

- **Initial Projected Value:** {round(first_forecast, 2) if first_forecast is not None else 'N/A'}
- **Terminal Projected Value:** {round(last_forecast, 2) if last_forecast is not None else 'N/A'}
- **Projected Delta:** {f'{pct_change:+.1f}%' if pct_change is not None else 'N/A'}

The 95% confidence bands (shaded region) illustrate the volatility risk over time. While the central projection 
remains the most probable outcome, the range between **{round(forecast.lower_ci[0], 2) if forecast.lower_ci else 'N/A'}** 
(conservative) and **{round(forecast.upper_ci[-1], 2) if forecast.upper_ci else 'N/A'}** (optimistic) 
provides the necessary boundaries for risk-adjusted planning.

---

## 4. Analytical Methodology & Rigor

Our team applied a multi-stage analytical pipeline to ensure decision-making rigor:

**Validation & Pre-processing:** Rigorous audits for duplicates, gaps, and outliers were conducted prior to model training.

**Stationarity Assessment (ADF/KPSS):** We evaluated the series for statistical stability. {stationarity_note}
Confidence levels: ADF (p={statistical.adf_p_value:.4f}), KPSS (p={statistical.kpss_p_value:.4f}).

**Structural Decomposition:** STL methodology was utilized to isolate underlying growth from seasonal noise and irregular shocks.

**Lag Dependency Analysis:** ACF and PACF visualizations were analyzed to determine the optimal memory depth for the forecasting algorithms.

**Model Selection Matrix:** Competing architectures (ARIMA, SARIMA, Holt-Winters) were stress-tested.
**{model_selection.selected_model}** emerged as the superior choice. {model_selection.explanation[:500]}

---

## 5. Critical Business Assumptions

Stakeholders should consider the following foundational assumptions:

1. **Structural Persistence:** The model assumes the economic and operational drivers of the past will persist. 
   Abrupt market shifts or policy changes are not factored into this baseline.

2. **Stationarity Framework:** {stationarity_note}

3. **Seasonal Stability:** {seasonality_note} {'The model incorporates a dynamic seasonal adjustment layer.' if statistical.seasonal_period else 'No significant seasonality was assumed for this projection.'}

4. **Cadence Consistency:** Future data is assumed to arrive at the current **{validation.frequency}** frequency.

5. **Univariate Isolation:** The model operates solely on historical values; it does not account for exogenous 
   variables like competitor activity or macro-economic indicators.

---

## 6. Model Reliability & Performance Assessment

For a visual benchmarking of our modeling accuracy, refer to the **Model Comparison Chart**.

| Reliability Metric | Performance Value | Executive Interpretation |
|--------------------|-------------------|--------------------------|
| RMSE               | {forecast.rmse:.4f} | Margin of error weighted for large deviations |
| MAE                | {forecast.mae:.4f}  | Average expected deviation per period |
| MAPE               | {forecast.mape:.2f}% | Scale-independent accuracy rating |

**Overall Model Confidence: {mape_quality.upper()}**

---

## 7. Strategic Risks & Operational Constraints

- **Regime Shifts:** Material changes to business operations will render the current model obsolete.
- **Horizon Decay:** Accuracy is highest in the short term; long-term projections carry significantly higher uncertainty.
- **Data Latency:** The forecast is only as current as the last data point. Real-time shifts may not be reflected.
- **Model Specificity:** {forecast.model_used} is optimized for specific data structures and may be less responsive 
  to sudden non-linear spikes compared to alternative methods.

---

## 8. Tactical Recommendations & Next Steps

Based on the quantitative outlook, we propose the following strategic responses:

1. **Growth Strategy Alignment:**
   The identified **{trend_direction}** trajectory indicates that {trend_action}.

2. **Seasonal Capacity Management:**
   {f'Leverage the {statistical.seasonal_period}-period cycle to optimize resource allocation during predictable peak and trough periods.' if statistical.seasonal_period else 'Continue monitoring for emerging seasonal signals as the dataset grows.'}

3. **Risk-Adjusted Budgeting:** Use the 95% Confidence Intervals for conservative financial forecasting and buffer planning.

4. **Continuous Calibration:** Implement a rolling re-estimation of the model on a {validation.frequency} basis 
   to capture emerging shifts in the trend.

5. **Multivariate Expansion:** Future iterations should incorporate exogenous market signals to improve predictive power.
"""
