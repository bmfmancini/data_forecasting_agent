from __future__ import annotations

from typing import Any
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from core.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_COMPLETION_TOKENS, GROQ_TOP_P, GROQ_REASONING_EFFORT
from core.logging_config import get_logger
from rag.knowledge_base import RAGKnowledgeBase
from schemas import ForecastResult, ModelSelectionResult, StatisticalResult, ValidationResult

logger = get_logger(__name__)

_REACT_PROMPT = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

_REPORT_SECTIONS = [
    "executive summary forecasting results business implications",
    "data quality validation time series",
    "statistical analysis stationarity trend seasonality",
    "model selection ARIMA SARIMA Holt-Winters criteria",
    "forecast accuracy metrics RMSE MAE MAPE confidence intervals interpretation",
    "methodology assumptions limitations forecasting",
    "forecast future outlook risk scenarios business recommendations",
]


def run_report_agent(
    validation: ValidationResult,
    statistical: StatisticalResult,
    model_selection: ModelSelectionResult,
    forecast: ForecastResult,
    rag_kb: RAGKnowledgeBase,
    user_prompt: str | None = None,
    preflight_options: dict[str, Any] | None = None,
) -> str:
    """Use the LLM with RAG context to write a 6-section analyst report."""

    # ── Build analysis context string ─────────────────────────────────────────
    # Derive useful derived stats for richer context
    first_forecast = forecast.forecast[0] if forecast.forecast else None
    last_forecast = forecast.forecast[-1] if forecast.forecast else None
    pct_change = (
        ((last_forecast - first_forecast) / abs(first_forecast)) * 100
        if first_forecast and first_forecast != 0 else None
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

    # Identify where the AI was asked to make the decision
    auto_choices = [k for k, v in (preflight_options or {}).items() if v == "Let AI Decide"]
    ai_decision_context = ""
    if auto_choices:
        ai_decision_context = f"\nAI-DRIVEN REMEDIATION ACTIVE FOR: {', '.join(auto_choices)}\n"

    analysis_context = f"""
ANALYSIS RESULTS SUMMARY
=========================
Data Quality:
  - Rows: {validation.row_count}
  - Missing timestamps: {validation.missing_timestamps}
  - Duplicate timestamps: {validation.duplicate_timestamps}
  - Missing values: {validation.missing_values}
  - Regular intervals: {validation.is_regular}
  - Detected frequency: {validation.frequency}
  - Issues: {'; '.join(validation.issues) if validation.issues else 'None'}
  - Validation summary: {validation.summary}
  {ai_decision_context}

Statistical Analysis:
  - ADF stationary: {statistical.is_stationary_adf} (p={statistical.adf_p_value:.4f}, statistic={statistical.adf_statistic:.4f})
  - KPSS stationary: {statistical.is_stationary_kpss} (p={statistical.kpss_p_value:.4f}, statistic={statistical.kpss_statistic:.4f})
  - Trend detected: {statistical.has_trend} (slope={statistical.trend_slope:.6f}, direction={trend_direction})
  - Seasonal period: {statistical.seasonal_period}
  - Dominant periodogram period: {statistical.dominant_period:.2f}
  - Statistical summary: {statistical.summary}

Model Selection:
  - Selected model: {model_selection.selected_model}
  - Full reasoning: {model_selection.explanation}
  - Holt-Winters: {model_selection.holt_winters_rejected_reason or 'Selected'}
  - ARIMA: {model_selection.arima_rejected_reason or 'Selected'}
  - SARIMA: {model_selection.sarima_rejected_reason or 'Selected'}

Forecast Results:
  - Model used: {forecast.model_used}
  - Forecast horizon: {len(forecast.forecast)} periods ({first_date} → {last_date})
  - RMSE: {forecast.rmse:.4f}
  - MAE: {forecast.mae:.4f}
  - MAPE: {forecast.mape:.2f}%
  - First forecast value: {round(first_forecast, 2) if first_forecast is not None else 'N/A'}
  - Last forecast value: {round(last_forecast, 2) if last_forecast is not None else 'N/A'}
  - Projected change over horizon: {f'{pct_change:+.1f}%' if pct_change is not None else 'N/A'}
  - 95% CI lower bound (first period): {round(first_lower, 2) if first_lower is not None else 'N/A'}
  - 95% CI upper bound (last period): {round(last_upper, 2) if last_upper is not None else 'N/A'}
  - First 10 forecast values: {forecast_values_sample}
"""

    # ── Tool definitions ──────────────────────────────────────────────────────
    retrieved_context: list[str] = []

    @tool
    def retrieve_from_rag(query: str) -> str:
        """Retrieve relevant methodology documentation from the knowledge base.
        Provide a clear topic query such as 'ARIMA model assumptions' or
        'stationarity testing interpretation'."""
        try:
            chunks = rag_kb.retrieve(query, k=3)
            text = "\n---\n".join(chunks)
            retrieved_context.append(text)
            return text
        except Exception as exc:
            logger.warning("RAG retrieval failed for query '%s': %s", query, exc)
            return f"RAG retrieval unavailable: {exc}"

    # ── Run ReAct agent ───────────────────────────────────────────────────────
    tools_list = [retrieve_from_rag]
    llm = ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_COMPLETION_TOKENS,
        model_kwargs={"top_p": GROQ_TOP_P, "reasoning_effort": GROQ_REASONING_EFFORT},
    )
    agent = create_react_agent(llm, tools_list, _REACT_PROMPT)
    executor = AgentExecutor(
        agent=agent, tools=tools_list, verbose=False,
        max_iterations=8, handle_parsing_errors=True,
    )

    rag_queries = " ".join(f"'{q}'," for q in _REPORT_SECTIONS)

    extra_instructions = (
        f"\n\nADDITIONAL USER INSTRUCTIONS:\n{user_prompt.strip()}\n"
        "Incorporate these instructions throughout the report where relevant."
        if user_prompt and user_prompt.strip() else ""
    )

    ai_logic_instruction = (
        "\nWhere 'Let AI Decide' was selected, you must assume full responsibility for the decision. "
        "Explain the statistical rationale for why the chosen data treatment (aggregation, interpolation, etc.) "
        "was the most robust choice to preserve signal and minimize forecast bias."
        if auto_choices else ""
    )

    try:
        result = executor.invoke({
            "input": (
                "You are a senior data scientist writing a formal forecast report for a C-suite executive (CEO level). "
                "The report must be clear, authoritative, and business-oriented — avoid raw jargon, but do not hide analytical rigour. "
                "Use the retrieve_from_rag tool to gather methodology context before writing each relevant section.\n\n"
                f"Here is the full analysis data:\n{analysis_context}\n\n"
                f"First, call retrieve_from_rag for each of these topics: {rag_queries}{ai_logic_instruction}\n\n"
                "Then write the complete report with EXACTLY these 8 sections using Markdown headings (## level):\n\n"
                "## 1. Executive Summary\n"
                "A concise 3-5 sentence brief suitable for a CEO. State the bottom line: what the data shows now, "
                "the headline forecast direction and magnitude, model accuracy, and one key risk or caveat.\n\n"
                "## 2. What We Currently See — State of the Data\n"
                "Describe the current state and historical pattern of the time series in plain business language. "
                "Cover: overall trend direction and strength, whether seasonality exists and what cycle length, "
                "data quality (completeness, regularity, any anomalies), and what the recent trajectory looks like. "
                "Reference specific numbers from the analysis.\n\n"
                "## 3. Where We Think We Are Going — Forecast Outlook\n"
                "Provide a narrative outlook covering the full forecast horizon. "
                "State the projected first and last forecast values, the overall projected change (with percentage), "
                "and describe the widening uncertainty bands as the horizon extends. "
                "Identify the best-case (upper CI) and worst-case (lower CI) scenarios. "
                "Characterise the confidence level in the forecast based on MAPE.\n\n"
                "## 4. Techniques Applied\n"
                "Explain, in plain language, each analytical technique used in this pipeline:\n"
                "- Data validation and cleaning steps\n"
                "- Stationarity tests (ADF and KPSS) — what they measure and what the results mean\n"
                "- STL decomposition — what it reveals about trend and seasonality\n"
                "- ACF/PACF analysis — what the autocorrelation structure tells us\n"
                "- The model selection process and why candidate models were accepted or rejected\n"
                "- The chosen model's mechanism and why it suits this data\n"
                "Incorporate relevant context retrieved from the knowledge base.\n\n"
                "## 5. Assumptions Made\n"
                "Be explicit about every assumption underpinning this forecast:\n"
                "- Stationarity treatment assumptions (differencing applied or not, and why)\n"
                "- Seasonality assumptions (period length, whether it is additive or multiplicative)\n"
                "- The assumption that historical patterns will persist into the future\n"
                "- External factors not modelled (economic shifts, policy changes, black swan events)\n"
                "- Data frequency and regularity assumptions\n"
                "- Any imputation or gap-filling performed on missing values\n\n"
                "## 6. Model Performance & Accuracy\n"
                "Interpret the error metrics for a non-technical reader. "
                "Explain what RMSE, MAE, and MAPE mean in practical terms and whether the accuracy is good, "
                "acceptable, or concerning for the business context. "
                "Compare against rule-of-thumb benchmarks (e.g. MAPE < 10% is generally considered high accuracy).\n\n"
                "## 7. Risks & Limitations\n"
                "Enumerate the key risks and limitations a decision-maker must understand:\n"
                "- Model-specific limitations of the chosen approach\n"
                "- Forecast degradation over longer horizons\n"
                "- Structural break risk\n"
                "- What types of events would invalidate this forecast\n"
                "- Recommendations for when to re-run or recalibrate the model\n\n"
                "## 8. Recommendations\n"
                "Provide 3-5 concrete, actionable recommendations for the business based on the forecast outlook, "
                "the identified trend, and the uncertainty bands. "
                "Frame them in business terms, not statistical ones.\n\n"
                "Write each section fully. Be specific — use the actual numbers from the analysis data above. "
                f"Do not use placeholder text. The report must be ready to present to an executive audience."
                f"{extra_instructions}"
            )
        })
        report = str(result.get("output", ""))
    except Exception as exc:
        logger.warning("Report agent LLM call failed: %s — generating fallback report.", exc)
        report = _fallback_report(validation, statistical, model_selection, forecast)

    if not report.strip():
        report = _fallback_report(validation, statistical, model_selection, forecast)

    logger.info("Report generation complete. Length: %d chars", len(report))
    return report


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
        if first_forecast and first_forecast != 0 else None
    )
    first_date = forecast.forecast_dates[0] if forecast.forecast_dates else "N/A"
    last_date = forecast.forecast_dates[-1] if forecast.forecast_dates else "N/A"
    if statistical.trend_slope > 0:
        trend_direction = "upward"
        trend_verb = "growing"
        trend_action = "capacity and resource planning should account for increasing demand"
        trend_plan = "growth"
    elif statistical.trend_slope < 0:
        trend_direction = "downward"
        trend_verb = "declining"
        trend_action = "cost optimisation and efficiency measures are advisable"
        trend_plan = "contraction"
    else:
        trend_direction = "flat"
        trend_verb = "broadly flat"
        trend_action = "the current operational baseline is broadly appropriate for the near term"
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

    return f"""## 1. Executive Summary

This report presents the results of an automated time series forecast conducted across **{validation.row_count} observations**
at **{validation.frequency}** frequency. The **{forecast.model_used}** model was selected as the most appropriate technique
for this data. Over the {len(forecast.forecast)}-period forecast horizon, the series is projected to move from
**{round(first_forecast, 2) if first_forecast is not None else 'N/A'}** to
**{round(last_forecast, 2) if last_forecast is not None else 'N/A'}**
({f'{pct_change:+.1f}%' if pct_change is not None else 'direction unclear'}).
Forecast accuracy is **{mape_quality}**, with MAPE = {forecast.mape:.2f}%.
The primary caveat is that this model assumes historical patterns will persist; any structural shift in the underlying
business environment would require the model to be recalibrated.

---

## 2. What We Currently See — State of the Data

The time series comprises **{validation.row_count} data points** recorded at **{validation.frequency}** intervals,
spanning from the beginning of the dataset to the most recent available observation.

**Trend:** A **{trend_direction}** trend is {'present' if statistical.has_trend else 'not clearly present'}
(slope = {statistical.trend_slope:.6f}), indicating that the underlying level of the series is
{trend_verb} over time.

**Seasonality:** {seasonality_note}

**Data Quality:** The dataset is {'clean with no issues detected' if not validation.issues else 'generally usable but has the following issues: ' + '; '.join(validation.issues)}.
Missing timestamps: {validation.missing_timestamps}. Missing values: {validation.missing_values}.
Duplicate timestamps: {validation.duplicate_timestamps}. Intervals are {'regular' if validation.is_regular else 'irregular'}.

{statistical.summary}

---

## 3. Where We Think We Are Going — Forecast Outlook

Over the **{len(forecast.forecast)}-period horizon** from **{first_date}** to **{last_date}**:

- **Opening forecast:** {round(first_forecast, 2) if first_forecast is not None else 'N/A'}
- **Closing forecast:** {round(last_forecast, 2) if last_forecast is not None else 'N/A'}
- **Projected change:** {f'{pct_change:+.1f}%' if pct_change is not None else 'N/A'}

The 95% confidence intervals widen progressively with the forecast horizon, reflecting the natural accumulation
of uncertainty in longer-range predictions.

- **Lower bound (conservative scenario, first period):** {round(forecast.lower_ci[0], 2) if forecast.lower_ci else 'N/A'}
- **Upper bound (optimistic scenario, final period):** {round(forecast.upper_ci[-1], 2) if forecast.upper_ci else 'N/A'}

Given the {mape_quality} forecast accuracy, the central projections should be treated as the most likely path,
with the confidence bands framing the realistic range of outcomes.

---

## 4. Techniques Applied

The following analytical pipeline was applied in sequence:

**Data Validation:** The input series was inspected for missing timestamps, duplicate entries, irregular intervals,
and missing values before any modelling took place.

**Stationarity Testing (ADF & KPSS):** The Augmented Dickey-Fuller (ADF) test checks whether the series has a
unit root (i.e., a non-stationary trend). The KPSS test complements this by testing the null of stationarity.
{stationarity_note}
ADF p-value: {statistical.adf_p_value:.4f} | KPSS p-value: {statistical.kpss_p_value:.4f}.

**STL Decomposition:** The series was decomposed into trend, seasonal, and residual components using STL
(Seasonal and Trend decomposition using Loess). This reveals the underlying structure hidden within the raw data.

**ACF / PACF Analysis:** Autocorrelation (ACF) and Partial Autocorrelation (PACF) functions were computed to
understand the serial dependence structure, which informs the lag order selection for ARIMA-family models.

**Model Selection:** Three candidate models were evaluated — ARIMA, SARIMA, and Holt-Winters.
The selection agent reviewed stationarity, trend, and seasonality characteristics to recommend the best fit.
**{model_selection.selected_model}** was selected. {model_selection.explanation[:500]}

---

## 5. Assumptions Made

The following assumptions are embedded in this forecast and must be understood by decision-makers:

1. **Historical pattern persistence:** The model assumes that the statistical structure of the past (trend, seasonality,
   autocorrelation) will continue into the future. Any material change in the business environment, market conditions,
   or data-generating process would invalidate this assumption.

2. **Stationarity treatment:** {stationarity_note} The modelling approach accounts for the degree of differencing required.

3. **Seasonality:** {seasonality_note} {'An additive or multiplicative seasonal component was incorporated into the model.' if statistical.seasonal_period else 'No seasonal adjustment was applied.'}

4. **Data regularity:** The forecast assumes that future observations will arrive at the same **{validation.frequency}** cadence as the historical data.

5. **No external regressors:** This is a univariate model. External factors — such as macroeconomic conditions,
   competitor actions, policy changes, or one-off events — are not incorporated.

6. **Missing value treatment:** {'Missing values were imputed before modelling.' if validation.missing_values > 0 else 'No missing values were present; no imputation was required.'}

7. **Out-of-sample validity:** The model was fitted on all available historical data. Accuracy metrics (RMSE, MAE, MAPE)
   are based on in-sample fit and should be interpreted as optimistic estimates of out-of-sample performance.

---

## 6. Model Performance & Accuracy

| Metric | Value | Interpretation |
|--------|-------|----------------|
| RMSE   | {forecast.rmse:.4f} | Root Mean Squared Error — penalises large errors more heavily |
| MAE    | {forecast.mae:.4f} | Mean Absolute Error — average absolute deviation per period |
| MAPE   | {forecast.mape:.2f}% | Mean Absolute Percentage Error — scale-independent accuracy |

**Overall accuracy: {mape_quality.upper()}**

As a benchmark: MAPE below 10% is generally considered high accuracy in forecasting practice;
10–20% is acceptable for planning purposes; above 20% warrants caution in operational decisions.

---

## 7. Risks & Limitations

- **Structural breaks:** If the business undergoes a significant change (new product, market disruption, regulatory shift),
  the historical pattern will no longer be a reliable guide. The model has no mechanism to detect or adapt to this.

- **Horizon degradation:** Forecast accuracy degrades as the horizon extends. Short-term forecasts (1–3 periods ahead)
  will be materially more reliable than long-term projections.

- **Model-specific constraints:** {forecast.model_used} performs well under its design assumptions but may underperform
  if the data contains nonlinearities, abrupt level shifts, or multiple overlapping seasonal cycles.

- **Univariate limitation:** No external signals (leading indicators, promotional calendars, etc.) are factored in.
  Adding relevant exogenous variables could significantly improve accuracy.

- **Recalibration cadence:** It is recommended to retrain the model whenever a material volume of new data becomes
  available, or at minimum on a {validation.frequency} rolling basis.

---

## 8. Recommendations

Based on the forecast results and identified patterns, the following actions are recommended:

1. **Plan for {trend_plan}:**
   The {trend_direction} trend suggests {trend_action}.

2. **{'Account for seasonal peaks and troughs:' if statistical.seasonal_period else 'Monitor for emerging seasonality:'}**
   {f'With a {statistical.seasonal_period}-period cycle, operational and financial plans should explicitly budget for the predictable high and low points within each cycle.' if statistical.seasonal_period else 'No strong seasonality was detected, but this should be re-evaluated as more data accumulates.'}

3. **Use confidence intervals for scenario planning:** Present the upper and lower 95% CI bounds to leadership
   as the optimistic and conservative scenarios respectively. Avoid treating the central forecast as a certainty.

4. **Set up a model monitoring process:** Track actual vs. forecast values each period. If the error consistently
   exceeds {forecast.mape:.1f}% for three or more consecutive periods, trigger a model review.

5. **Enrich the model with external data:** Consider incorporating relevant exogenous variables (e.g. economic
   indicators, marketing spend, weather) into a multivariate model to improve forecast accuracy.
"""
