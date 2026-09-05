"""Prompt templates for executive report narrative generation (Stage 2).

Each narrative section has its own focused :class:`ChatPromptTemplate` that
receives the pre-computed structured data for that section as JSON context,
together with the distilled business context and any user tone instructions.
The LLM is instructed to use ONLY the provided values — it must never invent
metrics, financial impacts, or business conclusions.

Common rules enforced by all prompts:
- Executive tone; the persona is a senior data analyst briefing leadership.
- No statistical jargon in prose (metric names are fine inside Evidence).
- No unsupported business conclusions (staffing, fleet, pricing, revenue,
  cost) unless explicitly supplied by the user via business context.
- No financial fabrication ("$X million" placeholders).
- Use only the values provided in the structured context.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from prompts.prompt_utils import apply_token_budget

# ── Shared system message fragment ───────────────────────────────────────────

_SYSTEM_PREAMBLE = (
    "You are a senior data analyst briefing leadership in plain, confident "
    "prose. Every number, score, and metric has already been computed by the "
    "analytics engine — you are writing the narrative, not forecasting.\n\n"
    "### ABSOLUTE RULES ###\n"
    "1. Use ONLY the values in the structured context. Do NOT invent, "
    "estimate, or fabricate any metric, score, or value.\n"
    "2. Do NOT generate financial impacts (e.g. '$X million') unless the "
    "context explicitly provides them. Write 'financial impact depends on "
    "average revenue per unit and other business KPIs' instead.\n"
    "3. Do NOT make unsupported business conclusions about staffing, fleet "
    "sizing, pricing, marketing, revenue, or operating costs unless those "
    "values appear in the business context. Hedge with 'may warrant a review "
    "of operational capacity' when a capacity implication is plausible.\n"
    "4. Vary sentence structure. Do NOT open consecutive sections or bullets "
    "with the same phrase. Be specific — cite the actual numbers from the "
    "context rather than restating labels.\n"
    "5. No statistical jargon in prose. Do NOT mention: ADF, KPSS, p-values, "
    "differencing, stationarity, residuals, prediction intervals (say "
    "'forecast range'), AR/MA/I components, or model order parameters. Metric "
    "names (MAPE, RMSE, MASE, WAPE) are permitted only inside Evidence "
    "references, not in flowing prose.\n"
    "6. Begin immediately with the narrative — no greetings, no section "
    "headers, no meta-commentary.\n"
    "7. Treat change points as candidates, not confirmed structural breaks. "
    "Recommend validating break dates, effect sizes, and persistence first. "
    "Only if a durable break is validated may you suggest comparing intervention "
    "terms, recency weighting, segmentation, or regime-specific models; never "
    "prescribe one without supporting evidence.\n"
    "8. When supplied, rolling-origin and untouched final-test results are "
    "completed out-of-sample validation. Describe comparisons with newly arriving "
    "actuals as ongoing monitoring, never as the first validation or as evidence "
    "still needed to establish that any validation occurred.\n"
)

# ── Executive Summary Narrative ──────────────────────────────────────────────

EXECUTIVE_SUMMARY_NARRATIVE_PROMPT = apply_token_budget(
    ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PREAMBLE),
            (
                "human",
                "Write a concise executive summary (3-5 sentences) for the "
                "following forecast. The audience should understand the "
                "forecast in less than one minute. Cover: strategic outlook, "
                "first-to-last endpoint change, why confidence is at its level, the primary "
                "risk, and the recommended action. Do not repeat the raw "
                "values verbatim — weave them into executive prose. Never call "
                "an endpoint change growth, decline, expansion, or contraction, "
                "especially for a seasonal/variable forecast.\n\n"
                "STRUCTURED CONTEXT:\n{section_json}",
            ),
        ]
    ),
    "narrative_executive_summary",
)

# ── Data Quality Narrative ───────────────────────────────────────────────────

DATA_QUALITY_NARRATIVE_PROMPT = apply_token_budget(
    ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PREAMBLE),
            (
                "human",
                "Write a 2-4 sentence data quality summary for executives. "
                "Explain the rating, the most significant issues (if any), "
                "and how data quality may influence forecast reliability. "
                "Preserve the supplied deterministic rating and explanation. "
                "Describe completeness and interval regularity separately from "
                "anomaly risk. Never call anomalies or outliers insignificant, "
                "negligible, immaterial, or too small to affect the rating; state "
                "only the supplied threshold comparison. "
                "Do not list every metric — highlight what matters for "
                "decision-making.\n\n"
                "STRUCTURED CONTEXT:\n{section_json}",
            ),
        ]
    ),
    "narrative_data_quality",
)

# ── Historical Analysis Narrative ────────────────────────────────────────────

HISTORICAL_ANALYSIS_NARRATIVE_PROMPT = apply_token_budget(
    ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PREAMBLE),
            (
                "human",
                "Write a 3-5 sentence historical performance summary for "
                "executives. Explain the trend direction, its business "
                "significance, and any seasonal patterns in plain language. "
                "Do not use statistical terminology.\n\n"
                "STRUCTURED CONTEXT:\n{section_json}",
            ),
        ]
    ),
    "narrative_historical_analysis",
)

# ── Forecast Outlook Narrative ───────────────────────────────────────────────

FORECAST_OUTLOOK_NARRATIVE_PROMPT = apply_token_budget(
    ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PREAMBLE),
            (
                "human",
                "Write a 3-5 sentence forecast outlook for executives. "
                "State metrics.forecast_pattern and the first-to-last endpoint "
                "change separately. Never interpret endpoint change as growth, "
                "decline, expansion, contraction, or trend. Never call a "
                "seasonal/variable path an "
                "upward or downward trajectory. If a "
                "seasonal peak is provided, distinguish that temporary peak "
                "from the endpoint change. Name only metrics.model_used; do "
                "not name any other forecasting model. Emphasise "
                "that forecasts carry uncertainty — reference the "
                "model-based or estimated 95% prediction range for planning. "
                "Never call intervals calibrated unless the structured context "
                "contains both empirical coverage and explicit calibration "
                "evidence; a technical interval label alone is not enough. Do not present "
                "forecasts as exact numbers without uncertainty.\n\n"
                "STRUCTURED CONTEXT:\n{section_json}",
            ),
        ]
    ),
    "narrative_forecast_outlook",
)

# ── Model Comparison Narrative ───────────────────────────────────────────────

MODEL_COMPARISON_NARRATIVE_PROMPT = apply_token_budget(
    ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PREAMBLE),
            (
                "human",
                "Write a 3-5 sentence explanation of why the selected "
                "forecasting model was chosen and what characteristics it "
                "captures. Do not claim it outperformed every alternative "
                "unless the structured rationale explicitly says so. Refer to "
                "displayed validation evidence only when explaining rejection; "
                "do not infer residual or seasonal failure from a higher error. "
                "SARIMA explicitly supports seasonality, so never describe it as "
                "leaving a detected seasonal cycle unmodeled. Refer to "
                "the model as 'the forecasting model' or 'our predictive "
                "model' — the model name may appear once. Do not use "
                "statistical jargon or model order parameters.\n\n"
                "STRUCTURED CONTEXT:\n{section_json}",
            ),
        ]
    ),
    "narrative_model_comparison",
)

# ── Statistical Audit Narrative ──────────────────────────────────────────────

STATISTICAL_AUDIT_NARRATIVE_PROMPT = apply_token_budget(
    ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PREAMBLE),
            (
                "human",
                "Write a 2-4 sentence independent statistical assessment "
                "for executives. Summarise the strongest evidence, key "
                "concerns (if any), and recommended follow-up. Frame any "
                "concerns as forward-looking recommendations, not process "
                "failures. Do not mention agent names or internal pipeline "
                "mechanics.\n\n"
                "STRUCTURED CONTEXT:\n{section_json}",
            ),
        ]
    ),
    "narrative_statistical_audit",
)

# ── Explainability Narrative ─────────────────────────────────────────────────

EXPLAINABILITY_NARRATIVE_PROMPT = apply_token_budget(
    ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PREAMBLE),
            (
                "human",
                "Write a 2-4 sentence explainability summary that helps "
                "executives understand why the AI reached its conclusions. "
                "Translate the findings into plain business language. Do "
                "not use statistical terminology.\n\n"
                "STRUCTURED CONTEXT:\n{section_json}",
            ),
        ]
    ),
    "narrative_explainability",
)

# ── Risk Narrative ───────────────────────────────────────────────────────────

RISK_NARRATIVE_PROMPT = apply_token_budget(
    ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PREAMBLE),
            (
                "human",
                "Write one flowing 2-3 sentence paragraph for the following "
                "risk. Lead with what was detected and the supporting numbers, "
                "then state why it matters, then give the concrete next step. "
                "Merge the description, impact, and mitigation into a single "
                "paragraph — do NOT use 'Risk:', 'Impact:', or 'Mitigation:' "
                "labels. Keep the change-point validation-first sequencing when "
                "the mitigation calls for it. Do not add financial impacts or "
                "business conclusions absent from the context.\n\n"
                "STRUCTURED CONTEXT:\n{section_json}",
            ),
        ]
    ),
    "narrative_risk",
)

# ── Assumption Narrative ─────────────────────────────────────────────────────

ASSUMPTION_NARRATIVE_PROMPT = apply_token_budget(
    ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PREAMBLE),
            (
                "human",
                "Write 1-2 flowing sentences for the following assumption, "
                "grounded in the business context when it is supplied. State "
                "the assumption and, where it fits naturally, the consequence "
                "of it being false — do not use 'Assumption:' or "
                "'Consequence:' labels. Do not introduce conditions, "
                "interventions, or covariates that the context does not "
                "mention. When the business context declares a holiday "
                "calendar, custom events, or covariates, cite them by type as "
                "listed — do not invent dates or counts beyond what is "
                "provided.\n\n"
                "STRUCTURED CONTEXT:\n{section_json}",
            ),
        ]
    ),
    "narrative_assumption",
)

# ── Recommendation Narrative ─────────────────────────────────────────────────

RECOMMENDATION_NARRATIVE_PROMPT = apply_token_budget(
    ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PREAMBLE),
            (
                "human",
                "Rewrite the following recommendation into one merged 2-4 "
                "sentence executive paragraph that combines the action, its "
                "rationale, and the expected outcome — do NOT use 'Action:', "
                "'Rationale:', or 'Expected Outcome:' labels. Do NOT change "
                "the intent, priority, or supporting evidence, and do NOT add "
                "financial impacts or business conclusions absent from the data. "
                "Cite the concrete numbers from the rationale and evidence. "
                "For change-point recommendations, preserve the required order: "
                "validate break dates, effect sizes, and persistence first; only "
                "after confirmation compare intervention terms, recency weighting, "
                "segmentation, or regime-specific models. "
                "If completed rolling-origin or untouched final-test evidence is "
                "present, describe future-actual comparisons as monitoring, not "
                "first-time out-of-sample validation.\n\n"
                "STRUCTURED CONTEXT:\n{section_json}",
            ),
        ]
    ),
    "narrative_recommendation",
)
