from __future__ import annotations

from typing import Any
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import GEMINI_MODEL
from core.logging_config import get_logger
from schemas import ModelSelectionResult, StatisticalResult

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
Thought: {agent_scratchpad}"""
)

_MODELS = ("ARIMA", "SARIMA", "Holt-Winters")


def run_model_selection_agent(stat_result: StatisticalResult) -> ModelSelectionResult:
    """Use the LLM to reason over statistical findings and select the best model."""

    # ── Tool definitions (closed over stat_result) ───────────────────────────

    @tool
    def evaluate_holt_winters_suitability(command: str) -> str:
        """Evaluate whether Holt-Winters is suitable for this time series."""
        points = []
        sp = stat_result.seasonal_period
        if sp and sp > 1:
            points.append(f"Seasonal period {sp} detected — Holt-Winters models seasonality natively.")
        else:
            points.append("No clear seasonal period — Holt-Winters seasonal component may not help.")
        if stat_result.has_trend:
            points.append(f"Trend detected (slope={stat_result.trend_slope:.4f}) — Holt-Winters handles trend via exponential smoothing.")
        else:
            points.append("No significant trend — simple exponential smoothing may suffice.")
        if not stat_result.is_stationary_adf:
            points.append("Non-stationary series — Holt-Winters does not require pre-differencing.")
        points.append("Holt-Winters is fast, interpretable, and robust on short-to-medium series.")
        return "Holt-Winters Assessment:\n" + "\n".join(f"- {p}" for p in points)

    @tool
    def evaluate_arima_suitability(command: str) -> str:
        """Evaluate whether ARIMA is suitable for this time series."""
        points = []
        sp = stat_result.seasonal_period
        if sp and sp > 1:
            points.append(f"Seasonal period {sp} detected — plain ARIMA ignores seasonality; SARIMA may be better.")
        else:
            points.append("No strong seasonality — ARIMA is appropriate.")
        if not stat_result.is_stationary_adf:
            points.append("Non-stationary series — ARIMA handles this via differencing (d parameter).")
        else:
            points.append("Series is stationary — ARIMA(p,0,q) sufficient.")
        if stat_result.has_trend:
            points.append("Trend present — ARIMA differencing (d≥1) will remove it.")
        points.append("ARIMA is well-suited for non-seasonal series with complex autocorrelation.")
        return "ARIMA Assessment:\n" + "\n".join(f"- {p}" for p in points)

    @tool
    def evaluate_sarima_suitability(command: str) -> str:
        """Evaluate whether SARIMA is suitable for this time series."""
        points = []
        sp = stat_result.seasonal_period
        if sp and sp > 1:
            points.append(f"Seasonal period {sp} confirmed — SARIMA explicitly models seasonal AR/MA/I components.")
            points.append("SARIMA is the gold standard for stationary-transformable seasonal series.")
        else:
            points.append("No seasonal period detected — SARIMA seasonal component would overfit.")
        if not stat_result.is_stationary_adf:
            points.append("Non-stationary — SARIMA seasonal differencing (D≥1) will address this.")
        points.append("SARIMA requires more data than ARIMA (at least 2 full seasonal cycles).")
        return "SARIMA Assessment:\n" + "\n".join(f"- {p}" for p in points)

    # ── Heuristic fallback selection (used if LLM fails) ─────────────────────
    sp = stat_result.seasonal_period or 1
    if sp > 1:
        fallback_model = "SARIMA"
        hw_reason = "Strong seasonality makes SARIMA/Holt-Winters preferable."
        arima_reason = "Seasonal pattern detected; plain ARIMA ignores seasonality."
        sarima_reason = None
    else:
        fallback_model = "ARIMA"
        hw_reason = "No clear seasonal pattern detected."
        arima_reason = None
        sarima_reason = "No seasonal period confirmed; SARIMA would overfit."

    # ── Run ReAct agent ───────────────────────────────────────────────────────
    tools_list = [
        evaluate_holt_winters_suitability,
        evaluate_arima_suitability,
        evaluate_sarima_suitability,
    ]
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    agent = create_react_agent(llm, tools_list, _REACT_PROMPT)
    executor = AgentExecutor(
        agent=agent, tools=tools_list, verbose=False, return_intermediate_steps=True,
        max_iterations=5, handle_parsing_errors=True,
    )

    reasoning_steps: list[dict[str, Any]] = []
    try:
        result = executor.invoke({
            "input": (
                "Evaluate the suitability of ARIMA, SARIMA, and Holt-Winters for the current dataset. "
                "You MUST use all three evaluation tools to assess the models based on the underlying "
                "statistical data before selecting the SINGLE best model.\n\n"
                "Your Final Answer MUST follow this exact structure:\n"
                "Selected model: <MODEL_NAME>\n\n"
                "## Why <MODEL_NAME> was chosen\n"
                "<Detailed explanation referencing the specific statistical values above: "
                "ADF/KPSS p-values, seasonal period, trend slope, dominant period. "
                "Explain how each metric directly supports this choice.>\n\n"
                "## Model Assessment Summary\n"
                "- **ARIMA**: <2-3 sentences on suitability for this specific data> — Suitability: High/Medium/Low\n"
                "- **SARIMA**: <2-3 sentences on suitability for this specific data> — Suitability: High/Medium/Low\n"
                "- **Holt-Winters**: <2-3 sentences on suitability for this specific data> — Suitability: High/Medium/Low\n\n"
                "## Why other models were not chosen\n"
                "- **<Rejected Model 1>**: <specific quantitative reason referencing actual values>\n"
                "- **<Rejected Model 2>**: <specific quantitative reason referencing actual values>"
            )
        })
        output = str(result.get("output", ""))
        logger.info("Model selection agent output: %s", output[:200])
        reasoning_steps = [
            {"thought": a.log, "observation": str(o)} for a, o in result.get("intermediate_steps", [])
        ]

        # Parse selected model from output
        selected_model = fallback_model
        for m in _MODELS:
            if f"Selected model: {m}" in output or f"selected model: {m}" in output.lower():
                selected_model = m
                break
        # Broader fallback scan
        if selected_model == fallback_model:
            upper = output.upper()
            for m in _MODELS:
                if m.upper() in upper[:100]:
                    selected_model = m
                    break

        explanation = output
        hw_rej = None if selected_model == "Holt-Winters" else "Not selected based on LLM reasoning."
        arima_rej = None if selected_model == "ARIMA" else "Not selected based on LLM reasoning."
        sarima_rej = None if selected_model == "SARIMA" else "Not selected based on LLM reasoning."

    except Exception as exc:
        logger.warning("Model selection agent LLM call failed: %s — using heuristic.", exc)
        selected_model = fallback_model
        explanation = f"Heuristic selection: {fallback_model} chosen based on seasonal period={sp}."
        hw_rej = hw_reason if selected_model != "Holt-Winters" else None
        arima_rej = arima_reason if selected_model != "ARIMA" else None
        sarima_rej = sarima_reason if selected_model != "SARIMA" else None
        reasoning_steps = [{
            "thought": f"Model selection agent failed: {str(exc)}",
            "observation": f"Falling back to heuristic selection: {selected_model}"
        }]

    logger.info("Selected model: %s", selected_model)

    return ModelSelectionResult(
        selected_model=selected_model,
        explanation=explanation,
        holt_winters_rejected_reason=hw_rej,
        arima_rejected_reason=arima_rej,
        sarima_rejected_reason=sarima_rej,
        reasoning_steps=reasoning_steps,
    )
