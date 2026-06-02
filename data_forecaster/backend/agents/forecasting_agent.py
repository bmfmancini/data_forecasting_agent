from __future__ import annotations

from typing import Any
import pandas as pd
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from core.config import GEMINI_MODEL, USE_OLLAMA, OLLAMA_BASE_URL, OLLAMA_MODEL
from core.logging_config import get_logger
from forecasting.arima_model import fit_arima
from forecasting.holt_winters import fit_holt_winters
from forecasting.sarima_model import fit_sarima
from schemas import ForecastResult, ModelSelectionResult, StatisticalResult

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


def run_forecasting_agent(
    series: pd.Series,
    model_selection: ModelSelectionResult,
    stat_result: StatisticalResult,
    forecast_horizon: int,
    freq: str,
) -> tuple[ForecastResult, dict[str, dict[str, float]]]:
    """Run all three forecasting models, return ForecastResult for the selected model
    and an all-metrics dict for the comparison chart.

    Returns:
        (ForecastResult, all_metrics_dict)
        all_metrics_dict: {"ARIMA": {"RMSE": x, "MAE": y, "MAPE": z}, ...}
    """
    seasonal_period = stat_result.seasonal_period or 12
    results_store: dict[str, dict[str, Any]] = {}

    # ── Tool definitions ──────────────────────────────────────────────────────

    @tool
    def run_holt_winters_tool(command: str) -> str:
        """Fit Holt-Winters exponential smoothing and return evaluation metrics."""
        try:
            res = fit_holt_winters(series, forecast_horizon)
            results_store["Holt-Winters"] = res
            return (
                f"Holt-Winters: RMSE={res['rmse']:.4f}, "
                f"MAE={res['mae']:.4f}, MAPE={res['mape']:.2f}%"
            )
        except Exception as exc:
            logger.warning("Holt-Winters tool failed: %s", exc)
            return f"Holt-Winters fitting failed: {exc}"

    @tool
    def run_arima_tool(command: str) -> str:
        """Fit ARIMA model using auto_arima and return evaluation metrics."""
        try:
            res = fit_arima(series, forecast_horizon)
            results_store["ARIMA"] = res
            return (
                f"ARIMA: RMSE={res['rmse']:.4f}, "
                f"MAE={res['mae']:.4f}, MAPE={res['mape']:.2f}%"
            )
        except Exception as exc:
            logger.warning("ARIMA tool failed: %s", exc)
            return f"ARIMA fitting failed: {exc}"

    @tool
    def run_sarima_tool(command: str) -> str:
        """Fit SARIMA model using auto_arima (seasonal=True) and return evaluation metrics."""
        try:
            res = fit_sarima(series, forecast_horizon, seasonal_period)
            results_store["SARIMA"] = res
            return (
                f"SARIMA (m={seasonal_period}): RMSE={res['rmse']:.4f}, "
                f"MAE={res['mae']:.4f}, MAPE={res['mape']:.2f}%"
            )
        except Exception as exc:
            logger.warning("SARIMA tool failed: %s", exc)
            return f"SARIMA fitting failed: {exc}"

    @tool
    def compute_all_metrics_tool(command: str) -> str:
        """Summarise and compare metrics across all fitted models."""
        if not results_store:
            return "No models have been fitted yet. Run the individual model tools first."
        lines = ["Model comparison (lower is better):"]
        for name, res in results_store.items():
            lines.append(
                f"  {name}: RMSE={res.get('rmse', 'N/A'):.4f}, "
                f"MAE={res.get('mae', 'N/A'):.4f}, MAPE={res.get('mape', 'N/A'):.2f}%"
            )
        return "\n".join(lines)

    # ── Run ReAct agent ───────────────────────────────────────────────────────
    tools_list = [run_holt_winters_tool, run_arima_tool, run_sarima_tool, compute_all_metrics_tool]
    if USE_OLLAMA:
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    else:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

    agent = create_react_agent(llm, tools_list, _REACT_PROMPT)
    executor = AgentExecutor(
        agent=agent, tools=tools_list, verbose=False, return_intermediate_steps=True,
        max_iterations=12, handle_parsing_errors=True, early_stopping_method="generate",
    )

    reasoning_steps: list[dict[str, Any]] = []
    try:
        result = executor.invoke({
            "input": (
                f"The pre-selected model is: {model_selection.selected_model}. "
                "Run all three forecasting tools (Holt-Winters, ARIMA, SARIMA) to fit each model "
                "and collect metrics. Then use compute_all_metrics to compare them. "
                "Report which model achieved the best MAPE."
            )
        })
        reasoning_steps = [
            {"thought": a.log, "observation": str(o)} for a, o in result.get("intermediate_steps", [])
        ]
    except Exception as exc:
        logger.warning("Forecasting agent LLM call failed: %s — running models directly.", exc)
        # Run all models directly as fallback
        for name, fn, kwargs in [
            ("Holt-Winters", fit_holt_winters, {}),
            ("ARIMA", fit_arima, {}),
            ("SARIMA", fit_sarima, {"seasonal_period": seasonal_period}),
        ]:
            try:
                results_store[name] = fn(series, forecast_horizon, **kwargs)
            except Exception as e:
                logger.warning("%s fallback failed: %s", name, e)
        reasoning_steps = [{
            "thought": f"Forecasting agent failed: {str(exc)}",
            "observation": "Attempting to fit models directly without LLM orchestration."
        }]

    # ── Select result for the chosen model ───────────────────────────────────
    selected = model_selection.selected_model
    if selected not in results_store:
        # Try to fit the selected model directly
        try:
            if selected == "Holt-Winters":
                results_store[selected] = fit_holt_winters(series, forecast_horizon)
            elif selected == "ARIMA":
                results_store[selected] = fit_arima(series, forecast_horizon)
            else:
                results_store[selected] = fit_sarima(series, forecast_horizon, seasonal_period)
        except Exception as exc:
            logger.error("Could not fit selected model %s: %s", selected, exc)
            # Fall back to any available result
            if results_store:
                selected = next(iter(results_store))
                logger.warning("Falling back to %s", selected)
            else:
                raise RuntimeError("All forecasting models failed.") from exc

    res = results_store[selected]

    # ── Generate forecast dates ───────────────────────────────────────────────
    last_date = series.index[-1] if hasattr(series.index, "max") else None
    forecast_dates: list[str] = []
    if last_date is not None:
        try:
            date_range = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq=freq)[1:]
            forecast_dates = date_range.strftime("%Y-%m-%d").tolist()
        except Exception:
            forecast_dates = [str(i + 1) for i in range(forecast_horizon)]
    else:
        forecast_dates = [str(i + 1) for i in range(forecast_horizon)]

    # ── Build all_metrics dict for comparison chart ───────────────────────────
    all_metrics = {
        name: {"RMSE": r["rmse"], "MAE": r["mae"], "MAPE": r["mape"]}
        for name, r in results_store.items()
    }

    logger.info("Forecasting complete. Selected: %s", selected)

    forecast_result = ForecastResult(
        model_used=selected,
        forecast=res["forecast"],
        lower_ci=res["lower_ci"],
        upper_ci=res["upper_ci"],
        forecast_dates=forecast_dates,
        rmse=res["rmse"],
        mae=res["mae"],
        mape=res["mape"],
        reasoning_steps=reasoning_steps,
    )
    return forecast_result, all_metrics
