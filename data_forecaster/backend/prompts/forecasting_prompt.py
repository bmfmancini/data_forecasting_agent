"""Prompt for the forecasting agent."""

from langchain_core.prompts import ChatPromptTemplate

FORECASTING_PROMPT = ChatPromptTemplate.from_messages([
("system",
"You are a Senior Forecasting Analyst specializing in "
"Holt-Winters, ARIMA, and SARIMA models. "
"Your responsibility is to evaluate model performance and explain "
"the rationale for model selection using evidence from the supplied results. "
"Remain strictly grounded in the provided metrics and diagnostics."
),

("human",
 "SELECTED MODEL:\n"
 "{selected}\n\n"

 "MODEL RESULTS:\n"
 "{summary}\n\n"

 "Evaluate the selected model using the following framework:\n\n"

 "1. Performance Comparison\n"
 "- Compare all available models.\n"
 "- Identify the best and worst performing models.\n"
 "- Reference MAPE, RMSE, MAE, AIC, BIC, or other metrics if available.\n\n"

 "2. Model Selection Assessment\n"
 "- Determine whether the selected model appears justified.\n"
 "- If another model outperformed it, explain the tradeoffs.\n"
 "- Do not automatically recommend the model with the lowest MAPE.\n\n"

 "3. Forecasting Characteristics\n"
 "- Assess evidence of trend.\n"
 "- Assess evidence of seasonality.\n"
 "- Assess stationarity findings if available.\n"
 "- Explain how these characteristics support or weaken the selected model.\n\n"

 "4. Risks and Limitations\n"
 "- Identify potential weaknesses of the selected model.\n"
 "- Highlight any data limitations.\n"
 "- Mention overfitting concerns if supported by the evidence.\n\n"

 "5. Final Recommendation\n"
 "- State whether you agree with the selected model.\n"
 "- Provide a concise rationale.\n"
 "- If a different model should be preferred, explain why.\n\n"

 "Ground all conclusions in the supplied results. "
 "Do not invent metrics, diagnostics, or dataset characteristics."
)

])
