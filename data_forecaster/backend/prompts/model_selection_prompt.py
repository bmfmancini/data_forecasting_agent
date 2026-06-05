"""Prompt for the model selection agent."""

from langchain_core.prompts import ChatPromptTemplate

MODEL_SELECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in time series model selection. Choose the best model based on statistical assessments."),
    ("human", (
        "Evaluate the suitability of ARIMA, SARIMA, Holt-Winters, and EWMA (Exponential Weighted Moving Average) based on these assessments:\n\n"
        "{suitability}\n\n"
        "Select the SINGLE best model and provide a detailed rationale.\n"
        "Your output MUST follow this exact structure:\n"
        "Selected model: <MODEL_NAME>\n\n"
        "## Why <MODEL_NAME> was chosen\n"
        "<Detailed explanation referencing metrics>\n\n"
        "## Model Assessment Summary\n"
        "- **ARIMA**: <suitability detail> — Suitability: High/Medium/Low\n"
        "- **SARIMA**: <suitability detail> — Suitability: High/Medium/Low\n"
        "- **Holt-Winters**: <suitability detail> — Suitability: High/Medium/Low\n"
        "- **EWMA**: <suitability detail> — Suitability: High/Medium/Low\n\n"
        "## Why other models were not chosen\n"
        "- **<Rejected Model 1>**: <reason>\n"
        "- **<Rejected Model 2>**: <reason>\n"
        "- **<Rejected Model 3>**: <reason>"
    ))
])