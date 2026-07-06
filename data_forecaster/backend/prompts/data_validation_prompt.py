"""Prompt for the data validation agent."""

from langchain_core.prompts import ChatPromptTemplate
from .prompt_utils import apply_token_budget, TOKEN_BUDGETS

DATA_VALIDATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a data validation expert. Summarize the data quality for forecasting. "
            "Use only the data provided. If a required metric is missing, state "
            "'Information not available.' Do not infer or fabricate values.",
        ),
        (
            "human",
            (
                "Validate the current time series dataset based on these metrics:\n\n"
                "{report}\n\n"
                "Provide a professional summary of the data quality and suitability for forecasting.{ai_instruction}"
            ),
        ),
    ]
)

# Apply token budget (example budget: 300 tokens)
DATA_VALIDATION_PROMPT = apply_token_budget(DATA_VALIDATION_PROMPT, "data_validation")
