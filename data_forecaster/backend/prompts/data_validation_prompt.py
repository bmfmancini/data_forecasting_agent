"""Prompt for the data validation agent."""

from langchain_core.prompts import ChatPromptTemplate

DATA_VALIDATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a data validation expert. Summarize the data quality for forecasting."),
    ("human", (
        "Validate the current time series dataset based on these metrics:\n\n"
        "{report}\n\n"
        "Provide a professional summary of the data quality and suitability for forecasting.{ai_instruction}"
    ))
])