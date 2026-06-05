"""Prompt for the forecasting agent."""

from langchain_core.prompts import ChatPromptTemplate

FORECASTING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a forecasting expert. Review the performance of multiple models."),
    ("human", (
        "The pre-selected model is: {selected}.\n\n"
        "Review these fitting results:\n"
        "{summary}\n\n"
        "Explain why the selected model is optimal or note if another model achieved better MAPE."
    ))
])