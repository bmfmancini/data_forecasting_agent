"""Prompts module for the forecasting agents."""

from __future__ import annotations

from prompts.data_validation_prompt import DATA_VALIDATION_PROMPT
from prompts.forecasting_prompt import FORECASTING_PROMPT
from prompts.general_chat_prompt import GENERAL_CHAT_PROMPT
from prompts.model_selection_prompt import MODEL_SELECTION_PROMPT
from prompts.orchestrator_prompt import ORCHESTRATOR_CHAT_PROMPT
from prompts.report_generation_prompt import REPORT_GENERATION_PROMPT
from prompts.statistical_analysis_prompt import STATISTICAL_ANALYSIS_PROMPT

__all__ = [
    "DATA_VALIDATION_PROMPT",
    "FORECASTING_PROMPT",
    "GENERAL_CHAT_PROMPT",
    "MODEL_SELECTION_PROMPT",
    "ORCHESTRATOR_CHAT_PROMPT",
    "REPORT_GENERATION_PROMPT",
    "STATISTICAL_ANALYSIS_PROMPT",
]
