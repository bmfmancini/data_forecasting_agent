"""Prompts module for the forecasting agents."""

from .data_validation_prompt import DATA_VALIDATION_PROMPT
from .forecasting_prompt import FORECASTING_PROMPT
from .model_selection_prompt import MODEL_SELECTION_PROMPT
from .statistical_analysis_prompt import STATISTICAL_ANALYSIS_PROMPT
from .report_generation_prompt import REPORT_GENERATION_PROMPT

__all__ = [
    "DATA_VALIDATION_PROMPT",
    "FORECASTING_PROMPT",
    "MODEL_SELECTION_PROMPT",
    "STATISTICAL_ANALYSIS_PROMPT",
    "REPORT_GENERATION_PROMPT",
]