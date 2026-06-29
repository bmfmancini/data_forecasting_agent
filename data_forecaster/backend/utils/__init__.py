"""Utility functions for the Time Series Data Forecaster Agent backend."""

from pathlib import Path

_frontend_utils = Path(__file__).resolve().parents[2] / "frontend" / "utils"
if _frontend_utils.is_dir():
    __path__.append(str(_frontend_utils))
