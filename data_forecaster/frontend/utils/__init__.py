"""
Utility functions for the Time Series Data Forecaster Agent frontend.
This package contains helper functions and utilities used across the frontend.
"""

from pathlib import Path

_backend_utils = Path(__file__).resolve().parents[2] / "backend" / "utils"
if _backend_utils.is_dir():
    __path__.append(str(_backend_utils))
