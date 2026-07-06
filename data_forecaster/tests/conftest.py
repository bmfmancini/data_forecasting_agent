"""Shared pytest configuration for the Data Forecaster test suite.

Adds the ``backend/`` directory to ``sys.path`` so test modules can
import backend packages (``utils``, ``core``, ``schemas``, etc.)
without per-file ``sys.path`` manipulation.
"""

import os
import sys

# Resolve the backend directory relative to this conftest.py
_backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
_backend_dir = os.path.abspath(_backend_dir)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
