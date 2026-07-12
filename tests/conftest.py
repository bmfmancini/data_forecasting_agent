"""Shared test import bootstrap for repository-level tests."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_ROOT = _REPO_ROOT / "data_forecaster" / "backend"
_FRONTEND_ROOT = _REPO_ROOT / "data_forecaster" / "frontend"

for _root in (_FRONTEND_ROOT, _BACKEND_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
