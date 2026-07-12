"""Matplotlib backend bootstrap for non-interactive server rendering."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as pyplot  # noqa: E402

__all__ = ["pyplot"]
