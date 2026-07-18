"""Exact statistical routines whose working memory is explicitly bounded."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class TheilSenResult:
    """SciPy-compatible Theil-Sen result values."""

    slope: float
    intercept: float
    low_slope: float
    high_slope: float


def exact_theilslopes_bounded(
    y: np.ndarray,
    x: np.ndarray | None = None,
    *,
    alpha: float = 0.95,
    workspace_mb: int = 512,
    temp_dir: str | None = None,
) -> TheilSenResult:
    """Compute the exact Theil-Sen result using a disk-backed slope vector.

    Every pairwise slope is retained, so this is not sampling or an
    approximation. RAM use is bounded by one input-sized row and in-place rank
    selection; the quadratic vector is backed by a temporary file.
    """
    values = np.asarray(y, dtype=float).reshape(-1)
    times = (
        np.arange(values.size, dtype=float)
        if x is None
        else np.asarray(x, dtype=float).reshape(-1)
    )
    if values.size != times.size:
        raise ValueError("Incompatible lengths for Theil-Sen inputs.")
    pairs = values.size * (values.size - 1) // 2
    if pairs == 0:
        raise ValueError("At least two observations are required.")

    fd, path = tempfile.mkstemp(prefix="forecast-theil-", suffix=".mmap", dir=temp_dir)
    os.close(fd)
    slopes: np.memmap | None = None
    try:
        slopes = np.memmap(path, dtype=np.float64, mode="w+", shape=(pairs,))
        cursor = 0
        # The outer block makes progress and cancellation instrumentation easy
        # while the per-row vectors keep live RAM strictly O(n).
        bytes_per_row = max(1, values.size * 8 * 2)
        block_rows = max(
            1, min(values.size, workspace_mb * 1024 * 1024 // bytes_per_row)
        )
        for block_start in range(0, values.size - 1, block_rows):
            block_end = min(values.size - 1, block_start + block_rows)
            for row in range(block_start, block_end):
                dx = times[row + 1 :] - times[row]
                valid = dx > 0
                row_slopes = (values[row + 1 :] - values[row])[valid] / dx[valid]
                next_cursor = cursor + row_slopes.size
                slopes[cursor:next_cursor] = row_slopes
                cursor = next_cursor
        if cursor != pairs:
            # Matches SciPy's rule of considering only positive x differences.
            slopes = slopes[:cursor]
        if slopes.size == 0:
            return TheilSenResult(*(float("nan"),) * 4)

        ranks = _theil_ranks(times, values, int(slopes.size), alpha)
        median_low = (slopes.size - 1) // 2
        median_high = slopes.size // 2
        requested = sorted({median_low, median_high, *ranks})
        slopes.partition(requested)
        slope = float((slopes[median_low] + slopes[median_high]) / 2)
        intercept = float(np.median(values) - slope * np.median(times))
        return TheilSenResult(
            slope=slope,
            intercept=intercept,
            low_slope=float(slopes[ranks[0]]),
            high_slope=float(slopes[ranks[1]]),
        )
    finally:
        if slopes is not None:
            slopes.flush()
            del slopes
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _theil_ranks(
    x: np.ndarray, y: np.ndarray, pair_count: int, alpha: float
) -> tuple[int, int]:
    """Return SciPy's confidence interval ranks, including tie correction."""
    tail_alpha = 1 - alpha if alpha > 0.5 else alpha
    z_value = norm.ppf(tail_alpha / 2)
    x_counts = np.unique(x, return_counts=True)[1]
    y_counts = np.unique(y, return_counts=True)[1]
    n = len(y)
    tie_term = sum(
        int(k) * (int(k) - 1) * (2 * int(k) + 5)
        for counts in (x_counts, y_counts)
        for k in counts
        if k > 1
    )
    sigma_squared = (n * (n - 1) * (2 * n + 5) - tie_term) / 18
    sigma = np.sqrt(sigma_squared)
    upper = min(int(np.round((pair_count - z_value * sigma) / 2)), pair_count - 1)
    lower = max(int(np.round((pair_count + z_value * sigma) / 2)) - 1, 0)
    return lower, upper
