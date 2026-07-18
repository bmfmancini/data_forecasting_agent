"""Forecast memory measurement, capacity discovery, and resource estimates."""

from __future__ import annotations

import os
import resource
import sys
from dataclasses import dataclass
from typing import Any

_MIB = 1024 * 1024


@dataclass(frozen=True)
class MemorySnapshot:
    """Current and process-lifetime peak resident memory in MiB."""

    current_rss_mb: float
    peak_rss_mb: float


@dataclass(frozen=True)
class ForecastMemoryEstimate:
    """Dimension-based forecast working-set estimate in MiB."""

    observations: int
    diagnostic_mb: int
    model_mb: int
    base_mb: int
    total_mb: int


def memory_snapshot() -> MemorySnapshot:
    """Measure process RSS without adding a heavyweight monitoring dependency."""
    current_bytes = 0
    try:
        with open("/proc/self/statm", encoding="ascii") as statm:
            resident_pages = int(statm.read().split()[1])
        current_bytes = resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        # ``ru_maxrss`` is the best portable fallback, although it is a peak.
        current_bytes = _peak_rss_bytes()
    return MemorySnapshot(
        current_rss_mb=current_bytes / _MIB,
        peak_rss_mb=_peak_rss_bytes() / _MIB,
    )


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS and BSD report bytes.
    return peak if sys.platform == "darwin" else peak * 1024


def cgroup_memory_limit_mb() -> int | None:
    """Return the effective cgroup memory limit, ignoring unlimited sentinels."""
    candidates = (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    )
    for path in candidates:
        try:
            with open(path, encoding="ascii") as limit_file:
                raw = limit_file.read().strip()
            if raw == "max":
                continue
            limit = int(raw)
            # Docker/Linux unlimited values are close to LONG_MAX.
            if 0 < limit < (1 << 60):
                return limit // _MIB
        except (OSError, ValueError):
            continue
    return None


def host_memory_mb() -> int | None:
    """Return physical host memory when discoverable."""
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / _MIB)
    except (OSError, ValueError):
        return None


def effective_memory_capacity_mb(configured_mb: int, headroom_mb: int) -> int:
    """Resolve an explicit budget or derive a conservative runtime budget."""
    if configured_mb > 0:
        return configured_mb
    limits = [value for value in (cgroup_memory_limit_mb(), host_memory_mb()) if value]
    visible_mb = min(limits) if limits else 4096
    return max(512, int(visible_mb * 0.75) - headroom_mb)


def estimate_theil_sen_workspace_mb(observations: int) -> int:
    """Estimate SciPy's exact pairwise-slope arrays from the series length."""
    pairs = max(0, observations * (observations - 1) // 2)
    # SciPy materializes two n-by-n difference arrays, boolean indexing
    # temporaries, the slope vector, and sort workspace. Empirical Linux peaks
    # are about 6.2x the final float64 pair vector; 6.5 retains headroom.
    return max(16, int((pairs * 8 * 6.5) / _MIB) + 16)


def estimate_bounded_theil_sen_resident_mb(observations: int) -> int:
    """Estimate file-backed RSS during exact in-place rank selection."""
    pairs = max(0, observations * (observations - 1) // 2)
    # Under low pressure mapped pages are visible in RSS even though the kernel
    # can reclaim them. Account for the full slope vector plus process scratch.
    return max(192, int(pairs * 8 / _MIB) + 192)


def estimate_arima_workspace_mb(observations: int, seasonal_period: int = 1) -> int:
    """Estimate peak stepwise ARIMA search workspace from modeled dimensions."""
    seasonal_factor = 1.0 + min(max(seasonal_period, 1), 365) / 730
    # Calibrated conservatively against representative 2k-14k point searches.
    return max(192, int((150 + observations * 0.115) * seasonal_factor))


def estimate_forecast_memory(
    observations: int,
    seasonal_period: int = 1,
    *,
    diagnostic_budget_mb: int = 512,
    horizon: int = 1,
    origins: int = 5,
    candidate_count: int = 8,
) -> ForecastMemoryEstimate:
    """Estimate one job using dimensions only, never dataset identity or domain."""
    observations = max(1, int(observations))
    in_memory_diagnostic_mb = estimate_theil_sen_workspace_mb(observations)
    diagnostic_mb = (
        in_memory_diagnostic_mb
        if in_memory_diagnostic_mb <= diagnostic_budget_mb
        else estimate_bounded_theil_sen_resident_mb(observations)
    )
    model_mb = estimate_arima_workspace_mb(observations, seasonal_period)
    model_mb += max(1, horizon) * max(1, seasonal_period) // 8
    base_mb = max(192, int(observations * 0.012) + 128)
    base_mb += max(1, origins) * max(1, candidate_count) * 2
    total_mb = base_mb + max(diagnostic_mb, model_mb)
    return ForecastMemoryEstimate(
        observations=observations,
        diagnostic_mb=diagnostic_mb,
        model_mb=model_mb,
        base_mb=base_mb,
        total_mb=total_mb,
    )


def estimate_modeled_observations(
    frame: Any, date_col: str, frequency: str | None
) -> int:
    """Estimate the regularized grid length without dataset-specific rules."""
    import pandas as pd

    if frame is None or date_col not in frame or len(frame) == 0:
        return 1
    dates = pd.to_datetime(frame[date_col], errors="coerce").dropna()
    if dates.empty or not frequency:
        return max(1, len(frame))
    try:
        return max(1, len(pd.date_range(dates.min(), dates.max(), freq=frequency)))
    except (TypeError, ValueError, OverflowError):
        return max(1, len(frame))
