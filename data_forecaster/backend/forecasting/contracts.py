"""Typed contracts shared by forecast adapters and evaluation services."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ForecastFitStatus(StrEnum):
    """Outcome of fitting and evaluating a forecasting model."""

    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"
    NOT_ESTIMABLE = "not_estimable"


class ForecastMetrics(BaseModel):
    """Central forecast metrics and their evaluation metadata."""

    rmse: float | None = None
    mae: float | None = None
    mape: float | None = None
    wape: float | None = None
    mase: float | None = None
    n_evaluated: int = Field(default=0, ge=0)
    unavailable_reasons: dict[str, str] = Field(default_factory=dict)


class ForecastAdapterResult(BaseModel):
    """Result emitted by every model adapter."""

    status: ForecastFitStatus
    forecast: list[float] = Field(default_factory=list)
    lower_ci: list[float] = Field(default_factory=list)
    upper_ci: list[float] = Field(default_factory=list)
    metrics: ForecastMetrics = Field(default_factory=ForecastMetrics)
    fitted_configuration: dict[str, object] = Field(default_factory=dict)
    failure_reason: str | None = None
    is_fallback: bool = False
    warnings: list[str] = Field(default_factory=list)

    @property
    def is_rankable(self) -> bool:
        """Return whether this result has valid point-error evidence."""
        return self.status == ForecastFitStatus.OK and all(
            value is not None
            for value in (self.metrics.rmse, self.metrics.mae, self.metrics.mape)
        )
