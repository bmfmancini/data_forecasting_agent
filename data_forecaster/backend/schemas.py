from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    rows: int
    columns: list[str]
    detected_date_col: Optional[str] = None
    detected_value_col: Optional[str] = None
    detected_frequency: Optional[str] = None


class AnalyzeRequest(BaseModel):
    file_id: str
    forecast_horizon: int
    date_col: Optional[str] = None
    value_col: Optional[str] = None
    forced_model: Optional[str] = None  # "Holt-Winters" | "ARIMA" | "SARIMA" | None (auto)


class ValidationResult(BaseModel):
    is_valid: bool
    row_count: int
    missing_timestamps: int
    duplicate_timestamps: int
    missing_values: int
    is_regular: bool
    frequency: Optional[str] = None
    frequency_alias: Optional[str] = None
    issues: list[str]
    summary: str


class StatisticalResult(BaseModel):
    is_stationary_adf: bool
    adf_statistic: float
    adf_p_value: float
    is_stationary_kpss: bool
    kpss_statistic: float
    kpss_p_value: float
    has_trend: bool
    trend_slope: float
    seasonal_period: Optional[int] = None
    dominant_period: Optional[float] = None
    summary: str


class ModelSelectionResult(BaseModel):
    selected_model: str
    explanation: str
    holt_winters_rejected_reason: Optional[str] = None
    arima_rejected_reason: Optional[str] = None
    sarima_rejected_reason: Optional[str] = None


class ForecastResult(BaseModel):
    model_used: str
    forecast: list[float]
    lower_ci: list[float]
    upper_ci: list[float]
    forecast_dates: list[str]
    rmse: float
    mae: float
    mape: float


class AnalysisResponse(BaseModel):
    file_id: str
    validation: ValidationResult
    statistical: StatisticalResult
    model_selection: ModelSelectionResult
    forecast: ForecastResult
    report: str
    chart_historical: dict
    chart_stl: dict
    chart_acf_pacf: str          # base64 PNG
    chart_forecast: dict
    chart_model_comparison: dict


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str  # "pending"


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending" | "running" | "done" | "error"
    progress: int  # 0–100
    step: str
    result: Optional[dict] = None
    error: Optional[str] = None
