from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    rows: int
    columns: list[str]
    detected_date_col: Optional[str] = None
    detected_value_col: Optional[str] = None
    detected_frequency: Optional[str] = None


class PreflightDecision(BaseModel):
    key: str
    label: str
    message: str
    options: list[str]
    default: Any
    required: bool = False
    allow_custom: bool = False


class PreflightResponse(BaseModel):
    status: str  # "ready" | "warning" | "needs_input"
    detected_frequency: Optional[str] = None
    row_count: int
    usable_observations: int
    duplicate_timestamps: int
    missing_values: int
    missing_timestamps: int
    is_regular: bool
    issues: list[str]
    warnings: list[str]
    decisions: list[PreflightDecision]
    defaults: dict[str, Any]


class AnalyzeRequest(BaseModel):
    file_id: str
    forecast_horizon: int
    date_col: Optional[str] = None
    value_col: Optional[str] = None
    forced_model: Optional[str] = None  # "Holt-Winters" | "ARIMA" | "SARIMA" | None (auto)
    user_prompt: Optional[str] = None   # Extra instructions appended to the report prompt
    preflight_options: Optional[dict[str, Any]] = Field(default_factory=dict)


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
    reasoning_steps: list[dict[str, Any]] = Field(default_factory=list)


class StatisticalResult(BaseModel):
    is_stationary_adf: bool
    adf_statistic: float
    adf_p_value: float
    is_stationary_kpss: bool
    kpss_statistic: float
    kpss_p_value: float
    has_trend: bool
    trend_slope: float
    outlier_count: int = 0
    outlier_ratio: float = 0.0
    is_white_noise: bool = False
    white_noise_p_value: float = 1.0
    recommended_remediation: list[str] = Field(default_factory=list) # e.g. ["iqr_clip", "box_cox"]
    domain: Optional[str] = None
    seasonal_period: Optional[int] = None
    dominant_period: Optional[float] = None
    summary: str
    reasoning_steps: list[dict[str, Any]] = Field(default_factory=list)


class ModelSelectionResult(BaseModel):
    selected_model: str
    explanation: str
    holt_winters_rejected_reason: Optional[str] = None
    arima_rejected_reason: Optional[str] = None
    sarima_rejected_reason: Optional[str] = None
    reasoning_steps: list[dict[str, Any]] = Field(default_factory=list)


class ForecastResult(BaseModel):
    model_used: str
    forecast: list[float]
    lower_ci: list[float]
    upper_ci: list[float]
    forecast_dates: list[str]
    rmse: float
    mae: float
    mape: float
    reasoning_steps: list[dict[str, Any]] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    file_id: str
    validation: ValidationResult
    statistical: StatisticalResult
    model_selection: ModelSelectionResult
    forecast: ForecastResult
    report: str
    report_reasoning: list[dict[str, Any]] = Field(default_factory=list)
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
