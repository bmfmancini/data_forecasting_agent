"""Pydantic request/response schemas for the Data Forecaster API.

Defines the data contracts exchanged between the FastAPI backend and the
Flask frontend (or any API client).  Schemas are grouped by pipeline stage:
upload, preflight, analysis (validation, statistical, model selection,
forecast), chat, jobs, and API key management.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response returned after a successful file upload."""

    file_id: str
    filename: str
    rows: int
    columns: list[str]
    detected_date_col: str | None = None
    detected_value_col: str | None = None
    detected_frequency: str | None = None


class PreflightDecision(BaseModel):
    """A single preflight decision the user must resolve before forecasting."""

    key: str
    label: str
    message: str
    options: list[str]
    default: Any
    required: bool = False
    allow_custom: bool = False


class PreflightResponse(BaseModel):
    """Result of preflight quality checks on an uploaded series."""

    status: str  # "ready" | "warning" | "needs_input"
    detected_frequency: str | None = None
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
    """Request body for submitting an asynchronous analysis job."""

    file_id: str
    forecast_horizon: int
    date_col: str | None = None
    value_col: str | None = None
    forced_model: str | None = None  # "Holt-Winters" | "ARIMA" | "SARIMA" | None (auto)
    user_prompt: str | None = None  # Extra instructions appended to the report prompt
    preflight_options: dict[str, Any] | None = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """Request schema for the data explorer chat."""

    file_id: str | None = None
    query: str = Field(..., max_length=2000)


class ChatResponse(BaseModel):
    """Response schema for the data explorer chat."""

    answer: str
    visualization_data: dict[str, Any] | None = None
    visualization_type: str | None = None


class ValidationResult(BaseModel):
    """Output of the data validation agent."""

    is_valid: bool
    row_count: int
    missing_timestamps: int
    duplicate_timestamps: int
    missing_values: int
    is_regular: bool
    frequency: str | None = None
    frequency_alias: str | None = None
    issues: list[str]
    summary: str
    reasoning_steps: list[dict[str, Any]] = Field(default_factory=list)
    token_usage: dict[str, Any] = Field(default_factory=dict)


class StatisticalResult(BaseModel):
    """Output of the statistical analysis agent."""

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
    recommended_remediation: list[str] = Field(
        default_factory=list
    )  # e.g. ["iqr_clip", "box_cox"]
    domain: str | None = None
    seasonal_period: int | None = None
    dominant_period: float | None = None
    summary: str
    reasoning_steps: list[dict[str, Any]] = Field(default_factory=list)
    token_usage: dict[str, Any] = Field(default_factory=dict)


class ModelSelectionResult(BaseModel):
    """Output of the model selection agent."""

    selected_model: str
    explanation: str
    holt_winters_rejected_reason: str | None = None
    arima_rejected_reason: str | None = None
    sarima_rejected_reason: str | None = None
    ewma_rejected_reason: str | None = None
    reasoning_steps: list[dict[str, Any]] = Field(default_factory=list)
    token_usage: dict[str, Any] = Field(default_factory=dict)


class ForecastResult(BaseModel):
    """Output of the forecasting agent for the selected model."""

    model_used: str
    forecast: list[float]
    lower_ci: list[float]
    upper_ci: list[float]
    forecast_dates: list[str]
    rmse: float
    mae: float
    mape: float
    reasoning_steps: list[dict[str, Any]] = Field(default_factory=list)
    token_usage: dict[str, Any] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    """Complete response returned by the full 5-agent analysis pipeline."""

    file_id: str
    validation: ValidationResult
    statistical: StatisticalResult
    model_selection: ModelSelectionResult
    forecast: ForecastResult
    report: str
    report_reasoning: list[dict[str, Any]] = Field(default_factory=list)
    strategic_visual_recommendations: list[dict[str, str]] = Field(default_factory=list)
    llm_fallback: bool = (
        False  # Indicates if the LLM was not used for report generation
    )
    chart_historical: dict
    chart_stl: dict
    chart_acf_pacf: str  # base64 PNG
    chart_forecast: dict
    chart_model_comparison: dict
    pipeline_token_usage: dict[str, Any] = Field(default_factory=dict)


class JobSubmitResponse(BaseModel):
    """Response returned when an analysis job is enqueued."""

    job_id: str
    status: str  # "pending"


class JobStatusResponse(BaseModel):
    """Current status of an asynchronous analysis job."""

    job_id: str
    status: str  # "pending" | "running" | "done" | "error"
    progress: int  # 0–100
    step: str
    result: dict | None = None
    error: str | None = None


# ── API Key Management Schemas ────────────────────────────────────────────────


class APIUserCreateRequest(BaseModel):
    """Request schema for creating a new API user."""

    username: str
    description: str = ""
    is_admin: bool = False


class APIUserResponse(BaseModel):
    """Response schema for a single API user (never includes the key hash)."""

    id: int
    username: str
    description: str
    enabled: bool
    bootstrap: bool
    is_admin: bool
    created_at: str
    last_used: str | None = None
    last_used_ip: str | None = None


class APIUserCreatedResponse(BaseModel):
    """Response schema after creating a user — includes one-time plaintext key."""

    user: APIUserResponse
    api_key: str


class APIKeyRotatedResponse(BaseModel):
    """Response schema after rotating a key — includes one-time plaintext key."""

    user_id: int
    api_key: str


class APIUserToggleRequest(BaseModel):
    """Request schema for enabling/disabling an API user."""

    enabled: bool


class APIUserSetAdminRequest(BaseModel):
    """Request schema for promoting or demoting an API user."""

    is_admin: bool


# ── Bootstrap / Auth Status Schemas ──────────────────────────────────────────


class BootstrapRequest(BaseModel):
    """Request schema for the one-time bootstrap endpoint.

    The admin supplies the desired username and API key for the first
    API user.  The ``admin_key`` field is sent via the ``X-Admin-Key``
    header rather than the body.
    """

    username: str
    api_key: str


class BootstrapResponse(BaseModel):
    """Response schema after a successful bootstrap — confirms the user."""

    user: APIUserResponse
    auth_enabled: bool = True


class AuthStatusResponse(BaseModel):
    """Response schema for the auth-status endpoint."""

    auth_enabled: bool
    has_users: bool
