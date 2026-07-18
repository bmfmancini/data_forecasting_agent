"""Pydantic request/response schemas for the Data Forecaster API.

Defines the data contracts exchanged between the FastAPI backend and the
Flask frontend (or any API client).  Schemas are grouped by pipeline stage:
upload, preflight, analysis (validation, statistical, model selection,
forecast), chat, jobs, and API key management.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from forecasting.contracts import ForecastFitStatus


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
    report_name: str = ""
    source_filename: str = ""
    custom_settings: list[dict[str, str]] | None = None
    application_user_id: int | None = None
    application_username: str | None = Field(default=None, max_length=64)
    application_user_is_admin: bool = False


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
    """Output of the statistical analysis agent.

    Includes typed evidence fields for seasonality, stationarity, anomalies,
    change points, and trend. The original scalar fields are preserved for
    backward compatibility with the report builder and statistical review
    agent.
    """

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
    disabled_tests: list[str] = Field(default_factory=list)
    summary: str
    reasoning_steps: list[dict[str, Any]] = Field(default_factory=list)
    token_usage: dict[str, Any] = Field(default_factory=dict)
    # ── Evidence-state additions ──────────────────────────────────────────
    stationarity_classification: str | None = None
    seasonal_strength: float | None = None
    seasonal_selection_provenance: str | None = None
    anomaly_count_adjusted: int | None = None
    anomaly_ratio_adjusted: float | None = None
    change_point_count: int | None = None
    variance_break_count: int | None = None
    trend_effect_size: float | None = None
    trend_p_value_robust: float | None = None
    diagnostic_statuses: dict[str, str] = Field(default_factory=dict)
    diagnostic_warnings: dict[str, list[str]] = Field(default_factory=dict)
    seasonality_candidates: list[int] = Field(default_factory=list)
    observed_frequency: str | None = None
    narrative_label: str = "llm_interpretation"
    narrative_evidence: list[str] = Field(default_factory=list)
    arch_effects: dict[str, Any] = Field(default_factory=dict)
    robust_monotonic_trend: dict[str, Any] = Field(default_factory=dict)
    intermittency: dict[str, Any] = Field(default_factory=dict)
    anomaly_classifications: dict[str, list[int]] = Field(default_factory=dict)


class ModelSelectionResult(BaseModel):
    """Output of the model selection agent.

    Includes fields recording the deterministic selection policy that
    produced this result, so the report can show whether the selection was
    evidence-based or heuristic.
    """

    selected_model: str
    explanation: str
    holt_winters_rejected_reason: str | None = None
    arima_rejected_reason: str | None = None
    sarima_rejected_reason: str | None = None
    ewma_rejected_reason: str | None = None
    reasoning_steps: list[dict[str, Any]] = Field(default_factory=list)
    token_usage: dict[str, Any] = Field(default_factory=dict)
    # ── Selection policy additions ──────────────────────────────────────────
    selection_method: str = "llm"  # "deterministic" | "llm" | "heuristic" | "forced"
    selection_evidence: dict[str, Any] = Field(default_factory=dict)
    narrative_claims: list[dict[str, Any]] = Field(default_factory=list)


class ResidualDiagnostics(BaseModel):
    """Output of residual analysis diagnostics.

    Includes typed fields for error type, interval coverage, and interval
    labelling. The original fields are preserved for backward compatibility
    with the report builder and statistical review agent.
    """

    mean: float
    is_zero_mean: bool | None = None
    ljung_box_p_value: float | None = None
    is_uncorrelated: bool | None = None
    shapiro_wilk_p_value: float | None = None
    is_normal: bool | None = None
    disabled_tests: list[str] = Field(default_factory=list)
    # ── Residual diagnostics additions ──────────────────────────────────────
    error_type: str = "innovations"
    n_errors: int = 0
    mean_ci_lower: float | None = None
    mean_ci_upper: float | None = None
    ljung_box_lag: int | None = None
    ljung_box_df_adjust: int = 0
    variance_by_horizon: dict[int, float] = Field(default_factory=dict)
    interval_coverage: float | None = None
    interval_mean_width: float | None = None
    winkler_score: float | None = None
    weighted_interval_score: float | None = None
    interval_coverage_by_horizon: dict[int, float] = Field(default_factory=dict)
    interval_width_by_horizon: dict[int, float] = Field(default_factory=dict)
    winkler_score_by_horizon: dict[int, float] = Field(default_factory=dict)
    nominal_coverage: float = 0.95
    coverage_estimable: bool = False
    warnings: list[str] = Field(default_factory=list)


class ForecastCandidateResult(BaseModel):
    """Fit status and evaluation evidence for one candidate model."""

    model: str
    status: ForecastFitStatus
    failure_reason: str | None = None
    is_fallback: bool = False
    rmse: float | None = None
    mae: float | None = None
    mape: float | None = None
    wape: float | None = None
    mase: float | None = None
    smape: float | None = None
    rmsse: float | None = None
    n_evaluated: int = 0
    n_missing: int = 0
    fitted_configuration: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    interval_label: str = "prediction_interval"
    validation_design: dict[str, Any] = Field(default_factory=dict)
    metric_intervals: dict[str, list[float]] = Field(default_factory=dict)
    skill_scores: dict[str, float] = Field(default_factory=dict)
    final_test_metrics: dict[str, Any] = Field(default_factory=dict)


class ForecastResult(BaseModel):
    """Output of the forecasting agent for the selected model."""

    model_used: str
    status: ForecastFitStatus
    failure_reason: str | None = None
    is_fallback: bool = False
    forecast: list[float]
    lower_ci: list[float]
    upper_ci: list[float]
    forecast_dates: list[str]
    rmse: float | None = None
    mae: float | None = None
    mape: float | None = None
    wape: float | None = None
    mase: float | None = None
    smape: float | None = None
    rmsse: float | None = None
    residual_diagnostics: ResidualDiagnostics | None = None
    candidate_results: list[ForecastCandidateResult] = Field(default_factory=list)
    reasoning_steps: list[dict[str, Any]] = Field(default_factory=list)
    token_usage: dict[str, Any] = Field(default_factory=dict)
    interval_label: str = "prediction_interval"
    validation_design: dict[str, Any] = Field(default_factory=dict)
    selection_metrics: dict[str, float | None] = Field(default_factory=dict)
    final_test_metrics: dict[str, Any] = Field(default_factory=dict)


class StatisticalReviewResult(BaseModel):
    """Output of the statistical review (QA) agent.

    A critic agent that reviews the outputs of the statistical analysis,
    model selection, and forecasting agents for consistency and correctness.

    Includes fields recording whether the review can override the
    deterministic selection policy and the typed reasons for any override.
    """

    verdict: str  # "pass" | "warn" | "fail"
    flags: list[dict[str, Any]] = Field(default_factory=list)
    endorsements: list[str] = Field(default_factory=list)
    summary: str
    reasoning_steps: list[dict[str, Any]] = Field(default_factory=list)
    token_usage: dict[str, Any] = Field(default_factory=dict)
    # ── Override eligibility additions ──────────────────────────────────────
    can_override_selection: bool = False
    override_reasons: list[str] = Field(default_factory=list)
    narrative_claims: list[dict[str, Any]] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    """Complete response returned by the full analysis pipeline."""

    file_id: str
    validation: ValidationResult
    statistical: StatisticalResult
    model_selection: ModelSelectionResult
    forecast: ForecastResult
    statistical_review: StatisticalReviewResult | None = None
    report: str
    executive_report: dict[str, Any] | None = None
    report_html: str | None = None
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
    chart_historical_png: str = ""  # base64 PNG for PDF export
    chart_stl_png: str = ""  # base64 PNG for PDF export
    chart_forecast_png: str = ""  # base64 PNG for PDF export
    chart_model_comparison_png: str = ""  # base64 PNG for PDF export
    pipeline_token_usage: dict[str, Any] = Field(default_factory=dict)


class JobSubmitResponse(BaseModel):
    """Response returned when an analysis job is enqueued."""

    job_id: str
    status: str  # "pending"


class JobStatusResponse(BaseModel):
    """Current status of an asynchronous analysis job."""

    job_id: str
    # "pending" | "running" | "cancelling" | "done" | "error" | "cancelled"
    status: str
    progress: int  # 0–100
    step: str
    result: dict | None = None
    error: str | None = None
    report_name: str = ""
    source_filename: str = ""
    forecast_horizon: int = 0
    custom_settings_json: str = "[]"


class ForecastJobQueueItem(BaseModel):
    """Administrator-facing summary of a forecast job."""

    job_id: str
    application_username: str
    status: str
    progress: int
    step: str
    forecast_horizon: int
    forced_model: str | None = None
    queued_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


class ForecastJobSettings(BaseModel):
    """Scheduler and history-retention configuration."""

    max_running_jobs_per_user: int = Field(ge=1, le=100)
    max_queued_jobs_per_user: int = Field(default=5, ge=1, le=100)
    retention_days: int | None = Field(default=30, ge=1)
    cleanup_enabled: bool = True


class UserJobQueueItem(BaseModel):
    """User-facing summary of a forecast job for the per-user queue page.

    This DTO is returned by the ``GET /jobs/mine`` endpoint.  It excludes
    ``application_user_id`` (not needed by the browser) and includes
    cancellation availability and report-linkage fields that the frontend
    proxy enriches after looking up the frontend ``forecast_reports`` table.
    """

    job_id: str
    report_name: str
    # "pending" | "running" | "cancelling" | "done" | "error" | "cancelled"
    status: str
    progress: int
    step: str
    queued_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    can_cancel: bool
    forecast_horizon: int = 0
    forced_model: str | None = None
    # The following fields are populated by the frontend proxy after
    # looking up the frontend report database.  The backend always
    # returns them as ``None``/``False``.
    report_id: int | None = None
    report_ready: bool = False
    finalization_error: str | None = None


class DeletedJobsResponse(BaseModel):
    """Count returned after deleting terminal forecast job records."""

    deleted_count: int


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
