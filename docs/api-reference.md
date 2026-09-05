# API Reference

The FastAPI backend runs on `https://localhost:8443` in the default Docker setup. Interactive Swagger docs are available at `/docs` and ReDoc at `/redoc` — those are always up to date with the code.

## Public endpoints

### `GET /health`

Health check. Returns `{"status": "ok"}`. No auth required.

### `GET /auth-status`

Returns whether API auth is enabled and whether any users exist. No auth required.

```json
{"auth_enabled": true, "has_users": true}
```

## Protected endpoints

All of these require `X-API-Username` and `X-API-Key` headers.

### `POST /upload`

Upload a CSV or XLSX file for forecasting.

**Request:** `multipart/form-data` with a `file` field.

**Response:**
```json
{"file_id": "abc123", "filename": "data.csv", "rows": 144, "columns": ["date", "value"]}
```

### `POST /preflight`

Run data quality checks on an uploaded file before forecasting.

**Request:**
```json
{
  "file_id": "abc123",
  "forecast_horizon": 12,
  "date_col": "date",
  "value_col": "value"
}
```

**Response:** Quality metrics, missing value counts, outlier detection results, and recommended transformations.

### `POST /analyze`

Start a forecasting job. Returns a `job_id` that you poll with `/jobs/{job_id}`.

**Request:**
```json
{
  "file_id": "abc123",
  "forecast_horizon": 12,
  "date_col": "date",
  "value_col": "value"
}
```

**Response:**
```json
{"job_id": "xyz789", "status": "queued"}
```

### `GET /jobs/{job_id}`

Poll job status. Returns the current state and, when complete, the full results.

**Response (in progress):**
```json
{"job_id": "xyz789", "status": "running", "progress": "Forecasting..."}
```

**Response (complete):**
```json
{
  "job_id": "xyz789",
  "status": "completed",
  "results": {
    "forecast": [...],
    "lower_ci": [...],
    "upper_ci": [...],
    "rmse": 2.34,
    "mae": 1.87,
    "mape": 5.2,
    "model": "ARIMA(1,1,1)",
    "report": "..."
  }
}
```

### Forecast monitoring

| Method | Path | Description |
|---|---|---|
| `POST` | `/jobs/{job_id}/actuals` | Record or correct observed values as a JSON timestamp-to-number mapping; return updated accuracy |
| `GET` | `/jobs/{job_id}/monitoring` | Read accuracy, bias, baseline skill, coverage, and horizon scores for an issued forecast |

Both endpoints require authentication and job ownership (or admin access). Predictions are immutable. Unknown jobs and jobs without snapshots return `404`; invalid actuals return `422`. See [statistical forecasting](statistical-improvements.md) for examples, model options, and interpretation.

### `POST /chat`

Ask a follow-up question about the analysis results.

**Request:**
```json
{"job_id": "xyz789", "query": "Why was ARIMA chosen over Holt-Winters?"}
```

**Response:**
```json
{"answer": "ARIMA was selected because..."}
```

## API key management

These endpoints also require auth. They let you manage API users from the admin panel.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api-users` | List all API users (never includes key hashes) |
| `POST` | `/api-users` | Create a new API user (returns plaintext key once) |
| `POST` | `/api-users/{id}/rotate` | Rotate a user's key (returns new plaintext key once) |
| `POST` | `/api-users/{id}/toggle` | Enable or disable a user |
| `DELETE` | `/api-users/{id}` | Delete a user |

## Setup wizard

First-run provisioning. Unauthenticated; the bootstrap endpoint is guarded by an atomic "no users exist yet" check (no preset admin token).

| Method | Path | Description |
|---|---|---|
| `GET` | `/setup/status` | Setup state flags (`setup_complete`, `admin_exists`, `llm_configured`, `models_enabled`) — never secrets |
| `POST` | `/setup/bootstrap` | One-time atomic bootstrap: creates the first admin API user, generates the backend encryption key, enables auth. `409` once any user exists |

## LLM configuration (admin)

| Method | Path | Description |
|---|---|---|
| `GET` | `/config/llm` | Masked LLM config (`provider`, `model`, `base_url`, `temperature`, `api_key_set`) — the key is never returned |
| `PUT` | `/config/llm` | Update LLM config; `api_key` is write-only (omit to keep the stored key) |

## Model registry (admin)

| Method | Path | Description |
|---|---|---|
| `GET` | `/models` | List the seven forecasting models with enabled state |
| `PUT` | `/models/{name}` | Enable/disable a model; `400` when disabling the last enabled model |

## Error responses

All errors return JSON with a `detail` field:

```json
{"detail": "Unauthorized"}
```

| Status | When |
|---|---|
| `400` | Bad file, unsupported extension, file too large, empty file, bad preflight options |
| `401` | Missing or invalid API key headers, or disabled account |
| `409` | `/setup/bootstrap` called after setup completed |
| `404` | Unknown `file_id` or `job_id` |
| `409` | Duplicate username, or bootstrap attempted when users already exist |
| `422` | Pydantic validation failure (e.g. chat query too long) |
| `500` | Unexpected server error (details logged server-side) |
| `503` | Worker not ready, or backend unreachable from frontend |
