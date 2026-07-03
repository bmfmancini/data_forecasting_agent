# Time Series Data Forecaster Agent

An intelligent, multi-agent system for time series forecasting that combines statistical models with AI-powered analysis to provide comprehensive forecasting solutions.

## Overview

The Time Series Data Forecaster Agent is a sophisticated application that automates the entire forecasting pipeline using a multi-agent architecture. It ingests time series data, performs comprehensive analysis, selects optimal forecasting models, generates predictions, and creates detailed reports with actionable insights.

### Key Features

- **Multi-Agent Architecture**: Five specialized agents handle different aspects of the forecasting pipeline
- **Multiple Forecasting Models**: ARIMA, SARIMA, Holt-Winters, and EWMA models
- **AI-Powered Analysis**: Statistical analysis and model selection powered by LLMs
- **Comprehensive Reporting**: Detailed reports with visualizations and business insights
- **Flask Web Interface**: Multi-page Flask application with authentication, admin panel, and role-based access control
- **Docker Deployment**: Containerized application for easy deployment
- **RAG Integration**: Memory-augmented analysis with ChromaDB vector database

## Architecture

![Agent Architecture](/data_forecaster/docs/agent_arch.jpeg)
![Agent Workflow](/data_forecaster/docs/agent_workflow.jpeg)

The system consists of five specialized agents working in sequence:

1. **Data Validation Agent**: Validates and preprocesses input data
2. **Statistical Analysis Agent**: Performs comprehensive statistical analysis
3. **Model Selection Agent**: Selects the optimal forecasting model using AI reasoning
4. **Forecasting Agent**: Generates forecasts using multiple statistical models
5. **Report Generation Agent**: Creates comprehensive reports with insights

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for local development)
- Google API Key (for Gemini models) OR **Ollama** (for local/cloud-based models accessed via Ollama)

> **Important Note on Ollama:** Ollama is currently required for both local and cloud-based model execution (e.g., `gpt-oss:120b-cloud`). You must install Ollama and manually download your model of choice using `ollama pull` before running the pipeline. 
>
> *Future Release Note:* In a near future release, if you are using **Ollama Cloud**, pre-pulling models will no longer be required. However, the `ollama pull` step will always remain mandatory for models running on local hardware.


### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd data_forecaster
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env to add your API keys
   ```

3. Start the application:
   ```bash
   cd docker
   docker-compose up --build
   ```

4. Access the application:
   - Frontend: http://localhost:5000
   - Backend API: http://localhost:8000

### Local Development

1. **Set up Ollama** (if using Ollama-based models):
   - Install Ollama.
   - Pull the required model:
     ```bash
     ollama pull gpt-oss:120b-cloud
     ```

2. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

   > **Note:** The backend also ships a `uv.txt` for [uv](https://github.com/astral-sh/uv)-based
   > installs (used in the Docker image).  Both files contain the same dependencies — use
   > `requirements.txt` with pip, or `uv pip install -r uv.txt` with uv.

3. Install frontend dependencies:
   ```bash
   cd frontend
   pip install -r requirements.txt
   ```

4. Start the backend:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

4. Start the frontend:
   ```bash
   cd frontend
   flask run
   ```
   Or with Gunicorn:
   ```bash
   gunicorn --bind 0.0.0.0:5000 wsgi:application
   ```

4. Access the application:
   - Frontend: http://localhost:5000
   - Backend API: http://localhost:8000

## Project Structure

```
data_forecaster/
├── backend/                 # FastAPI backend service
│   ├── agents/             # Specialized AI agents
│   ├── auth/               # API key authentication (Argon2id)
│   ├── core/               # Configuration and logging
│   ├── forecasting/        # Statistical forecasting models
│   ├── prompts/            # LLM prompts for agents
│   ├── rag/                # RAG knowledge base
│   ├── utils/              # Utility functions
│   ├── main.py             # API endpoints
│   └── orchestrator.py     # Pipeline orchestration
├── frontend/               # Flask web application
│   ├── blueprints/         # Route blueprints (main, auth, admin)
│   │   ├── main/           # Core app pages and AJAX endpoints
│   │   ├── auth/           # Login / logout
│   │   └── admin/          # User & API-credential management
│   ├── db/                 # SQLite helpers and schema
│   ├── services/           # Backend API client and PDF export
│   ├── static/             # CSS and JavaScript assets
│   ├── app.py              # Application factory
│   ├── models.py           # Flask-Login User model
│   ├── config.py           # Environment-based configuration
│   └── wsgi.py             # Gunicorn entry point
├── data/                   # Sample data
├── docker/                 # Docker configuration
└── docs/                   # Documentation
```

## Agents

### 1. Data Validation Agent
- Validates input data format and quality
- Detects missing values, outliers, and inconsistencies
- Prepares data for analysis

### 2. Statistical Analysis Agent
- Performs comprehensive statistical analysis
- Calculates ACF/PACF, trend, seasonality
- Recommends data transformations

### 3. Model Selection Agent
- Evaluates multiple forecasting models
- Selects optimal model based on data characteristics
- Provides reasoning for model selection

### 4. Forecasting Agent
- Implements multiple statistical forecasting models:
  - ARIMA (AutoRegressive Integrated Moving Average)
  - SARIMA (Seasonal ARIMA)
  - Holt-Winters (Triple Exponential Smoothing)
  - EWMA (Exponentially Weighted Moving Average)
- Generates forecasts with confidence intervals

### 5. Report Generation Agent
- Creates comprehensive analysis reports
- Provides business insights and recommendations
- Generates visualizations and charts

## Supported Models

- **ARIMA**: AutoRegressive Integrated Moving Average for non-seasonal data
- **SARIMA**: Seasonal ARIMA for data with seasonal patterns
- **Holt-Winters**: Triple exponential smoothing for trend and seasonality
- **EWMA**: Exponentially Weighted Moving Average for simple forecasting

## API Endpoints

### Backend (http://localhost:8000)

**Public endpoints (no authentication):**

- `GET /health` - Health check

**Protected endpoints (require `X-API-Username` and `X-API-Key` headers):**

- `POST /upload` - Upload time series data
- `POST /preflight` - Run data quality checks
- `POST /analyze` - Start forecasting analysis
- `GET /jobs/{job_id}` - Get job status and results
- `POST /chat` - Chat with the analysis results

**API key management endpoints (require authentication):**

- `GET /api-users` - List all API key users
- `POST /api-users` - Create a new API key user
- `POST /api-users/{id}/rotate` - Rotate an API user's key
- `POST /api-users/{id}/toggle` - Enable or disable an API user
- `DELETE /api-users/{id}` - Delete an API user
- `GET /api-users/bootstrap-status` - Check if a bootstrap user exists

### API Error Responses

All errors return a JSON object with a `detail` field containing a human-readable message:

```json
{"detail": "Unauthorized"}
```

Common status codes:

| Status | Meaning | When |
|--------|---------|------|
| `400` | Bad Request | Invalid file format, unsupported extension, file too large, empty file, invalid preflight options |
| `401` | Unauthorized | Missing or invalid `X-API-Username` / `X-API-Key` headers, or disabled account |
| `403` | Forbidden | Missing or invalid `X-Admin-Key` on the bootstrap endpoint |
| `404` | Not Found | Unknown `file_id` or `job_id`, or session data evicted from memory |
| `409` | Conflict | Bootstrap attempted when users already exist; duplicate username on create |
| `422` | Unprocessable Entity | Pydantic validation failure (e.g. chat query exceeds 2000 characters) |
| `500` | Internal Server Error | Unexpected server-side failure (generic message; details logged server-side) |
| `503` | Service Unavailable | Background worker not ready, or backend unreachable from frontend |

> **Interactive docs:** FastAPI auto-generates OpenAPI/Swagger documentation at
> [`http://localhost:8000/docs`](http://localhost:8000/docs) and ReDoc at
> [`http://localhost:8000/redoc`](http://localhost:8000/redoc).  These include request/response
> schemas for every endpoint.

## Configuration

Environment variables can be set in the `.env` file:

```bash
# LLM Configuration
GOOGLE_API_KEY=your_google_api_key
GEMINI_MODEL=gemini-1.5-flash
USE_OLLAMA=False
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3

# File Upload Settings
MAX_UPLOAD_MB=200
ALLOWED_EXTENSIONS=csv,xlsx

# Storage
CHROMA_PERSIST_DIR=./chroma_db

# API Key Authentication (backend)
API_KEY_ENABLED=true                      # set to false to disable auth (dev only)
API_KEY_DB_PATH=./data                     # directory for the API key SQLite DB

# Flask Frontend
FLASK_ENV=production
SECRET_KEY=your-secret-key
FLASK_ENCRYPTION_KEY=your-encryption-key   # used to encrypt stored API credentials
```

### First-run setup

On first startup the Flask app auto-seeds the SQLite database (`instance/forecaster.db`) with two default roles (`admin`, `user`).  Create users and configure the backend URL via the **Admin** panel at `/admin/` (accessible to accounts with the `admin` role).

## API Key Authentication

The FastAPI backend requires API key authentication for all protected endpoints.  API keys are hashed with **Argon2id** (via `argon2-cffi`) and stored in a dedicated SQLite database (`api_keys.db`) inside the backend container.  Plaintext keys are never stored — they are displayed only once at creation or rotation time.

### How It Works

1. **Request headers**: Clients send `X-API-Username` and `X-API-Key` headers with every request to a protected endpoint.
2. **Validation**: The backend looks up the username in SQLite, checks the account is enabled, and verifies the supplied key against the stored Argon2id hash.
3. **Audit trail**: On successful authentication, the backend updates the `last_used` and `last_used_ip` columns.
4. **Failure handling**: Any authentication failure returns a generic `401 Unauthorized` — the error never reveals whether the username or the key was invalid.

### Bootstrap API User

On first startup (when the API key database is empty), the backend automatically creates a bootstrap API user:

- **Username**: `frontend`
- **API Key**: A cryptographically secure random string generated with `secrets.token_urlsafe(32)`
- **Bootstrap flag**: `true` — the admin UI displays a warning until this account is replaced or removed

The plaintext key is printed **once** to the backend container's stdout in a banner:

```
========================================

Initial API Credentials Created

Username:
frontend

API Key:
xxxxxxxxxxxxxxxxxxxxxxxx

This key will only be displayed once.

Log into the Admin panel and rotate
or replace this credential.

========================================
```

Retrieve it with:

```bash
docker logs <backend-container-name> 2>&1 | grep -A 20 "Initial API Credentials"
```

On subsequent restarts, the bootstrap check finds existing users and does nothing — credentials are never regenerated.  The API key database is persisted via a Docker volume (`api_key_data`) so it survives container restarts.

### Initial Deployment Steps

1. **Start the stack**: `docker-compose up --build`
2. **Read the bootstrap credentials** from the backend logs (see above).
3. **Log into the Flask admin panel** at `http://localhost:5000` (default: `admin` / `admin`).
4. **Go to Admin → API Config** and enter:
   - **Backend API Base URL**: `http://backend:8000`
   - **API Username**: `frontend`
   - **API Key**: *(the bootstrap key from the logs)*
   - Click **Save Configuration**.
5. **Go to Admin → API Keys** — you will see the `frontend` bootstrap user with a ⚠ badge.
6. **Create a replacement API user** — click "+ Create API User", enter a username and description.  The plaintext key is displayed once — **copy it immediately**.
7. **Update the API Config** with the new username and key.
8. **Delete the bootstrap account** — return to API Keys and delete the `frontend` user.  The ⚠ warning disappears from the dashboard.

### Admin API Key Management

The Flask admin panel (`/admin/api-keys`) provides full CRUD for API key users:

| Action | Description |
|---|---|
| **List** | Shows username, description, enabled status, bootstrap badge, created date, last used timestamp, and last used IP.  Never displays the key or hash. |
| **Create** | Generates a new API key, hashes it with Argon2id, stores the hash, and displays the plaintext key once. |
| **Rotate** | Generates a new key, replaces the stored hash, and displays the new plaintext key once.  The old key is invalidated immediately. |
| **Enable / Disable** | Toggles the `enabled` flag.  Disabled users cannot authenticate. |
| **Delete** | Permanently removes the API user. |

### How the Frontend Authenticates

The Flask frontend stores the API username and key (Fernet-encrypted) in its own `api_credentials` SQLite table.  On every request to the backend, the `BackendAPIClient` decrypts the stored credentials and sends them as `X-API-Username` and `X-API-Key` headers.  The plaintext key never exists in the frontend's memory beyond the duration of a single request.

### Rotating Compromised Credentials

If an API key is compromised:

1. Go to **Admin → API Keys**.
2. Click 🔄 (rotate) on the affected user.
3. Copy the new plaintext key (displayed once).
4. Go to **Admin → API Config** and update the stored credentials with the new key.
5. The old key is immediately invalid — any client still using it will receive `401 Unauthorized`.

### Disabling Authentication (Development Only)

Set `API_KEY_ENABLED=false` in the backend environment to make the `require_api_key` dependency a no-op.  This is intended for local development only — **never disable authentication in production**.

## Testing

Run backend tests:
```bash
cd backend
python -m pytest tests/
```

## Documentation

- [Forecasting Best Practices](docs/forecasting_best_practices.txt)
- [ARIMA Model Guide](docs/arima.txt)
- [SARIMA Model Guide](docs/sarima.txt)
- [Holt-Winters Model Guide](docs/holt_winters.txt)



## License

This project is licensed under the GNU General Public License v2.0  License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Bala Priya C](https://www.freecodecamp.org/news/author/balapriyac/) for the data cleaning methods in the freeCodeCamp post [*How to Clean Time Series Data in Python*](https://www.freecodecamp.org/news/how-to-clean-time-series-data-in-python/) that inspired the `utils.data_cleaning` module

- [Forecasting: Principles and Practice (3rd ed.)](https://otexts.com/fpp3/) by Hyndman & Athanasopoulos — the canonical reference for the ARIMA, SARIMA and exponential-smoothing methodology implemented in the `backend/forecasting` package

- [Diogo Franquinho](https://diogofranquinho.com/notes/econometrics/time-series-analysis.html) for the concise *Time Series Analysis* technical notes covering stationarity, ACF, ARIMA and AIC-based model selection that informed the RAG knowledge base (`backend/rag/docs/`)
As well as his Udemy courses!

Other resources used

- [Statsmodels](https://www.statsmodels.org/) for statistical modeling
- [Pmdarima](https://alkaline-ml.com/pmdarima/) for ARIMA modeling
- [Langchain](https://github.com/langchain-ai/langchain) for LLM integration
- [Flask](https://flask.palletsprojects.com/) for the frontend framework
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
