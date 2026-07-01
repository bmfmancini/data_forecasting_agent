# GitHub Copilot Instructions — Time Series Data Forecaster Agent

## Project Overview

A multi-agent time series forecasting system.  
- **Backend**: FastAPI + LangChain agents + statistical forecasting models (ARIMA, SARIMA, Holt-Winters, EWMA)  
- **Frontend**: Flask web application (replaced Streamlit in a prior refactor)  
- **Deployment**: Docker Compose — backend on port `8000`, frontend on port `5000`

---

## Repository Layout

```
data_forecaster/
├── backend/          # FastAPI service
│   ├── agents/       # LangChain-based AI agents
│   ├── core/         # config.py, logging_config.py
│   ├── forecasting/  # arima_model.py, sarima_model.py, holt_winters.py, ewma_model.py
│   ├── prompts/      # LLM prompt strings per agent
│   ├── rag/          # ChromaDB knowledge base + txt/md docs
│   ├── utils/        # data_cleaning, data_parser, schemas, statistical, visualization
│   ├── main.py       # FastAPI app + route definitions
│   └── orchestrator.py
├── frontend/         # Flask application
│   ├── blueprints/
│   │   ├── main/     # Core pages (chat, overview, quality, stats, model, forecast, report)
│   │   │             # and AJAX endpoints (upload, preflight, analyze, job-status, chat …)
│   │   ├── auth/     # /auth/login, /auth/logout
│   │   └── admin/    # User management, API credential management (admin role only)
│   ├── db/           # SQLite helpers (db.py, crypto.py) and schema.sql
│   ├── services/     # BackendAPIClient (api_client.py), PDF export (pdf_service.py)
│   ├── static/       # CSS (main.css) and JS (app.js, charts.js, chat.js, polling.js)
│   ├── app.py        # Application factory (create_app)
│   ├── config.py     # DevelopmentConfig / ProductionConfig / TestingConfig
│   ├── extensions.py # csrf, login_manager singletons
│   ├── models.py     # Flask-Login User model
│   ├── manage.py     # CLI commands
│   └── wsgi.py       # Gunicorn entry point (wsgi:application)
├── data/             # Sample CSVs
├── docker/           # Dockerfile.backend, Dockerfile.flask, docker-compose.yml
└── tests/            # pytest test suite
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI, Uvicorn |
| AI / LLM | LangChain, Google Gemini or Ollama |
| Forecasting | pmdarima, statsmodels |
| Vector DB | ChromaDB |
| Frontend | Flask 3, Flask-Login, Flask-WTF, Flask-Session |
| Frontend auth | Username/password with `werkzeug.security` hashed passwords |
| Frontend DB | SQLite via `db/db.py` helpers (no ORM) |
| PDF export | fpdf2 |
| Deployment | Docker Compose, Gunicorn |
| Python version | 3.11 (backend), 3.12 (frontend Docker image) |

---

## Coding Conventions

### General (all Python files)
- **Python 3.11** compatible syntax throughout.
- **PEP 8** with 4-space indentation and max line length **88** characters.
- **Type hints** on all function parameters and return values.
- **Import order**: standard library → third-party → local; alphabetised within each group; no wildcard imports.
- **Docstrings**: Google style on all public functions, methods, and classes.
- **Logging**: use `utils.logging_config.get_logger` (backend) — never `print()`.
- **Custom exceptions**: raise from `backend.exceptions` for domain validation errors.

### Backend (`backend/`)
- Follow the forecasting instruction file at `.github/instructions/forecasting.instructions.md` for any changes to `backend/forecasting/`.
- Keep agent logic inside `backend/agents/`; prompt strings inside `backend/prompts/`.
- Data schemas live in `backend/utils/schemas.py` and `backend/schemas.py`.

### Frontend (`frontend/`)
- Use the **application factory** pattern — do not import `app` directly; use `current_app`.
- Keep routes inside blueprints; do not add routes to `app.py`.
- All form classes use **Flask-WTF** with CSRF enabled — never disable CSRF.
- Database access goes through `db.db.query_db` / `db.db.execute_db`; no raw `sqlite3` calls outside `db/`.
- Sensitive values (API credentials) are encrypted via `db/crypto.py` before storage.
- Admin-only routes must use the `admin_required` decorator from `blueprints/admin/routes.py`.
- JavaScript assets (`static/js/`) communicate with Flask AJAX endpoints, which proxy requests to the backend.
- Sanitize any HTML rendered from backend responses with `bleach` before passing to templates.

### Testing
- Tests live in `tests/` and are run with `pytest`.
- All public functions must have corresponding unit tests.
- Mock external HTTP calls and LLM calls; do not make real network requests in tests.

---

## Environment Variables

| Variable | Used by | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | backend | Gemini API key |
| `GEMINI_MODEL` | backend | Model name (default `gemini-1.5-flash`) |
| `USE_OLLAMA` | backend | Switch to Ollama instead of Gemini |
| `OLLAMA_BASE_URL` | backend | Ollama server URL |
| `OLLAMA_MODEL` | backend | Ollama model name |
| `MAX_UPLOAD_MB` | both | Max file upload size (default 200) |
| `CHROMA_PERSIST_DIR` | backend | ChromaDB storage path |
| `FLASK_ENV` | frontend | `development` / `production` / `testing` |
| `SECRET_KEY` | frontend | Flask session secret |
| `FLASK_ENCRYPTION_KEY` | frontend | Key for encrypting stored API credentials |
| `BACKEND_URL` | frontend | URL of the FastAPI backend |

---

## Key Architectural Notes

- The Flask frontend is a **thin client**: it delegates all forecasting work to the FastAPI backend via `services/api_client.py` (`BackendAPIClient`).
- The frontend stores a `job_id` in the Flask session and polls `/ajax/job-status/<job_id>` via `static/js/polling.js`.
- User accounts and roles are stored in SQLite (`instance/forecaster.db`). The DB is auto-seeded on first startup.
- The admin blueprint (`/admin/`) allows admin users to create accounts, reset passwords, and configure the backend URL and HTTP Basic Auth credentials.
- The previous Streamlit frontend has been fully replaced by this Flask application. Do not reference Streamlit anywhere in the codebase.
