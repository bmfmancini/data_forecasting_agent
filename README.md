# Time Series Data Forecaster Agent

A multi-agent system that takes time series data, runs it through statistical forecasting models, and gives you back forecasts with AI-generated analysis and reports.

## What it does

You upload a CSV or Excel file with time series data. Six AI agents then work through the pipeline:

1. **Data Validation** — cleans and validates the data
2. **Statistical Analysis** — runs ACF/PACF, trend/seasonality checks, recommends transformations
3. **Model Selection** — explains candidate models using the data characteristics
4. **Forecasting** — compares enabled models, baselines, ensembles, and training windows on common rolling origins, then produces forecasts and prediction intervals
5. **Statistical Review** — QA agent that reviews the outputs of the previous stages for consistency and correctness
6. **Report Generation** — puts together a report with charts and plain-English insights

Python computes all statistical metrics; the LLM interprets the pre-computed results — it never computes numbers itself.

Models include ARIMA, SARIMA, Holt-Winters, EWMA, Prophet, dynamic regression, and optional intermittent-demand forecasting. An untouched final period evaluates the selected procedure when sufficient history is available. Issued forecasts can be monitored against subsequent actuals through the API.

You interact with all of this through a Flask web UI with auth, an admin panel, and role-based access.

## Quick start

The easiest way to run this is with Docker. You'll need Docker and Docker Compose installed.

```bash
git clone <repository-url>
cd data_forecasting_agent/data_forecaster

# Build and start everything (single-machine mode)
./scripts/build_containers.sh --single
```

That's it. Four containers come up:

| Container | What it does | Port |
|---|---|---|
| `nginx-frontend` | TLS termination for the Flask app | `https://localhost` (443) |
| `frontend` | Flask web UI | internal only |
| `nginx-backend` | TLS termination for the API | `https://localhost:8443` |
| `backend` | FastAPI + forecasting engine | internal only |

Open `https://localhost` in your browser. On first run you'll be redirected to the **setup wizard** (`/setup`), which walks you through: backend connection → LLM provider and credentials → enabling API auth → choosing forecasting models → creating the first admin account. No `.env` secrets are needed — keys are generated and stored encrypted at setup time.

## LLM setup

The agents need an LLM to do their analysis. You can use either Google Gemini or Ollama — configured in the setup wizard or later under **Admin → LLM Config** (keys are stored encrypted in the backend database, never in the frontend).

If you're running Ollama locally, pull the model first: `ollama pull llama3`.

## Project layout

```
data_forecaster/
├── backend/              # FastAPI service
│   ├── agents/           # Pipeline agents
│   ├── auth/             # API key auth (Argon2id)
│   ├── forecasting/      # Models, rolling validation, metrics, uncertainty
│   ├── rag/              # ChromaDB knowledge base
│   └── main.py           # API endpoints
├── frontend/             # Flask web app
│   ├── blueprints/       # Routes (main, auth, admin)
│   ├── db/               # SQLite helpers
│   └── services/         # Backend API client, PDF export
├── docker/               # Compose files, Dockerfiles, nginx configs
├── certs/                # TLS certs (auto-generated or BYO)
└── scripts/              # build_containers.sh and other helpers
```

## Documentation

Detailed docs are split out so this README stays short:

- [Deployment guide](docs/deployment.md) — single-machine vs distributed, TLS certs, SSL verification
- [API authentication](docs/api-auth.md) — how API keys work, rotating credentials, the default `frontend` user
- [API reference](docs/api-reference.md) — endpoint list, error codes, request/response schemas
- [Local development](docs/local-dev.md) — running without Docker, running the test suite
- [Statistical forecasting](docs/statistical-improvements.md) — validation, model additions, monitoring, options, and limitations
- [User management scripts](docs/user-management-scripts.md) — CLI runbook for frontend users and backend API users

## Tech stack

| Layer | Tech |
|---|---|
| Backend | FastAPI, Uvicorn |
| Frontend | Flask, Gunicorn, Flask-Login |
| AI / LLM | LangChain, Google Gemini or Ollama |
| Forecasting | pmdarima, statsmodels |
| Vector DB | ChromaDB |
| Deployment | Docker Compose, Nginx (TLS termination) |
| Python | 3.11 (backend), 3.12 (frontend Docker image) |

## Testing

```bash
# From the repository root, with development dependencies installed
python -m pytest tests/ data_forecaster/tests/
```

## License

GPL v2 — see [LICENSE](LICENSE).

## Acknowledgments

- [Forecasting: Principles and Practice (3rd ed.)](https://otexts.com/fpp3/) by Hyndman & Athanasopoulos — the forecasting methodology this is based on
- [Bala Priya C](https://www.freecodecamp.org/news/author/balapriyac/) — data cleaning techniques that inspired `utils.data_cleaning`
- [Diogo Franquinho](https://diogofranquinho.com/notes/econometrics/time-series-analysis.html) — time series analysis notes that informed the RAG knowledge base
- [Statsmodels](https://www.statsmodels.org/), [Pmdarima](https://alkaline-ml.com/pmdarima/), [LangChain](https://github.com/langchain-ai/langchain), [Flask](https://flask.palletsprojects.com/), [FastAPI](https://fastapi.tiangola.com/)
