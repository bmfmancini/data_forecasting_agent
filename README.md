# Time Series Forecasting  Agent

A multi-agent system that takes time series data, runs it through statistical forecasting models, and gives you back forecasts with AI-generated analysis and reports.

## What it does

Upload a CSV or Excel file, select a date column and numeric target, and choose a forecast horizon. Preflight options control calendar aggregation, missing values, outliers, and business context. Six stages then work through the pipeline:

1. **Data Validation** — audits data quality after calendar preparation and explains preparation choices
2. **Statistical Analysis** — assesses stationarity, trend, seasonality, anomalies, structural changes, and residual characteristics
3. **Model Selection** — proposes an initial model and explains suitability using the data characteristics
4. **Forecasting** — compares enabled models, baselines, ensembles, and training windows on common rolling origins, then produces forecasts and prediction intervals
5. **Statistical Review** — QA agent that reviews the outputs of the previous stages for consistency and correctness
6. **Report Generation** — puts together a report with charts and plain-English insights

Python fits the models and computes forecasts, statistical tests, error metrics, and prediction intervals. The LLM explains the evidence, critiques results, and writes report narratives. In Auto mode it can recommend the decision metric; Python controls the final numerical ranking. Explicit user model and metric choices are supported.

Models include ARIMA, SARIMA, Holt-Winters, EWMA, Prophet, dynamic regression, and optional intermittent-demand forecasting. An untouched final period evaluates the selected procedure when sufficient history is available. Issued forecasts can be monitored against subsequent actuals through the API.

Training-window preprocessing prevents validation observations from influencing imputation, clipping, or transformation fitting. Naive, Seasonal Naive, Mean Forecast, and Drift baselines compete alongside enabled model families, eligible transformed variants, recent-window variants, and a simple ensemble. The selection policy accounts for near ties and retains a baseline when complexity does not provide sufficient improvement.

You interact through a Flask web UI with authentication, an admin panel, and role-based access. Outputs include charts, structured reports, Markdown/HTML rendering, PDF export, and data-explorer chat. See a [sample report](docs/forecast_report_sample_airline_passengers-6.pdf) generated from the airline passengers sample dataset.

Read the [architecture and methodology document](docs/architecture-and-methodology.md) for cleaning formulas, statistical tests, model implementations, selection rules, LLM responsibilities, and privacy boundaries.

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

Configure Google Gemini, Ollama Cloud, or local/self-hosted Ollama in the setup wizard or later under **Admin → LLM Config**. Provider keys are stored encrypted in the backend database. Several stages can fall back to computed summaries when LLM calls fail; a valid provider configuration is still expected for normal application operation.

If you're running Ollama locally, pull the model first: `ollama pull llama3`.

## Data privacy

Local Ollama can keep inference on infrastructure you control. Cloud providers receive prompt content, including statistical/report context and user instructions. Data-explorer chat can also send selected record values and dates. The application does not automatically anonymize this content.

The application provides backend API-key hashing, encrypted stored credentials, ownership checks, and HTTPS through the supplied Nginx deployment. Uploaded datasets are stored as Parquet files without application-level encryption; storage protection and retention across datasets, analysis memory, logs, and backups require deployment configuration. See the [privacy section](docs/architecture-and-methodology.md#7-data-privacy-implemented-controls-and-boundaries) for details.

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
- [Architecture and methodology](docs/architecture-and-methodology.md) — statistical methods, cleaning, supported models, selection, LLM roles, and privacy
- [User management scripts](docs/user-management-scripts.md) — CLI runbook for frontend users and backend API users
- [Sample forecast report](docs/forecast_report_sample_airline_passengers-6.pdf) — full report generated from the airline passengers sample dataset

## Tech stack

| Layer | Tech |
|---|---|
| Backend | FastAPI, Uvicorn |
| Frontend | Flask, Gunicorn, Flask-Login |
| AI / LLM | LangChain, Google Gemini, Ollama Cloud or self-hosted Ollama |
| Forecasting and statistics | pandas, NumPy, SciPy, pmdarima, statsmodels, Prophet |
| Storage and retrieval | SQLite, Parquet, ChromaDB, sentence-transformers |
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
