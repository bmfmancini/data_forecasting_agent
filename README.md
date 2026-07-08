# Time Series Data Forecaster Agent

A multi-agent system that takes time series data, runs it through statistical forecasting models, and gives you back forecasts with AI-generated analysis and reports.

## What it does

You upload a CSV or Excel file with time series data. Six AI agents then work through the pipeline:

1. **Data Validation** — cleans and validates the data
2. **Statistical Analysis** — runs ACF/PACF, trend/seasonality checks, recommends transformations
3. **Model Selection** — picks the best model based on the data characteristics
4. **Forecasting** — runs ARIMA, SARIMA, Holt-Winters, or EWMA and generates predictions with confidence intervals
5. **Statistical Review** — QA agent that reviews the outputs of the previous stages for consistency and correctness
6. **Report Generation** — puts together a report with charts and plain-English insights

Python computes all statistical metrics; the LLM interprets the pre-computed results — it never computes numbers itself.

You interact with all of this through a Flask web UI with auth, an admin panel, and role-based access.

## Quick start

The easiest way to run this is with Docker. You'll need Docker and Docker Compose installed.

```bash
git clone <repository-url>
cd data_forecasting_agent/data_forecaster

# The .env file already has sensible defaults — just edit the LLM section
# to add your API key

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

Open `https://localhost` in your browser. Log in with `admin` / `admin` (you'll be prompted to change the password). The default API credentials (`frontend` / `frontend`) are already configured, so the frontend can talk to the backend out of the box.

> **Heads up:** The default `frontend` API key is publicly known. Rotate it before exposing this to anything beyond your local machine. See [docs/api-auth.md](docs/api-auth.md) for how.

## LLM setup

The agents need an LLM to do their analysis. You can use either Google Gemini or Ollama.

**Gemini** (easiest — just add your key):
```bash
# In .env
GOOGLE_API_KEY=your_key_here
USE_OLLAMA=false
```

**Ollama** (runs locally or via Ollama Cloud):
```bash
# In .env
USE_OLLAMA=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

If you're running Ollama locally, pull the model first: `ollama pull llama3`.

## Project layout

```
data_forecaster/
├── backend/              # FastAPI service
│   ├── agents/           # The five AI agents
│   ├── auth/             # API key auth (Argon2id)
│   ├── forecasting/      # ARIMA, SARIMA, Holt-Winters, EWMA
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
cd data_forecaster
python -m pytest tests/
```

## License

GPL v2 — see [LICENSE](LICENSE).

## Acknowledgments

- [Forecasting: Principles and Practice (3rd ed.)](https://otexts.com/fpp3/) by Hyndman & Athanasopoulos — the forecasting methodology this is based on
- [Bala Priya C](https://www.freecodecamp.org/news/author/balapriyac/) — data cleaning techniques that inspired `utils.data_cleaning`
- [Diogo Franquinho](https://diogofranquinho.com/notes/econometrics/time-series-analysis.html) — time series analysis notes that informed the RAG knowledge base
- [Statsmodels](https://www.statsmodels.org/), [Pmdarima](https://alkaline-ml.com/pmdarima/), [LangChain](https://github.com/langchain-ai/langchain), [Flask](https://flask.palletsprojects.com/), [FastAPI](https://fastapi.tiangola.com/)
