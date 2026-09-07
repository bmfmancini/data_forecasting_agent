# Local Development

If you want to run the frontend and backend without Docker — for debugging, hot reload, or running tests — here's how.

## Prerequisites

- Python 3.11+
- An LLM provider (Google Gemini API key, or Ollama running locally)

## Backend

```bash
cd data_forecasting_agent/data_forecaster/backend

# Create a virtual env and install deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Or with uv (faster):
# uv pip install -r uv.txt

# Start the backend
uvicorn main:app --reload --port 8000
```

The backend is now at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

## Frontend

```bash
cd data_forecasting_agent/data_forecaster/frontend

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Initialize the database
flask --app app init-db

# Start the frontend
flask run --port 5000
```

The frontend is now at `http://localhost:5000`. Log in with `admin` / `admin`.

## Running tests

```bash
cd data_forecasting_agent

# Install development test dependencies (PEP 735 `dev` group in pyproject.toml)
uv pip install --group dev
# Or with plain pip: pip install pytest pytest-asyncio httpx

# All tests
python -m pytest tests/ data_forecaster/tests/

# With verbose output
python -m pytest tests/ data_forecaster/tests/ -v

# A specific test file
python -m pytest tests/test_statistical_improvements.py
```

## LLM setup for development

**Gemini:**
```bash
export GOOGLE_API_KEY=your_key
export USE_OLLAMA=false
```

**Ollama (local):**
```bash
# Install Ollama, then pull a model
ollama pull llama3

export USE_OLLAMA=true
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3
```

The LLM connection test initially allows `http://localhost:11434`,
`http://host.docker.internal:11434`, `https://ollama.com`, and
`https://api.ollama.com`. Administrators can edit these under **Admin → LLM
Config → Manage allowed URLs**, one URL per line. Save the list before testing
a custom server. Changes persist across restarts and apply immediately to new
tests; deleted defaults are not restored. An empty list blocks Ollama tests.
Gemini always uses its fixed provider address.

The test matches the complete base URL (ignoring surrounding whitespace and
trailing slashes) and does not follow redirects. Include a reverse proxy path
in the allowed URL when needed. Saving provider settings or changing
`OLLAMA_BASE_URL` does not grant access to a new destination. Both the UI and
the backend allowlist endpoints require admin access; the latter require valid
admin API credentials even when general API authentication is disabled.

## Useful tips

- The backend uses `--reload` which watches for file changes and auto-restarts. Great for iterating on agents or API endpoints.
- The frontend in development mode has Flask debug enabled — you get the interactive debugger in the browser on errors.
- The frontend generates its session and encryption keys on first startup and stores them under `frontend/instance/` (mode 0600). Back up that directory with the frontend database.
- Backend API credentials are entered in the frontend under Admin -> API Config, not stored in an environment file.
- ChromaDB persists to `./chroma_db` by default. Delete that directory if you want a clean RAG knowledge base.
