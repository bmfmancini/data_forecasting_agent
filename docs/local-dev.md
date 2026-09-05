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

# Install development test dependencies in your activated environment
pip install 'pytest==8.3.*' 'pytest-asyncio==0.24.*'

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

## Useful tips

- The backend uses `--reload` which watches for file changes and auto-restarts. Great for iterating on agents or API endpoints.
- The frontend in development mode has Flask debug enabled — you get the interactive debugger in the browser on errors.
- The frontend generates its session and encryption keys on first startup and stores them under `frontend/instance/` (mode 0600). Back up that directory with the frontend database.
- Backend API credentials are entered in the frontend under Admin -> API Config, not stored in an environment file.
- ChromaDB persists to `./chroma_db` by default. Delete that directory if you want a clean RAG knowledge base.
