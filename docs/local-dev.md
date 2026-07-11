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

# Set up environment variables
cp ../.env .  # or symlink it

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

# Point the frontend at the local backend
export BACKEND_URL=http://localhost:8000
export API_VERIFY_SSL=false
export FLASK_ENV=development
export SECRET_KEY=dev-secret
export FLASK_ENCRYPTION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")

# Initialize the database
flask --app app init-db

# Start the frontend
flask run --port 5000
```

The frontend is now at `http://localhost:5000`. Log in with `admin` / `admin`.

## Running tests

```bash
cd data_forecasting_agent/data_forecaster

# All tests
python -m pytest tests/

# With verbose output
python -m pytest tests/ -v

# A specific test file
python -m pytest tests/test_zscore_outliers.py
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
- The `FLASK_ENCRYPTION_KEY` is used to encrypt stored API credentials at rest. Generate one with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`.
- If you change the `FRONTEND_API_KEY` in `.env`, delete the stale `backend.db` and restart the backend — it only auto-creates the user when no users exist.
- ChromaDB persists to `./chroma_db` by default. Delete that directory if you want a clean RAG knowledge base.
