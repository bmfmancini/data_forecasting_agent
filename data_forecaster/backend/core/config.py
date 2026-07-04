import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))

USE_OLLAMA: bool = os.getenv("USE_OLLAMA", "False").lower() == "true"
USE_OLLAMA_CLOUD: bool = os.getenv("USE_OLLAMA_CLOUD", "False").lower() == "true"
# Single base URL for both local and cloud Ollama.  When USE_OLLAMA_CLOUD
# is true, set this to https://ollama.com (or your cloud endpoint).  When
# false, set it to your local Ollama daemon URL.
OLLAMA_BASE_URL: str = os.getenv(
    "OLLAMA_BASE_URL",
    "https://ollama.com" if USE_OLLAMA_CLOUD else "http://host.docker.internal:11434",
)
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_API_KEY: str | None = os.getenv("OLLAMA_API_KEY")

MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024

# Bounds for in-memory storage to prevent OOM restarts.  Each uploaded
# DataFrame is held in process memory; these caps limit how many files
# and jobs can accumulate before the oldest is evicted.
MAX_INMEMORY_FILES: int = int(os.getenv("MAX_INMEMORY_FILES", "50"))
MAX_INMEMORY_JOBS: int = int(os.getenv("MAX_INMEMORY_JOBS", "100"))

ALLOWED_EXTENSIONS: list[str] = os.getenv("ALLOWED_EXTENSIONS", "csv,xlsx").split(",")

ALLOWED_MIME_TYPES: list[str] = [
    "text/csv",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
]

CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# Directory for disk-backed uploaded file storage.  DataFrames are
# persisted as parquet files so they survive process restarts and don't
# consume process memory.  An in-memory metadata index is kept for fast
# lookups; the DataFrame itself is loaded lazily from disk on demand.
FILE_STORAGE_DIR: str = os.getenv("FILE_STORAGE_DIR", "./file_store")

API_KEY_DB_PATH: str = os.getenv("API_KEY_DB_PATH", "./data")
API_KEY_ENABLED: bool = os.getenv("API_KEY_ENABLED", "false").lower() == "true"

# Deployment-time secret used to protect the one-time bootstrap endpoint
# (POST /api-users/bootstrap).  Set this to a strong random value in .env
# and share it with the admin who will enable API authentication.
ADMIN_API_KEY: str | None = os.getenv("ADMIN_API_KEY")

# Pre-shared credentials for the frontend service account.  When both are
# set, the backend auto-creates a ``frontend`` API user on first startup
# (if no users exist yet) and the frontend auto-stores them in its SQLite
# DB (if no credentials are stored yet).  This eliminates the need to
# scrape bootstrap keys from container logs.
#
# Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
FRONTEND_API_USERNAME: str | None = os.getenv("FRONTEND_API_USERNAME")
FRONTEND_API_KEY: str | None = os.getenv("FRONTEND_API_KEY")


def set_api_key_enabled(enabled: bool) -> None:
    """Update ``API_KEY_ENABLED`` at runtime (without restarting).

    Used by the bootstrap endpoint to flip auth on after the first API
    user is created, and by the startup lifespan to re-enable auth when
    the database already contains users from a prior deployment.

    Args:
        enabled: ``True`` to require API key auth, ``False`` for open mode.
    """
    global API_KEY_ENABLED
    API_KEY_ENABLED = enabled


def validate_llm_config() -> None:
    """Validate that at least one LLM provider is properly configured.

    Called at startup to fail fast when the backend cannot reach any LLM
    provider.  Raises ``RuntimeError`` with a descriptive message when
    the configuration is incomplete.

    Raises:
        RuntimeError: When no LLM provider is available.
    """
    if USE_OLLAMA:
        if USE_OLLAMA_CLOUD and not OLLAMA_API_KEY:
            raise RuntimeError(
                "USE_OLLAMA_CLOUD is enabled but OLLAMA_API_KEY is not set. "
                "Configure it in the backend .env file."
            )
    elif not GOOGLE_API_KEY:
        raise RuntimeError(
            "No LLM provider configured. Either set USE_OLLAMA=true with a "
            "running Ollama instance, or set GOOGLE_API_KEY for Gemini."
        )
