import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
GEMINI_MAX_TOKENS: int = int(os.getenv("GEMINI_MAX_TOKENS", "8192"))

USE_OLLAMA: bool = os.getenv("USE_OLLAMA", "False").lower() == "true"
USE_OLLAMA_CLOUD: bool = os.getenv("USE_OLLAMA_CLOUD", "False").lower() == "true"
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_API_KEY: str | None = os.getenv("OLLAMA_API_KEY")

MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024

ALLOWED_EXTENSIONS: list[str] = os.getenv(
    "ALLOWED_EXTENSIONS", "csv,xlsx"
).split(",")

ALLOWED_MIME_TYPES: list[str] = [
    "text/csv",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
]

CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
