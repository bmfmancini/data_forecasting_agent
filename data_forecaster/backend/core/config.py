import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY: str = os.environ["GROQ_API_KEY"]
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "1"))
GROQ_MAX_COMPLETION_TOKENS: int = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "8192"))
GROQ_TOP_P: float = float(os.getenv("GROQ_TOP_P", "1"))
GROQ_REASONING_EFFORT: str = os.getenv("GROQ_REASONING_EFFORT", "medium")

MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
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
