"""DB-backed LLM provider configuration with env fallback.

The ``llm_config`` singleton table is the authoritative source of LLM
settings once it exists (i.e. after setup). Before setup — or on installs
that have never written the row — values fall back to the environment
variables in :mod:`core.config`, preserving backward compatibility.

The API key is stored encrypted (Fernet, :mod:`core.secret_store`) and is
only ever decrypted in-process, at call time. Reads are cached keyed on
the row's monotonically increasing ``version`` so writes invalidate
cheaply without a process restart.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass

import core.config as settings
from core import secret_store
from core.database import get_connection
from core.logging_config import get_logger

logger = get_logger(__name__)

_PROVIDER_GEMINI = "gemini"
_PROVIDER_OLLAMA = "ollama"
_PROVIDER_OLLAMA_CLOUD = "ollama_cloud"

_cache: LLMConfig | None = None
_cache_version: int | None = None
_cache_lock = threading.Lock()


@dataclass(frozen=True)
class LLMConfig:
    """Resolved LLM provider configuration.

    Attributes:
        provider: One of ``gemini``, ``ollama``, or ``ollama_cloud``.
        model: Model name (e.g. ``gemini-1.5-flash``, ``llama3``).
        base_url: Provider base URL; ``None`` for Gemini.
        api_key: Decrypted API key; ``None`` when not required or unset.
        temperature: Default sampling temperature.
        version: Row version this config was read from; ``0`` when the
            values came from the environment fallback.
    """

    provider: str
    model: str
    base_url: str | None
    api_key: str | None
    temperature: float
    version: int


def _env_fallback() -> LLMConfig:
    """Build an :class:`LLMConfig` from environment variables."""
    if settings.USE_OLLAMA and settings.USE_OLLAMA_CLOUD:
        return LLMConfig(
            provider=_PROVIDER_OLLAMA_CLOUD,
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            api_key=settings.OLLAMA_API_KEY,
            temperature=settings.GEMINI_TEMPERATURE,
            version=0,
        )
    if settings.USE_OLLAMA:
        return LLMConfig(
            provider=_PROVIDER_OLLAMA,
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            api_key=settings.OLLAMA_API_KEY,
            temperature=settings.GEMINI_TEMPERATURE,
            version=0,
        )
    return LLMConfig(
        provider=_PROVIDER_GEMINI,
        model=settings.GEMINI_MODEL,
        base_url=None,
        api_key=settings.GOOGLE_API_KEY,
        temperature=settings.GEMINI_TEMPERATURE,
        version=0,
    )


def _read_row(connection: sqlite3.Connection) -> sqlite3.Row | None:
    """Return the singleton ``llm_config`` row, or ``None`` if unset."""
    return connection.execute(
        "SELECT provider, model, base_url, encrypted_api_key, temperature,"
        "       version FROM llm_config WHERE singleton = 1"
    ).fetchone()


def get_llm_config(db_path: str | None = None) -> LLMConfig:
    """Return the effective LLM configuration, DB-first with env fallback.

    Results are cached and keyed on the row ``version``; a write via
    :func:`put_llm_config` makes the next call re-read the row.

    Args:
        db_path: Optional database path override (testing).

    Returns:
        The resolved :class:`LLMConfig`.
    """
    global _cache, _cache_version
    with _cache_lock:
        with get_connection(db_path) as connection:
            row = _read_row(connection)
        if row is None:
            return _env_fallback()
        version = int(row["version"])
        if _cache is not None and _cache_version == version:
            return _cache
        encrypted_key = row["encrypted_api_key"]
        api_key = secret_store.decrypt(encrypted_key) if encrypted_key else None
        config = LLMConfig(
            provider=row["provider"],
            model=row["model"],
            base_url=row["base_url"],
            api_key=api_key,
            temperature=float(row["temperature"]),
            version=version,
        )
        _cache = config
        _cache_version = version
        return config


def put_llm_config(
    provider: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    temperature: float,
    db_path: str | None = None,
) -> None:
    """Write the LLM configuration, encrypting the key and bumping version.

    Args:
        provider: One of ``gemini``, ``ollama``, ``ollama_cloud``.
        model: Model name.
        base_url: Provider base URL; ``None`` for Gemini.
        api_key: Plaintext key to encrypt and store; ``None`` keeps the
            existing stored key (or stores none if no row exists yet).
        temperature: Default sampling temperature.
        db_path: Optional database path override (testing).

    Raises:
        ValueError: When the provider is not recognised.
    """
    if provider not in (_PROVIDER_GEMINI, _PROVIDER_OLLAMA, _PROVIDER_OLLAMA_CLOUD):
        raise ValueError(f"Unknown LLM provider: {provider=}")
    if api_key:
        # Idempotent: keeps the existing key when already generated.
        secret_store.generate_and_persist_key()
    encrypted_key = secret_store.encrypt(api_key) if api_key else None
    with get_connection(db_path) as connection:
        if encrypted_key is None:
            connection.execute(
                "INSERT INTO llm_config"
                " (singleton, provider, model, base_url, temperature,"
                "  version, updated_at)"
                " VALUES (1, ?, ?, ?, ?, 1, datetime('now'))"
                " ON CONFLICT(singleton) DO UPDATE SET"
                "   provider = excluded.provider,"
                "   model = excluded.model,"
                "   base_url = excluded.base_url,"
                "   temperature = excluded.temperature,"
                "   version = llm_config.version + 1,"
                "   updated_at = datetime('now')",
                (provider, model, base_url, temperature),
            )
        else:
            connection.execute(
                "INSERT INTO llm_config"
                " (singleton, provider, model, base_url, encrypted_api_key,"
                "  temperature, version, updated_at)"
                " VALUES (1, ?, ?, ?, ?, ?, 1, datetime('now'))"
                " ON CONFLICT(singleton) DO UPDATE SET"
                "   provider = excluded.provider,"
                "   model = excluded.model,"
                "   base_url = excluded.base_url,"
                "   encrypted_api_key = excluded.encrypted_api_key,"
                "   temperature = excluded.temperature,"
                "   version = llm_config.version + 1,"
                "   updated_at = datetime('now')",
                (provider, model, base_url, encrypted_key, temperature),
            )
        connection.commit()
    logger.info(
        "LLM config updated (provider=%s, model=%s, key_updated=%s)",
        provider,
        model,
        api_key is not None,
    )


def is_configured(db_path: str | None = None) -> bool:
    """Return ``True`` when an ``llm_config`` row exists in the DB."""
    with get_connection(db_path) as connection:
        return _read_row(connection) is not None


def reset_cache() -> None:
    """Drop the cached config (test support only)."""
    global _cache, _cache_version
    with _cache_lock:
        _cache = None
        _cache_version = None
