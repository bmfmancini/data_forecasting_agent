"""RAG knowledge base service for the Data Forecaster backend.

Provides a thread-safe singleton accessor for the
:class:`RAGKnowledgeBase` so it is initialised once and shared across
requests.
"""

from __future__ import annotations

import threading

from core.logging_config import get_logger
from rag.knowledge_base import RAGKnowledgeBase

logger = get_logger(__name__)

# Shared RAG knowledge base (initialised once on first access).
_rag_kb: RAGKnowledgeBase | None = None
_rag_lock = threading.Lock()


def get_rag_kb(persist_directory: str = "./chroma_db") -> RAGKnowledgeBase:
    """Return the shared :class:`RAGKnowledgeBase` singleton.

    Initialises the knowledge base on first call and loads documents.
    Subsequent calls return the cached instance.  Thread-safe via a
    module-level lock.

    Args:
        persist_directory: Path to the ChromaDB persistence directory.

    Returns:
        The initialised :class:`RAGKnowledgeBase` instance.
    """
    global _rag_kb  # pylint: disable=global-statement
    if _rag_kb is None:
        with _rag_lock:
            # Double-check inside the lock to avoid duplicate init
            if _rag_kb is None:
                logger.info(
                    "Initialising RAG knowledge base at %s", persist_directory
                )
                _rag_kb = RAGKnowledgeBase(persist_directory=persist_directory)
                _rag_kb.load_documents()
    return _rag_kb