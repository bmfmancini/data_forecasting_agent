from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from core.logging_config import get_logger

logger = get_logger(__name__)

DOCS_DIR = Path(__file__).parent / "docs"
COLLECTION_NAME = "forecasting_methodology"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400      # characters per chunk
CHUNK_OVERLAP = 80    # characters overlapping between consecutive chunks


class RAGKnowledgeBase:
    def __init__(self, persist_directory: str = "./chroma_db") -> None:
        self._persist_directory = persist_directory
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection: Optional[chromadb.Collection] = None
        self._embedder: Optional[SentenceTransformer] = None

    def load_documents(self) -> None:
        """Initialise ChromaDB, load and upsert all .txt docs from docs/."""
        logger.info("Initialising RAG knowledge base at %s", self._persist_directory)

        os.makedirs(self._persist_directory, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self._persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = SentenceTransformer(EMBED_MODEL)

        doc_files = sorted(DOCS_DIR.glob("*.txt"))
        if not doc_files:
            logger.warning("No .txt documents found in %s", DOCS_DIR)
            return

        all_ids: list[str] = []
        all_texts: list[str] = []
        all_metas: list[dict] = []

        for doc_path in doc_files:
            text = doc_path.read_text(encoding="utf-8")
            chunks = _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_path.stem}__{i}"
                all_ids.append(chunk_id)
                all_texts.append(chunk)
                all_metas.append({"source": doc_path.name, "chunk_index": i})
            logger.info("Loaded %d chunks from %s", len(chunks), doc_path.name)

        embeddings = self._embedder.encode(all_texts, show_progress_bar=False).tolist()

        self._collection.upsert(
            ids=all_ids,
            documents=all_texts,
            embeddings=embeddings,
            metadatas=all_metas,
        )
        logger.info("Upserted %d chunks into collection '%s'", len(all_ids), COLLECTION_NAME)

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Return top-k relevant text chunks for the given query."""
        if self._collection is None or self._embedder is None:
            raise RuntimeError("Knowledge base not loaded. Call load_documents() first.")

        query_embedding = self._embedder.encode([query], show_progress_bar=False).tolist()
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents"],
        )
        docs: list[str] = results.get("documents", [[]])[0]
        logger.debug("RAG retrieved %d chunks for query: %.80s", len(docs), query)
        return docs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start += chunk_size - overlap
    return [c for c in chunks if c]
