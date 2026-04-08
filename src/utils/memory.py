"""Local vector-memory utility backed by ChromaDB."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150


def _chunk_text(
    text: str, size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + size)
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


@dataclass
class MemoryResult:
    text: str
    score: float
    metadata: dict[str, Any]


class CrewMemory:
    def __init__(
        self,
        *,
        persist_dir: Path,
        ollama_client: Any,
        embedding_model: str = "nomic-embed-text",
        collection_name: str = "crew_memory",
    ) -> None:
        self._enabled = False
        self._collection: Any = None
        self._ollama_client = ollama_client
        self._embedding_model = embedding_model

        try:
            import chromadb  # type: ignore[import-untyped]
        except Exception:
            return

        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = client.get_or_create_collection(name=collection_name)
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled and self._collection is not None

    def add_artifact(self, *, role: str, task: str, content: str) -> None:
        if not self.enabled:
            return
        chunks = _chunk_text(content)
        if not chunks:
            return
        for idx, chunk in enumerate(chunks):
            vector = self._embed_text(chunk)
            if not vector:
                continue
            digest = hashlib.sha256(f"{role}:{task}:{idx}:{chunk}".encode("utf-8")).hexdigest()
            self._collection.upsert(
                ids=[digest],
                embeddings=[vector],
                documents=[chunk],
                metadatas=[{"role": role, "task": task, "chunk_index": idx}],
            )

    def search(self, *, query: str, limit: int = 3) -> list[MemoryResult]:
        if not self.enabled or not query.strip() or limit <= 0:
            return []
        vector = self._embed_text(query)
        if not vector:
            return []
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=limit,
            include=["documents", "distances", "metadatas"],
        )
        docs = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        if not (len(docs) == len(distances) == len(metadatas)):
            return []
        items: list[MemoryResult] = []
        for doc, distance, meta in zip(docs, distances, metadatas, strict=True):
            if not isinstance(doc, str):
                continue
            score = 1.0 / (1.0 + float(distance)) if isinstance(distance, (float, int)) else 0.0
            items.append(
                MemoryResult(
                    text=doc,
                    score=score,
                    metadata=meta if isinstance(meta, dict) else {},
                )
            )
        return items

    def _embed_text(self, text: str) -> list[float]:
        if not text.strip():
            return []
        try:
            embedding = self._ollama_client.embed(text, model=self._embedding_model)
            if isinstance(embedding, list) and all(isinstance(v, (float, int)) for v in embedding):
                return [float(v) for v in embedding]
        except Exception:
            return []
        return []
