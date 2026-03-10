from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .embeddings import EmbeddingProvider, LocalEmbeddingProvider


@dataclass(frozen=True)
class CacheEntry:
    query: str
    response: str
    embedding: list[float]
    intent: str
    endpoint: str
    created_at: datetime


@dataclass(frozen=True)
class CacheHit:
    entry: CacheEntry
    score: float


class SemanticCache:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        similarity_threshold: float = 0.88,
        max_entries: int = 5000,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if max_entries <= 0:
            raise ValueError("max_entries must be greater than zero")

        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._entries: list[CacheEntry] = []

    def lookup(self, query: str, intent: str, endpoint: str) -> CacheHit | None:
        query_embedding = self.embedding_provider.embed(query)

        best_hit: CacheHit | None = None
        for entry in self._entries:
            if entry.intent != intent or entry.endpoint != endpoint:
                continue

            score = LocalEmbeddingProvider.cosine_similarity(query_embedding, entry.embedding)
            if score < self.similarity_threshold:
                continue

            if best_hit is None or score > best_hit.score:
                best_hit = CacheHit(entry=entry, score=score)

        return best_hit

    def store(self, query: str, response: str, intent: str, endpoint: str) -> None:
        entry = CacheEntry(
            query=query,
            response=response,
            embedding=self.embedding_provider.embed(query),
            intent=intent,
            endpoint=endpoint,
            created_at=datetime.now(timezone.utc),
        )
        self._entries.append(entry)

        if len(self._entries) > self.max_entries:
            overflow = len(self._entries) - self.max_entries
            self._entries = self._entries[overflow:]

    def size(self) -> int:
        return len(self._entries)
