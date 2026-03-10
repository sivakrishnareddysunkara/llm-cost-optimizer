from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from .embeddings import EmbeddingProvider, LocalEmbeddingProvider
from .semantic_cache import CacheEntry, CacheHit


class RedisSemanticCache:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        redis_url: str,
        similarity_threshold: float = 0.88,
        max_entries: int = 10000,
        namespace: str = "llm-optimizer",
        redis_auth_mode: str = "key",
        redis_entra_username: str | None = None,
        redis_aad_scope: str = "https://redis.azure.com/.default",
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if max_entries <= 0:
            raise ValueError("max_entries must be greater than zero")

        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.namespace = namespace
        self._redis = self._build_redis_client(
            redis_url=redis_url,
            redis_auth_mode=redis_auth_mode,
            redis_entra_username=redis_entra_username,
            redis_aad_scope=redis_aad_scope,
        )

    def lookup(self, query: str, intent: str, endpoint: str) -> CacheHit | None:
        query_embedding = self.embedding_provider.embed(query)
        payloads = self._redis.lrange(self._key(endpoint=endpoint, intent=intent), 0, -1)

        best_hit: CacheHit | None = None
        for payload in payloads:
            parsed = self._parse_payload(payload)
            if parsed is None:
                continue

            entry = parsed
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
        key = self._key(endpoint=endpoint, intent=intent)
        self._redis.lpush(key, self._serialize_entry(entry))
        self._redis.ltrim(key, 0, self.max_entries - 1)

    def size(self, endpoint: str, intent: str) -> int:
        return int(self._redis.llen(self._key(endpoint=endpoint, intent=intent)))

    def _key(self, endpoint: str, intent: str) -> str:
        return f"{self.namespace}:semantic-cache:{endpoint}:{intent}"

    @staticmethod
    def _serialize_entry(entry: CacheEntry) -> str:
        return json.dumps(
            {
                "query": entry.query,
                "response": entry.response,
                "embedding": entry.embedding,
                "intent": entry.intent,
                "endpoint": entry.endpoint,
                "created_at": entry.created_at.isoformat(),
            }
        )

    @staticmethod
    def _parse_payload(payload: str) -> CacheEntry | None:
        try:
            data = json.loads(payload)
            created_at = datetime.fromisoformat(data["created_at"])
            return CacheEntry(
                query=str(data["query"]),
                response=str(data["response"]),
                embedding=[float(value) for value in data["embedding"]],
                intent=str(data["intent"]),
                endpoint=str(data["endpoint"]),
                created_at=created_at,
            )
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _build_redis_client(
        redis_url: str,
        redis_auth_mode: str,
        redis_entra_username: str | None,
        redis_aad_scope: str,
    ) -> Any:
        try:
            from redis import Redis
        except ImportError as exc:
            raise RuntimeError(
                "redis package is not installed. Install production extras with: pip install -e .[production]"
            ) from exc

        if redis_auth_mode == "key":
            return Redis.from_url(redis_url, decode_responses=True)

        if redis_auth_mode != "entra":
            raise ValueError("redis_auth_mode must be either 'key' or 'entra'")
        if not redis_entra_username:
            raise ValueError("redis_entra_username is required when redis_auth_mode='entra'")

        credential_provider = _RedisEntraCredentialProvider(
            username=redis_entra_username,
            aad_scope=redis_aad_scope,
        )

        try:
            return Redis.from_url(
                redis_url,
                decode_responses=True,
                credential_provider=credential_provider,
            )
        except TypeError:
            # Fallback path for redis clients that do not accept credential_provider in from_url.
            parsed = urlparse(redis_url)
            is_tls = parsed.scheme == "rediss"
            port = parsed.port or (6380 if is_tls else 6379)
            db = int(parsed.path.strip("/") or "0")
            return Redis(
                host=parsed.hostname,
                port=port,
                db=db,
                ssl=is_tls,
                decode_responses=True,
                credential_provider=credential_provider,
            )


class _RedisEntraCredentialProvider:
    def __init__(self, username: str, aad_scope: str) -> None:
        try:
            from azure.identity import DefaultAzureCredential
        except ImportError as exc:
            raise RuntimeError(
                "azure-identity package is not installed. Install production extras with: pip install -e .[production]"
            ) from exc

        self._username = username
        self._scope = aad_scope
        self._credential = DefaultAzureCredential()

    def get_credentials(self) -> tuple[str, str]:
        token = self._credential.get_token(self._scope).token
        return self._username, token
