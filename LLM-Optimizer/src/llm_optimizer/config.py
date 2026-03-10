from __future__ import annotations

import os
from dataclasses import dataclass


def _env_str(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or not value.strip():
        raise ValueError(f"Missing required environment variable: {name}")
    return value.strip()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean")


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    chat_model: str
    embedding_model: str
    base_url: str | None

    @classmethod
    def from_env(cls) -> OpenAIConfig:
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url is not None and not base_url.strip():
            base_url = None

        return cls(
            api_key=_env_str("OPENAI_API_KEY"),
            chat_model=_env_str("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            embedding_model=_env_str("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            base_url=base_url,
        )


@dataclass(frozen=True)
class AzureOpenAIConfig:
    endpoint: str
    api_version: str
    chat_deployment: str
    embedding_deployment: str
    aad_scope: str

    @classmethod
    def from_env(cls) -> AzureOpenAIConfig:
        return cls(
            endpoint=_env_str("AZURE_OPENAI_ENDPOINT"),
            api_version=_env_str("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            chat_deployment=_env_str("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            embedding_deployment=_env_str("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            aad_scope=_env_str("AZURE_OPENAI_AAD_SCOPE", "https://cognitiveservices.azure.com/.default"),
        )


def use_azure_openai_from_env() -> bool:
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return True
    return _env_bool("USE_AZURE_OPENAI", False)


@dataclass(frozen=True)
class RedisConfig:
    namespace: str
    similarity_threshold: float
    max_entries: int
    auth_mode: str
    redis_url: str
    entra_username: str | None
    entra_scope: str

    @classmethod
    def from_env(cls) -> RedisConfig:
        threshold = _env_float("SEMANTIC_CACHE_THRESHOLD", 0.88)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("SEMANTIC_CACHE_THRESHOLD must be between 0.0 and 1.0")

        max_entries = _env_int("SEMANTIC_CACHE_MAX_ENTRIES", 10000)
        if max_entries <= 0:
            raise ValueError("SEMANTIC_CACHE_MAX_ENTRIES must be greater than zero")

        auth_mode = _env_str("REDIS_AUTH_MODE", "key").lower()
        if auth_mode not in {"key", "entra"}:
            raise ValueError("REDIS_AUTH_MODE must be either 'key' or 'entra'")

        if auth_mode == "entra":
            host = _env_str("REDIS_HOST")
            port = _env_int("REDIS_PORT", 6380)
            if port <= 0:
                raise ValueError("REDIS_PORT must be greater than zero")

            db = _env_int("REDIS_DB", 0)
            if db < 0:
                raise ValueError("REDIS_DB must be zero or greater")

            use_tls = _env_bool("REDIS_TLS", True)
            scheme = "rediss" if use_tls else "redis"
            redis_url = f"{scheme}://{host}:{port}/{db}"

            entra_username = _env_str("REDIS_ENTRA_USERNAME")
            entra_scope = _env_str("REDIS_AAD_SCOPE", "https://redis.azure.com/.default")
        else:
            redis_url = _env_str("REDIS_URL", "redis://localhost:6379/0")
            entra_username = None
            entra_scope = "https://redis.azure.com/.default"

        return cls(
            namespace=_env_str("SEMANTIC_CACHE_NAMESPACE", "llm-optimizer"),
            similarity_threshold=threshold,
            max_entries=max_entries,
            auth_mode=auth_mode,
            redis_url=redis_url,
            entra_username=entra_username,
            entra_scope=entra_scope,
        )
