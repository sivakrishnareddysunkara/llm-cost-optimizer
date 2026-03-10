from __future__ import annotations

from typing import Any

from .embeddings import EmbeddingProvider
from .pipeline import LLMClient


def _build_openai_client(api_key: str, base_url: str | None = None) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai package is not installed. Install production extras with: pip install -e .[production]"
        ) from exc

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _build_azure_openai_client_with_ad(endpoint: str, api_version: str, aad_scope: str) -> Any:
    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai package is not installed. Install production extras with: pip install -e .[production]"
        ) from exc

    token_provider = _build_azure_ad_token_provider(aad_scope)
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_ad_token_provider=token_provider,
    )


def _build_azure_ad_token_provider(aad_scope: str) -> Any:
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    except ImportError as exc:
        raise RuntimeError(
            "azure-identity package is not installed. Install production extras with: pip install -e .[production]"
        ) from exc

    credential = DefaultAzureCredential()
    return get_bearer_token_provider(credential, aad_scope)


class _ChatAdapterMixin:
    @staticmethod
    def _to_chat_tools(tools: dict[str, dict]) -> list[dict[str, Any]]:
        if not tools:
            return []

        formatted_tools: list[dict[str, Any]] = []
        for name, schema in tools.items():
            parameters = schema.get("parameters") or {"type": "object", "properties": {}}
            formatted_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(schema.get("description", "")).strip() or name,
                        "parameters": parameters,
                    },
                }
            )

        return formatted_tools

    @staticmethod
    def _extract_chat_usage(response: Any) -> tuple[int, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0
        return int(getattr(usage, "prompt_tokens", 0) or 0), int(
            getattr(usage, "completion_tokens", 0) or 0
        )

    @staticmethod
    def _extract_chat_text(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""

        message = getattr(choices[0], "message", None)
        if message is None:
            return ""

        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                text = None
                if isinstance(block, dict):
                    text = block.get("text")
                else:
                    text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
            return "\n".join(parts).strip()

        return str(content).strip()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.client = _build_openai_client(api_key=api_key, base_url=base_url)

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)


class AzureOpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        endpoint: str,
        api_version: str,
        deployment: str,
        aad_scope: str = "https://cognitiveservices.azure.com/.default",
    ) -> None:
        self.deployment = deployment
        self.client = _build_azure_openai_client_with_ad(
            endpoint=endpoint,
            api_version=api_version,
            aad_scope=aad_scope,
        )

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.deployment, input=text)
        return list(response.data[0].embedding)


class OpenAILLMClient(LLMClient, _ChatAdapterMixin):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.client = _build_openai_client(api_key=api_key, base_url=base_url)

    def complete(self, system_prompt: str, user_prompt: str, tools: dict[str, dict]) -> tuple[str, int, int]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=self._to_chat_tools(tools),
            tool_choice="auto" if tools else "none",
        )

        output_text = self._extract_chat_text(response)
        prompt_tokens, completion_tokens = self._extract_chat_usage(response)
        return output_text, prompt_tokens, completion_tokens


class AzureOpenAILLMClient(LLMClient, _ChatAdapterMixin):
    def __init__(
        self,
        endpoint: str,
        api_version: str,
        deployment: str,
        aad_scope: str = "https://cognitiveservices.azure.com/.default",
    ) -> None:
        self.deployment = deployment
        self.client = _build_azure_openai_client_with_ad(
            endpoint=endpoint,
            api_version=api_version,
            aad_scope=aad_scope,
        )

    def complete(self, system_prompt: str, user_prompt: str, tools: dict[str, dict]) -> tuple[str, int, int]:
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=self._to_chat_tools(tools),
            tool_choice="auto" if tools else "none",
        )

        output_text = self._extract_chat_text(response)
        prompt_tokens, completion_tokens = self._extract_chat_usage(response)
        return output_text, prompt_tokens, completion_tokens
