from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .metrics import TokenUsageMonitor
from .prompt_trimmer import Message, PromptTrimmer
from .semantic_cache import CacheHit, SemanticCache


class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str, tools: dict[str, dict]) -> tuple[str, int, int]:
        ...


class CacheBackend(Protocol):
    def lookup(self, query: str, intent: str, endpoint: str) -> CacheHit | None:
        ...

    def store(self, query: str, response: str, intent: str, endpoint: str) -> None:
        ...


@dataclass(frozen=True)
class OptimizedRequest:
    endpoint: str
    user_query: str
    chat_history: list[Message]
    available_tools: dict[str, dict]
    required_tool_names: list[str] | None = None


@dataclass(frozen=True)
class OptimizedResponse:
    text: str
    from_cache: bool
    similarity_score: float | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    intent: str
    selected_tool_names: list[str]


class OptimizerEngine:
    def __init__(
        self,
        llm_client: LLMClient,
        cache: CacheBackend,
        trimmer: PromptTrimmer,
        monitor: TokenUsageMonitor,
    ) -> None:
        self.llm_client = llm_client
        self.cache = cache
        self.trimmer = trimmer
        self.monitor = monitor

    def run(self, request: OptimizedRequest) -> OptimizedResponse:
        trim_result = self.trimmer.trim(
            user_query=request.user_query,
            chat_history=request.chat_history,
            available_tools=request.available_tools,
            required_tool_names=request.required_tool_names,
        )

        cache_query = self._build_cache_query(
            user_query=request.user_query,
            intent=trim_result.intent,
            state_summary=trim_result.state_summary,
        )
        cache_hit = self.cache.lookup(
            query=cache_query,
            intent=trim_result.intent,
            endpoint=request.endpoint,
        )
        if cache_hit is not None:
            self.monitor.record(
                endpoint=request.endpoint,
                prompt_tokens=0,
                completion_tokens=0,
                cache_hit=True,
            )
            return OptimizedResponse(
                text=cache_hit.entry.response,
                from_cache=True,
                similarity_score=cache_hit.score,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                intent=trim_result.intent,
                selected_tool_names=list(trim_result.selected_tools.keys()),
            )

        system_prompt = self._build_system_prompt(
            intent=trim_result.intent,
            state_summary=trim_result.state_summary,
        )
        user_prompt = request.user_query

        response_text, prompt_tokens, completion_tokens = self.llm_client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=trim_result.selected_tools,
        )

        if prompt_tokens <= 0:
            prompt_tokens = self._estimate_tokens(system_prompt + "\n" + user_prompt)

        if completion_tokens <= 0:
            completion_tokens = self._estimate_tokens(response_text)

        self.cache.store(
            query=cache_query,
            response=response_text,
            intent=trim_result.intent,
            endpoint=request.endpoint,
        )
        self.monitor.record(
            endpoint=request.endpoint,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_hit=False,
        )

        return OptimizedResponse(
            text=response_text,
            from_cache=False,
            similarity_score=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            intent=trim_result.intent,
            selected_tool_names=list(trim_result.selected_tools.keys()),
        )

    @staticmethod
    def _build_cache_query(user_query: str, intent: str, state_summary: str) -> str:
        _ = state_summary
        return f"intent={intent}\\nquery={user_query.strip()}"

    @staticmethod
    def _build_system_prompt(intent: str, state_summary: str) -> str:
        base = (
            "You are a helpful assistant. Keep answers concise and correct. "
            f"Current intent: {intent}."
        )
        if state_summary:
            base += f" Relevant conversation state: {state_summary}"
        return base

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4))
