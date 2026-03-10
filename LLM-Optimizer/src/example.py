from __future__ import annotations

from llm_optimizer.embeddings import LocalEmbeddingProvider
from llm_optimizer.metrics import TokenUsageMonitor
from llm_optimizer.pipeline import LLMClient, OptimizedRequest, OptimizerEngine
from llm_optimizer.prompt_trimmer import Message, PromptTrimmer
from llm_optimizer.semantic_cache import SemanticCache


class MockLLMClient(LLMClient):
    def complete(self, system_prompt: str, user_prompt: str, tools: dict[str, dict]) -> tuple[str, int, int]:
        tool_suffix = f" Tools used: {', '.join(tools)}." if tools else ""
        response = f"Resolved request: {user_prompt}.{tool_suffix}"
        prompt_tokens = max(1, (len(system_prompt) + len(user_prompt)) // 4)
        completion_tokens = max(1, len(response) // 4)
        return response, prompt_tokens, completion_tokens


def main() -> None:
    engine = OptimizerEngine(
        llm_client=MockLLMClient(),
        cache=SemanticCache(embedding_provider=LocalEmbeddingProvider(), similarity_threshold=0.60),
        trimmer=PromptTrimmer(max_history_messages=6, summary_max_chars=400),
        monitor=TokenUsageMonitor(),
    )

    tools = {
        "refund_lookup": {"description": "Find refund status and policy details."},
        "invoice_reader": {"description": "Read invoice records by invoice ID."},
        "account_notes": {"description": "Get account-level support notes."},
    }

    history = [
        Message(role="system", content="You support billing and refunds."),
        Message(role="user", content="My last invoice looks incorrect."),
        Message(role="assistant", content="Please share your invoice ID."),
    ]

    requests = [
        "Can you check why I was charged twice this month?",
        "I think there is a duplicate charge on my latest bill, can you verify it?",
    ]

    for query in requests:
        result = engine.run(
            OptimizedRequest(
                endpoint="support/billing",
                user_query=query,
                chat_history=history,
                available_tools=tools,
            )
        )
        print(
            {
                "query": query,
                "from_cache": result.from_cache,
                "intent": result.intent,
                "selected_tools": result.selected_tool_names,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "similarity_score": result.similarity_score,
            }
        )

    print("\\nWeekly token report:")
    for row in engine.monitor.weekly_report():
        print(
            {
                "endpoint": row.endpoint,
                "requests": row.requests,
                "total_tokens": row.total_tokens,
                "cache_hit_rate": round(row.cache_hit_rate, 2),
            }
        )


if __name__ == "__main__":
    main()
