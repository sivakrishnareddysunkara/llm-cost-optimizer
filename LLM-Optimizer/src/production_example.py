from __future__ import annotations

from llm_optimizer.config import AzureOpenAIConfig, OpenAIConfig, RedisConfig, use_azure_openai_from_env
from llm_optimizer.metrics import TokenUsageMonitor
from llm_optimizer.openai_adapters import (
    AzureOpenAIEmbeddingProvider,
    AzureOpenAILLMClient,
    OpenAIEmbeddingProvider,
    OpenAILLMClient,
)
from llm_optimizer.pipeline import OptimizedRequest, OptimizerEngine
from llm_optimizer.prompt_trimmer import Message, PromptTrimmer
from llm_optimizer.redis_cache import RedisSemanticCache


def main() -> None:
    redis_config = RedisConfig.from_env()

    if use_azure_openai_from_env():
        azure_config = AzureOpenAIConfig.from_env()
        embedding_provider = AzureOpenAIEmbeddingProvider(
            endpoint=azure_config.endpoint,
            api_version=azure_config.api_version,
            deployment=azure_config.embedding_deployment,
            aad_scope=azure_config.aad_scope,
        )
        llm_client = AzureOpenAILLMClient(
            endpoint=azure_config.endpoint,
            api_version=azure_config.api_version,
            deployment=azure_config.chat_deployment,
            aad_scope=azure_config.aad_scope,
        )
        provider_name = "azure-openai"
    else:
        openai_config = OpenAIConfig.from_env()
        embedding_provider = OpenAIEmbeddingProvider(
            api_key=openai_config.api_key,
            model=openai_config.embedding_model,
            base_url=openai_config.base_url,
        )
        llm_client = OpenAILLMClient(
            api_key=openai_config.api_key,
            model=openai_config.chat_model,
            base_url=openai_config.base_url,
        )
        provider_name = "openai"

    cache = RedisSemanticCache(
        embedding_provider=embedding_provider,
        redis_url=redis_config.redis_url,
        similarity_threshold=redis_config.similarity_threshold,
        max_entries=redis_config.max_entries,
        namespace=redis_config.namespace,
        redis_auth_mode=redis_config.auth_mode,
        redis_entra_username=redis_config.entra_username,
        redis_aad_scope=redis_config.entra_scope,
    )

    engine = OptimizerEngine(
        llm_client=llm_client,
        cache=cache,
        trimmer=PromptTrimmer(max_history_messages=8, summary_max_chars=650),
        monitor=TokenUsageMonitor(),
    )

    tools = {
        "refund_lookup": {
            "description": "Retrieve refund state and payout timeline for a customer",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "invoice_id": {"type": "string"},
                },
                "required": ["customer_id"],
            },
        },
        "invoice_reader": {
            "description": "Fetch invoice metadata, line items, and payment status",
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {"type": "string"},
                },
                "required": ["invoice_id"],
            },
        },
    }

    history = [
        Message(role="system", content="You are a billing support agent."),
        Message(role="user", content="I was charged twice and need help."),
        Message(role="assistant", content="I can help with that. Share your customer id."),
    ]

    request = OptimizedRequest(
        endpoint="support/billing",
        user_query="Can you check if my latest invoice has a duplicate charge?",
        chat_history=history,
        available_tools=tools,
    )

    result = engine.run(request)
    print(
        {
            "provider": provider_name,
            "from_cache": result.from_cache,
            "intent": result.intent,
            "selected_tools": result.selected_tool_names,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        }
    )
    print("response:", result.text)

    print("\nweekly usage by endpoint:")
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
