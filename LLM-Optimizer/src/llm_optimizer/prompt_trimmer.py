from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class TrimResult:
    intent: str
    selected_history: list[Message]
    state_summary: str
    selected_tools: dict[str, dict]


class PromptTrimmer:
    _intent_keywords = {
        "billing": {"bill", "billing", "invoice", "charge", "refund", "payment"},
        "support": {"error", "issue", "bug", "help", "support", "broken", "fix"},
        "sales": {"pricing", "plan", "enterprise", "quote", "trial", "demo"},
    }

    def __init__(self, max_history_messages: int = 8, summary_max_chars: int = 650) -> None:
        if max_history_messages <= 0:
            raise ValueError("max_history_messages must be greater than zero")
        if summary_max_chars <= 0:
            raise ValueError("summary_max_chars must be greater than zero")

        self.max_history_messages = max_history_messages
        self.summary_max_chars = summary_max_chars

    def trim(
        self,
        user_query: str,
        chat_history: list[Message],
        available_tools: dict[str, dict],
        required_tool_names: list[str] | None = None,
    ) -> TrimResult:
        intent = self.classify_intent(user_query)
        selected_history = self._select_relevant_history(chat_history, intent)
        state_summary = self._summarize_history(selected_history)
        selected_tools = self._select_tools(
            user_query=user_query,
            available_tools=available_tools,
            required_tool_names=required_tool_names,
        )
        return TrimResult(
            intent=intent,
            selected_history=selected_history,
            state_summary=state_summary,
            selected_tools=selected_tools,
        )

    def classify_intent(self, user_query: str) -> str:
        lowered = user_query.lower()

        best_intent = "general"
        best_score = 0
        for intent, keywords in self._intent_keywords.items():
            score = sum(1 for token in keywords if token in lowered)
            if score > best_score:
                best_intent = intent
                best_score = score

        return best_intent

    def _select_relevant_history(self, chat_history: list[Message], intent: str) -> list[Message]:
        if not chat_history:
            return []

        keywords = self._intent_keywords.get(intent, set())

        system_messages = [message for message in chat_history if message.role == "system"]
        relevant_messages = [
            message
            for message in chat_history
            if any(keyword in message.content.lower() for keyword in keywords)
        ]

        recent_messages = chat_history[-self.max_history_messages :]

        merged: list[Message] = []
        seen: set[tuple[str, str]] = set()
        for message in system_messages + relevant_messages + recent_messages:
            key = (message.role, message.content)
            if key in seen:
                continue
            seen.add(key)
            merged.append(message)

        return merged[-self.max_history_messages :]

    def _summarize_history(self, selected_history: list[Message]) -> str:
        if not selected_history:
            return ""

        lines: list[str] = []
        for message in selected_history:
            prefix = "U" if message.role == "user" else "A" if message.role == "assistant" else "S"
            compact = " ".join(message.content.strip().split())
            if len(compact) > 120:
                compact = compact[:117] + "..."
            lines.append(f"{prefix}: {compact}")

        summary = " | ".join(lines)
        if len(summary) > self.summary_max_chars:
            summary = summary[: self.summary_max_chars - 3] + "..."

        return summary

    @staticmethod
    def _select_tools(
        user_query: str,
        available_tools: dict[str, dict],
        required_tool_names: list[str] | None,
    ) -> dict[str, dict]:
        if required_tool_names:
            return {
                tool_name: schema
                for tool_name, schema in available_tools.items()
                if tool_name in set(required_tool_names)
            }

        lowered_query = user_query.lower()
        selected: dict[str, dict] = {}
        for tool_name, schema in available_tools.items():
            description = str(schema.get("description", "")).lower()
            name_tokens = set(tool_name.replace("_", " ").lower().split())
            if any(token in lowered_query for token in name_tokens) or (
                description and any(token in description for token in lowered_query.split())
            ):
                selected[tool_name] = schema

        return selected
