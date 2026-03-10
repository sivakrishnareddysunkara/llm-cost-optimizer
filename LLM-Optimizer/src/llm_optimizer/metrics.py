from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class TokenEvent:
    endpoint: str
    prompt_tokens: int
    completion_tokens: int
    cache_hit: bool
    timestamp: datetime


@dataclass(frozen=True)
class EndpointUsage:
    endpoint: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    requests: int
    cache_hit_rate: float


class TokenUsageMonitor:
    def __init__(self) -> None:
        self._events: list[TokenEvent] = []

    def record(self, endpoint: str, prompt_tokens: int, completion_tokens: int, cache_hit: bool) -> None:
        self._events.append(
            TokenEvent(
                endpoint=endpoint,
                prompt_tokens=max(prompt_tokens, 0),
                completion_tokens=max(completion_tokens, 0),
                cache_hit=cache_hit,
                timestamp=datetime.now(timezone.utc),
            )
        )

    def weekly_report(self, week_start: datetime | None = None) -> list[EndpointUsage]:
        if week_start is None:
            now = datetime.now(timezone.utc)
            week_start = now - timedelta(days=now.weekday())
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

        weekly_events = [event for event in self._events if event.timestamp >= week_start]

        by_endpoint: dict[str, list[TokenEvent]] = {}
        for event in weekly_events:
            by_endpoint.setdefault(event.endpoint, []).append(event)

        usage: list[EndpointUsage] = []
        for endpoint, events in by_endpoint.items():
            prompt_tokens = sum(event.prompt_tokens for event in events)
            completion_tokens = sum(event.completion_tokens for event in events)
            total_tokens = prompt_tokens + completion_tokens
            requests = len(events)
            cache_hits = sum(1 for event in events if event.cache_hit)
            cache_hit_rate = cache_hits / requests if requests > 0 else 0.0

            usage.append(
                EndpointUsage(
                    endpoint=endpoint,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    requests=requests,
                    cache_hit_rate=cache_hit_rate,
                )
            )

        usage.sort(key=lambda item: item.total_tokens, reverse=True)
        return usage
