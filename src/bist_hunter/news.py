"""News/KAP ingestion contracts.

Network clients are intentionally adapters: credentials and provider-specific APIs
must be configured separately and never hard-coded in the strategy layer.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class EventType(StrEnum):
    KAP = "KAP"
    NEWS = "NEWS"
    MACRO = "MACRO"
    SOCIAL = "SOCIAL"


@dataclass(frozen=True, slots=True)
class NewsEvent:
    source: str
    published_at: datetime
    title: str
    url: str
    event_type: EventType
    symbols: tuple[str, ...] = ()
    sentiment: float = 0.0
    reliability: float = 0.5

    def __post_init__(self) -> None:
        if not -1 <= self.sentiment <= 1:
            raise ValueError("sentiment must be between -1 and 1")
        if not 0 <= self.reliability <= 1:
            raise ValueError("reliability must be between 0 and 1")


def deduplicate_events(events: list[NewsEvent]) -> list[NewsEvent]:
    """Remove exact duplicate headlines/URLs while preserving chronological order."""
    seen: set[tuple[str, str]] = set()
    result: list[NewsEvent] = []
    for event in sorted(events, key=lambda x: x.published_at):
        key = (event.title.strip().casefold(), event.url.strip())
        if key not in seen:
            seen.add(key)
            result.append(event)
    return result
