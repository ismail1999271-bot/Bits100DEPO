"""News/social event intelligence and manipulation-risk controls."""
from dataclasses import dataclass
from datetime import datetime
import re

from .news import NewsEvent, deduplicate_events


@dataclass(frozen=True, slots=True)
class SourceProfile:
    name: str
    reliability: float
    is_official: bool = False


@dataclass(frozen=True, slots=True)
class EventSignal:
    source: str
    symbols: tuple[str, ...]
    sentiment: float
    reliability: float
    manipulation_risk: float
    published_at: datetime


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def source_adjusted_event(event: NewsEvent, profile: SourceProfile, duplicate_count: int = 0) -> EventSignal:
    """Discount low-quality, duplicated and unusually promotional signals."""
    text = event.title.casefold()
    hype_terms = len(re.findall(r"tavan|uçacak|roket|garanti|kesin", text))
    duplicate_risk = min(0.35, duplicate_count * 0.05)
    hype_risk = min(0.35, hype_terms * 0.08)
    risk = _clamp((1.0 - profile.reliability) * 0.45 + duplicate_risk + hype_risk)
    return EventSignal(
        event.source,
        tuple(sorted(set(event.symbols))),
        max(-1.0, min(1.0, event.sentiment)),
        _clamp(event.reliability * profile.reliability),
        risk,
        event.published_at,
    )


def normalize_event_batch(events: list[NewsEvent], profiles: dict[str, SourceProfile]) -> list[EventSignal]:
    unique = deduplicate_events(events)
    counts: dict[str, int] = {}
    for event in events:
        key = event.title.casefold().strip()
        counts[key] = counts.get(key, 0) + 1
    result = []
    for event in unique:
        profile = profiles.get(event.source, SourceProfile(event.source, 0.25))
        result.append(source_adjusted_event(event, profile, counts.get(event.title.casefold().strip(), 1) - 1))
    return sorted(result, key=lambda x: x.published_at)
