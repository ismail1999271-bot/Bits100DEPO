"""Provider contracts. Concrete licensed BIST/KAP/news adapters plug in here."""
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

from .domain import MarketBar


@dataclass(frozen=True, slots=True)
class NewsItem:
    source: str
    published_at: datetime
    title: str
    url: str
    symbols: tuple[str, ...] = ()
    event_type: str = "unknown"
    sentiment: float = 0.0
    source_reliability: float = 0.5


class MarketDataProvider(Protocol):
    def bars(self, symbols: Sequence[str], start: datetime, end: datetime, interval: str = "1d") -> list[MarketBar]: ...


class NewsProvider(Protocol):
    def news(self, start: datetime, end: datetime) -> list[NewsItem]: ...


def deduplicate_news(items: Sequence[NewsItem]) -> list[NewsItem]:
    """Remove exact/near-exact duplicate headlines while preserving chronology."""
    seen: set[tuple[str, str]] = set()
    result: list[NewsItem] = []
    for item in sorted(items, key=lambda x: x.published_at):
        key = (item.title.casefold().strip(), item.url.strip())
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
