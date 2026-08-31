"""Provider-neutral interfaces for market, news and social data.

Concrete integrations should implement these protocols without leaking provider
logic into the signal engine.
"""

from datetime import datetime
from typing import Protocol, Sequence

from .domain import MarketBar
from .news import NewsEvent


class MarketDataProvider(Protocol):
    def bars(self, symbol: str, start: datetime, end: datetime, interval: str) -> Sequence[MarketBar]: ...

    def symbols(self) -> Sequence[str]: ...


class NewsProvider(Protocol):
    def events(self, start: datetime, end: datetime) -> Sequence[NewsEvent]: ...


class SentimentProvider(Protocol):
    def mentions(self, symbols: Sequence[str], start: datetime, end: datetime) -> dict[str, float]: ...
