from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Iterable

from .schema import NormalizedBar, NormalizedNews


class MarketDataProvider(ABC):
    name = "abstract"

    @abstractmethod
    def daily_bars(self, symbols: Iterable[str], start: date, end: date) -> list[NormalizedBar]:
        raise NotImplementedError


class NewsProvider(ABC):
    name = "abstract"

    @abstractmethod
    def news(self, start: date, end: date) -> list[NormalizedNews]:
        raise NotImplementedError


class KapProvider(NewsProvider):
    """KAP is treated as a first-party disclosure source.

    Network access is intentionally injected into an adapter. This keeps parsing and
    normalization testable and prevents the research engine from depending on a
    fragile website request implementation.
    """

    name = "kap"

    def __init__(self, fetcher=None):
        self.fetcher = fetcher

    def news(self, start: date, end: date) -> list[NormalizedNews]:
        if self.fetcher is None:
            raise RuntimeError("KAP fetcher is not configured")
        return list(self.fetcher(start, end))
