from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True, slots=True)
class NormalizedBar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str


@dataclass(frozen=True, slots=True)
class NormalizedNews:
    published_at: datetime
    source: str
    title: str
    url: str
    symbols: tuple[str, ...] = ()
    category: str = "unknown"
    sentiment: float = 0.0
    source_reliability: float = 0.5
    language: Literal["tr", "en", "unknown"] = "unknown"

    def __post_init__(self) -> None:
        if not -1 <= self.sentiment <= 1:
            raise ValueError("sentiment must be between -1 and 1")
        if not 0 <= self.source_reliability <= 1:
            raise ValueError("source_reliability must be between 0 and 1")
