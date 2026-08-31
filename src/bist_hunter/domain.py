"""Domain models shared by scanners, backtests and reporting."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class SignalAction(StrEnum):
    WATCH = "WATCH"
    ENTRY = "ENTRY"
    HOLD = "HOLD"
    EXIT = "EXIT"


@dataclass(frozen=True, slots=True)
class MarketBar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class LimitUpEvent:
    symbol: str
    event_time: datetime
    reference_price: float
    limit_price: float
    close_price: float
    first_touch: bool
    held_at_close: bool


@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    timestamp: datetime
    score: float
    action: SignalAction
    confidence: float
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 100:
            raise ValueError("score must be between 0 and 100")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
