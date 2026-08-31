"""Deterministic limit-up event extraction primitives.

The detector is deliberately small: market-specific price-limit rules belong in a
separate adapter once the authoritative BIST data source is selected.
"""

from dataclasses import dataclass

from .domain import MarketBar


@dataclass(frozen=True, slots=True)
class LimitRule:
    """Price-limit rule expressed as a decimal percentage."""

    limit_pct: float

    def __post_init__(self) -> None:
        if self.limit_pct <= 0:
            raise ValueError("limit_pct must be positive")

    def limit_price(self, reference_price: float) -> float:
        return reference_price * (1 + self.limit_pct)


def is_limit_up_close(bar: MarketBar, reference_price: float, rule: LimitRule, tolerance: float = 1e-6) -> bool:
    """Return whether a bar closed at its theoretical upper price limit."""
    target = rule.limit_price(reference_price)
    return bar.close >= target - tolerance


def first_touch_time(
    bars: list[MarketBar],
    reference_price: float,
    rule: LimitRule,
    tolerance: float = 1e-6,
):
    """Find the first bar touching the theoretical upper limit.

    Bars must be ordered ascending by timestamp. ``None`` means no touch occurred.
    """
    for bar in bars:
        if bar.high >= rule.limit_price(reference_price) - tolerance:
            return bar.timestamp
    return None
