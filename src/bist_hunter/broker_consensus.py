"""Broker research consensus from normalized recommendation snapshots."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BrokerView:
    broker: str
    symbol: str
    action: str
    target_price: float | None = None
    stop_price: float | None = None
    observed_at: str = ""


_ACTION_WEIGHT = {"BUY": 1.0, "OVERWEIGHT": 0.8, "HOLD": 0.0, "NEUTRAL": 0.0, "UNDERWEIGHT": -0.8, "SELL": -1.0}


def consensus(views: list[BrokerView]) -> dict[str, float]:
    """Return 0-100 broker consensus, counting each broker once per symbol."""
    grouped: dict[str, dict[str, float]] = {}
    for view in views:
        action = view.action.strip().upper()
        if action not in _ACTION_WEIGHT:
            continue
        grouped.setdefault(view.symbol, {})[view.broker] = _ACTION_WEIGHT[action]
    result: dict[str, float] = {}
    for symbol, broker_scores in grouped.items():
        if not broker_scores:
            continue
        mean = sum(broker_scores.values()) / len(broker_scores)
        result[symbol] = round(50.0 + 50.0 * mean, 2)
    return result
