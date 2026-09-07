"""Real-data intraday monitor for paper positions."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from .adapters import BistMarketDataAdapter, parse_json_records
from .intraday import MonitorInput, decide


@dataclass(frozen=True, slots=True)
class MonitorDecision:
    symbol: str
    decision: str
    current_return: float
    observed_at: datetime


def monitor_position(adapter: BistMarketDataAdapter, symbol: str, entry_price: float, entry_score: float) -> MonitorDecision:
    payload = adapter.bars(symbol, "intraday", datetime.now(UTC).date().isoformat())
    rows = parse_json_records(payload)
    if not rows:
        raise ValueError(f"no intraday data for {symbol}")
    latest = rows[-1]
    current_price = float(latest["close"])
    current_score = float(latest.get("score", entry_score))
    current_return = current_price / entry_price - 1.0
    return MonitorDecision(symbol, decide(MonitorInput(entry_score, current_score, current_return)), current_return, datetime.now(UTC))
