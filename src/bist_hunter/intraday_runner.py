"""Turn real intraday bars into the deterministic HOLD/EXIT/CANCEL decision."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from .intraday import MonitorInput, decide
from .real_adapters import intraday_bars


def monitor_symbol(symbol: str, entry_score: float, entry_price: float, *, now: datetime | None = None) -> str:
    end = now or datetime.now(UTC)
    bars = intraday_bars(symbol, end - timedelta(hours=8), end, interval="5m")
    if not bars:
        return "CANCEL"
    frame = pd.DataFrame(bars).sort_values("timestamp")
    last = frame.iloc[-1]
    close = float(last["close"])
    ret = close / float(entry_price) - 1.0
    # Current score is intentionally market-only here; external event/fund feeds can be merged upstream.
    score = max(0.0, min(100.0, 50.0 + ret * 500.0))
    return decide(MonitorInput(entry_score=entry_score, current_score=score, current_return=ret))
