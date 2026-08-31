"""Deterministic synthetic data for CI and regression tests only."""
from datetime import datetime, timedelta, timezone
import random


def make_dataset(symbols=("AAA", "BBB", "CCC"), days=40, seed=7):
    rng = random.Random(seed)
    rows = []
    start = datetime(2026, 1, 2, 0, tzinfo=timezone.utc)
    for symbol in symbols:
        price = 100.0
        for day in range(days):
            ts = start + timedelta(days=day)
            jump = 0.10 if (day % 13 == 0 and symbol == symbols[0]) else rng.uniform(-0.03, 0.035)
            open_ = price
            close = round(price * (1 + jump), 4)
            high = close if jump >= 0.09 else max(open_, close) * 1.005
            low = min(open_, close) * 0.995
            rows.append({"symbol": symbol, "timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": 100_000 + rng.randint(0, 50_000)})
            price = close
    return rows
