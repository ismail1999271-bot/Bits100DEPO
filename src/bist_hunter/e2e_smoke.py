"""Small deterministic E2E runner used by CI to validate the core pipeline."""
from datetime import datetime, timedelta, timezone

from .pipeline import LabelConfig, add_forward_labels, normalize_ohlcv


def synthetic_rows() -> list[dict]:
    start = datetime(2026, 1, 5, 10, tzinfo=timezone.utc)
    rows = []
    for symbol, closes in {"AAA": [100, 103, 110], "BBB": [100, 101, 102]}.items():
        for i, close in enumerate(closes):
            rows.append({"symbol": symbol, "timestamp": start + timedelta(days=i),
                         "open": closes[max(0, i-1)], "high": close,
                         "low": close * .98, "close": close, "volume": 1000 + i * 100})
    return rows


def run() -> dict[str, int]:
    normalized = normalize_ohlcv(synthetic_rows())
    labeled = add_forward_labels(normalized, LabelConfig(limit_pct=.05, horizon_bars=1))
    return {
        "rows": len(labeled),
        "symbols": labeled["symbol"].nunique(),
        "positive_labels": int(labeled["hit_limit_forward"].sum()),
    }


if __name__ == "__main__":
    print(run())
