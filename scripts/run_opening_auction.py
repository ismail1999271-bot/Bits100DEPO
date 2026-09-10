"""Run the 09:40-09:55 opening-auction collector in fail-closed mode.

The provider must supply the licensed real-time/order-book observations. This
runner deliberately refuses to continue when the market feed is absent.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from time import sleep
from zoneinfo import ZoneInfo

from bist_hunter.live_data_stack import LiveDataGateway
from bist_hunter.opening_auction import AuctionSnapshot, score_opening_auction


TZ = ZoneInfo("Europe/Istanbul")
CHECK_MINUTES = (0, 5, 10, 15)


def _float(record: dict, *names: str, default: float | None = None) -> float | None:
    for name in names:
        value = record.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return default


def main() -> int:
    gateway = LiveDataGateway()
    symbols = [s.strip().upper() for s in os.getenv("BIST_SYMBOLS", "").split(",") if s.strip()]
    if not symbols:
        raise SystemExit("LIVE_DATA_BLOCKED: BIST_SYMBOLS is not configured")

    now = datetime.now(TZ)
    for minute in CHECK_MINUTES:
        target = now.replace(hour=9, minute=40, second=0, microsecond=0) + timedelta(minutes=minute)
        if now < target:
            sleep((target - now).total_seconds())
        observed_at = datetime.now(TZ).isoformat()
        for symbol in symbols:
            envelope = gateway.snapshot(symbol, observed_at, observed_at, observed_at)
            if not envelope.market:
                continue
            row = envelope.market[-1]
            required = {
                "indicative_price": _float(row, "indicative_price", "auction_price"),
                "reference_price": _float(row, "reference_price", "previous_close"),
                "bid_qty": _float(row, "bid_qty", "total_bid_qty"),
                "ask_qty": _float(row, "ask_qty", "total_ask_qty"),
            }
            if any(value is None for value in required.values()):
                print(f"{symbol}: BLOCKED_MISSING_AUCTION_FIELDS")
                continue
            signal = score_opening_auction(AuctionSnapshot(symbol=symbol, observed_at=observed_at, **required))
            print(signal)
        now = datetime.now(TZ)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
