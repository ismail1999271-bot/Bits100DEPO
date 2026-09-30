"""Collect the 09:40 / 09:45 / 09:50 / 09:55 opening-auction snapshots (fail-closed).

The licensed real-time provider must supply the observations. Missing
credentials/endpoints produce BLOCKED; no value is ever estimated or faked.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from time import sleep

from bist_hunter.adapters import ProviderError
from bist_hunter.auction_features import SLOTS, TZ, build_auction_features, observation_from_record
from bist_hunter.live_data_stack import LiveDataGateway


def main() -> int:
    symbols = [s.strip().upper() for s in os.getenv("BIST_SYMBOLS", "").split(",") if s.strip()]
    if not symbols:
        print("BLOCKED: BIST_SYMBOLS is not configured")
        return 2
    gateway = LiveDataGateway()
    session = datetime.now(TZ).date()
    collected: dict[str, list] = {symbol: [] for symbol in symbols}
    errors: dict[str, str] = {}
    for slot in SLOTS:
        hour, minute = (int(x) for x in slot.split(":"))
        target = datetime(session.year, session.month, session.day, hour, minute, tzinfo=TZ)
        wait = (target - datetime.now(TZ)).total_seconds()
        if wait > 0:
            sleep(wait)
        observed_at = datetime.now(TZ).isoformat()
        for symbol in symbols:
            try:
                envelope = gateway.snapshot(symbol, observed_at, observed_at, observed_at)
                collected[symbol].append(observation_from_record(envelope.market[-1], symbol=symbol))
            except (ProviderError, ValueError) as exc:
                errors[f"{symbol}@{slot}"] = f"BLOCKED: {exc}"
    now = datetime.now(TZ)
    report = {symbol: asdict(build_auction_features(obs, now=now)) for symbol, obs in collected.items()}
    print(json.dumps({"session": session.isoformat(), "features": report, "errors": errors},
                     ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
