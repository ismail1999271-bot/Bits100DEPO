from datetime import datetime, timedelta, timezone

import pytest

from bist_hunter import market_contracts as mc
from bist_hunter.level2_features import order_book_features, order_book_score

NOW = datetime(2026, 9, 11, 7, 30, tzinfo=timezone.utc)


def book(level="L2", orders=True):
    def lv(p, q, o):
        return [p, q, o] if orders else [p, q]
    return mc.parse_order_book({
        "symbol": "THYAO", "timestamp": (NOW - timedelta(seconds=2)).isoformat(), "level": level,
        "bids": [lv(100.0, 3000, 10), lv(99.9, 1000, 5), lv(99.0, 500, 2)],
        "asks": [lv(100.1, 1000, 4), lv(100.2, 500, 3), lv(101.5, 500, 2)],
    })


def event(i, side, action, qty=100, seconds=30):
    return mc.parse_order_event({"symbol": "THYAO", "timestamp": (NOW - timedelta(seconds=seconds)).isoformat(),
                                 "order_id": str(i), "side": side, "action": action, "price": 100.0,
                                 "quantity": qty})


def test_level2_core_features():
    f = order_book_features(book(), now=NOW)
    assert f.status == "OK"
    assert f.spread == pytest.approx(0.1)
    assert f.spread_bps == pytest.approx(0.1 / 100.05 * 10_000, rel=1e-3)
    assert f.bid_depth == 4500 and f.ask_depth == 2000
    assert f.depth_imbalance == pytest.approx(2500 / 6500, rel=1e-5)
    assert f.bid_order_count == 17 and f.ask_order_count == 9
    assert 0 < f.depth_concentration < 1
    assert f.large_bid_levels == 1
    assert f.microprice is None and f.band_imbalance == ()  # L2+ only
    assert f.order_arrival_buy is None  # no events supplied -> not zero
    assert "bid_heavy_book" in f.reasons


def test_level2_plus_deeper_features():
    f = order_book_features(book("L2+"), now=NOW)
    assert f.microprice == pytest.approx((100.0 * 1000 + 100.1 * 3000) / 4000)
    assert [b for b, _ in f.band_imbalance] == [50, 100, 200]
    assert f.book_slope_bid > 0 and f.book_slope_ask > 0


def test_order_events_arrival_cancel_large():
    events = [event(1, "BUY", "NEW"), event(2, "BUY", "NEW", qty=5000), event(3, "SELL", "NEW"),
              event(4, "BUY", "CANCEL"), event(5, "SELL", "NEW", seconds=900)]
    f = order_book_features(book(), events, now=NOW)
    assert (f.order_arrival_buy, f.order_arrival_sell) == (2, 1)
    assert (f.cancel_buy, f.cancel_sell) == (1, 0)
    assert f.cancellation_ratio == pytest.approx(1 / 3, rel=1e-5)
    assert f.large_orders == 1
    assert 50 < order_book_score(f) <= 100


def test_missing_counts_stay_none_and_missing_book_blocks():
    assert order_book_features(book(orders=False), now=NOW).bid_order_count is None
    blocked = order_book_features(None, symbol="THYAO")
    assert blocked.status == "BLOCKED_MISSING_ORDER_BOOK" and order_book_score(blocked) is None
    future = order_book_features(book(), now=NOW - timedelta(minutes=1))
    assert future.status == "BLOCKED_FUTURE_TIMESTAMP"
