"""Level-2 / Level-2+ order-book features from REAL provider snapshots only.

Input objects come from :mod:`bist_hunter.market_contracts`, so they have
already passed the contract (sorted, uncrossed, timezone-aware). If no
snapshot is available the result is ``BLOCKED_MISSING_ORDER_BOOK`` - a book is
never synthesized. Fields that need data the provider did not send (e.g.
per-level order counts, order events) stay ``None``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median

from .market_contracts import BookLevel, OrderBookSnapshot, OrderEvent

DEPTH_BANDS_BPS = (50, 100, 200)


@dataclass(frozen=True, slots=True)
class OrderBookFeatures:
    symbol: str
    observed_at: str | None
    level: str | None
    status: str
    mid: float | None = None
    spread: float | None = None
    spread_bps: float | None = None
    bid_depth: float | None = None
    ask_depth: float | None = None
    depth_imbalance: float | None = None  # (bid-ask)/(bid+ask), -1..1
    top_level_imbalance: float | None = None
    bid_order_count: int | None = None
    ask_order_count: int | None = None
    depth_concentration: float | None = None  # HHI of level quantities, 0..1
    large_bid_levels: int | None = None
    large_ask_levels: int | None = None
    order_arrival_buy: int | None = None
    order_arrival_sell: int | None = None
    cancel_buy: int | None = None
    cancel_sell: int | None = None
    cancellation_ratio: float | None = None
    large_orders: int | None = None
    # Level-2+ only (deeper book)
    microprice: float | None = None
    band_imbalance: tuple[tuple[int, float | None], ...] = ()
    book_slope_bid: float | None = None
    book_slope_ask: float | None = None
    reasons: tuple[str, ...] = ()


def _qty(levels: tuple[BookLevel, ...]) -> float:
    return sum(level.quantity for level in levels)


def _orders(levels: tuple[BookLevel, ...]) -> int | None:
    if any(level.orders is None for level in levels):
        return None
    return sum(level.orders for level in levels)  # type: ignore[misc]


def _imbalance(bid: float, ask: float) -> float | None:
    total = bid + ask
    return None if total <= 0 else round((bid - ask) / total, 6)


def _hhi(levels: tuple[BookLevel, ...]) -> float | None:
    total = _qty(levels)
    if total <= 0:
        return None
    return round(sum((level.quantity / total) ** 2 for level in levels), 6)


def _slope(levels: tuple[BookLevel, ...], mid: float) -> float | None:
    """Cumulative quantity per bps of distance from mid (steeper = thicker book)."""
    far = abs(levels[-1].price - mid) / mid * 10_000
    return None if far <= 0 else round(_qty(levels) / far, 6)


def _band(levels: tuple[BookLevel, ...], mid: float, bps: int) -> float:
    return sum(lv.quantity for lv in levels if abs(lv.price - mid) / mid * 10_000 <= bps)


def order_book_features(
    snapshot: OrderBookSnapshot | None,
    events: list[OrderEvent] | None = None,
    *,
    now: datetime | None = None,
    event_window: timedelta = timedelta(minutes=5),
    large_level_multiple: float = 3.0,
    large_order_quantity: float | None = None,
    symbol: str = "",
) -> OrderBookFeatures:
    if snapshot is None:
        return OrderBookFeatures(symbol, None, None, "BLOCKED_MISSING_ORDER_BOOK", reasons=("no order book",))
    if now is not None and snapshot.timestamp > now:
        return OrderBookFeatures(snapshot.symbol, snapshot.timestamp.isoformat(), snapshot.level,
                                 "BLOCKED_FUTURE_TIMESTAMP", reasons=("snapshot after decision time",))
    bids, asks = snapshot.bids, snapshot.asks
    best_bid, best_ask = bids[0].price, asks[0].price
    mid = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    bid_depth, ask_depth = _qty(bids), _qty(asks)
    all_levels = bids + asks
    typical = median(level.quantity for level in all_levels)
    reasons: list[str] = []

    imbalance = _imbalance(bid_depth, ask_depth)
    if imbalance is not None and imbalance >= 0.30:
        reasons.append("bid_heavy_book")
    elif imbalance is not None and imbalance <= -0.30:
        reasons.append("ask_heavy_book")

    arrival_buy = arrival_sell = cancel_buy = cancel_sell = large_orders = None
    cancellation_ratio = None
    if events is not None:
        end = now or snapshot.timestamp
        window = [e for e in events if e.symbol == snapshot.symbol and end - event_window <= e.timestamp <= end]
        new = [e for e in window if e.action == "NEW"]
        cancels = [e for e in window if e.action == "CANCEL"]
        arrival_buy = sum(e.side == "BUY" for e in new)
        arrival_sell = sum(e.side == "SELL" for e in new)
        cancel_buy = sum(e.side == "BUY" for e in cancels)
        cancel_sell = sum(e.side == "SELL" for e in cancels)
        cancellation_ratio = round(len(cancels) / len(new), 6) if new else None
        threshold = large_order_quantity if large_order_quantity is not None else typical * large_level_multiple
        large_orders = sum(e.quantity >= threshold for e in new)
        if arrival_buy > 1.5 * max(1, arrival_sell):
            reasons.append("buy_order_arrival")
        if cancellation_ratio is not None and cancellation_ratio > 0.8:
            reasons.append("high_cancellation")

    microprice = band = slope_bid = slope_ask = None
    if snapshot.level == "L2+":
        top_bid, top_ask = bids[0].quantity, asks[0].quantity
        microprice = round((best_bid * top_ask + best_ask * top_bid) / (top_bid + top_ask), 6)
        band = tuple((bps, _imbalance(_band(bids, mid, bps), _band(asks, mid, bps))) for bps in DEPTH_BANDS_BPS)
        slope_bid, slope_ask = _slope(bids, mid), _slope(asks, mid)

    return OrderBookFeatures(
        symbol=snapshot.symbol,
        observed_at=snapshot.timestamp.isoformat(),
        level=snapshot.level,
        status="OK",
        mid=round(mid, 6),
        spread=round(spread, 6),
        spread_bps=round(spread / mid * 10_000, 4),
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        depth_imbalance=imbalance,
        top_level_imbalance=_imbalance(bids[0].quantity, asks[0].quantity),
        bid_order_count=_orders(bids),
        ask_order_count=_orders(asks),
        depth_concentration=_hhi(all_levels),
        large_bid_levels=sum(lv.quantity >= typical * large_level_multiple for lv in bids),
        large_ask_levels=sum(lv.quantity >= typical * large_level_multiple for lv in asks),
        order_arrival_buy=arrival_buy,
        order_arrival_sell=arrival_sell,
        cancel_buy=cancel_buy,
        cancel_sell=cancel_sell,
        cancellation_ratio=cancellation_ratio,
        large_orders=large_orders,
        microprice=microprice,
        band_imbalance=band or (),
        book_slope_bid=slope_bid,
        book_slope_ask=slope_ask,
        reasons=tuple(reasons),
    )


def order_book_score(features: OrderBookFeatures) -> float | None:
    """0-100 research score from available book features (None when blocked)."""
    if features.status != "OK" or features.depth_imbalance is None:
        return None
    parts = [(0.6, 50 + features.depth_imbalance * 50)]
    if features.top_level_imbalance is not None:
        parts.append((0.2, 50 + features.top_level_imbalance * 50))
    if features.order_arrival_buy is not None and features.order_arrival_sell is not None:
        total = features.order_arrival_buy + features.order_arrival_sell
        if total:
            parts.append((0.2, 100 * features.order_arrival_buy / total))
    weight = sum(w for w, _ in parts)
    return round(max(0.0, min(100.0, sum(w * v for w, v in parts) / weight)), 2)
