"""BIST opening-auction (açılış seansı) 09:40 -> 09:45 -> 09:50 -> 09:55 features.

Every field is optional: ``None`` means the provider did not deliver it and it
is *never* replaced by 0, an estimate or synthetic data. Snapshots are
validated against the official schedule (Europe/Istanbul) and fail closed on
future timestamps, duplicate slots, mixed symbols/dates or invalid prices.

The auction component produced here is a research heuristic; its usefulness
must be proven by the historical backtest before it is trusted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import isfinite
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Istanbul")
SLOTS = ("09:40", "09:45", "09:50", "09:55")
SLOT_TOLERANCE = timedelta(seconds=90)


@dataclass(frozen=True, slots=True)
class AuctionObservation:
    """One provider snapshot of the opening auction for one symbol."""

    symbol: str
    observed_at: datetime
    indicative_price: float | None = None
    reference_price: float | None = None
    bid_qty: float | None = None
    ask_qty: float | None = None
    bid_orders: int | None = None
    ask_orders: int | None = None
    new_buy_orders: int | None = None
    new_sell_orders: int | None = None
    cancelled_buy_orders: int | None = None
    cancelled_sell_orders: int | None = None
    matched_volume: float | None = None
    bid_depth_levels: int | None = None
    ask_depth_levels: int | None = None


@dataclass(frozen=True, slots=True)
class SnapshotFeatures:
    slot: str
    observed_at: str
    indicative_price: float | None
    reference_price: float | None
    indicative_change_pct: float | None
    bid_qty: float | None
    ask_qty: float | None
    imbalance: float | None  # (bid-ask)/(bid+ask), -1..1
    bid_ask_ratio: float | None
    order_count: int | None
    new_orders: int | None
    cancelled_orders: int | None
    volume: float | None
    depth: float | None
    depth_levels: int | None


@dataclass(frozen=True, slots=True)
class TransitionFeatures:
    from_slot: str
    to_slot: str
    indicative_change_delta: float | None
    imbalance_delta: float | None
    bid_qty_change_pct: float | None
    ask_qty_change_pct: float | None
    volume_delta: float | None
    order_count_delta: int | None


@dataclass(frozen=True, slots=True)
class AuctionTrajectoryFeatures:
    symbol: str
    session_date: str
    status: str
    snapshots: tuple[SnapshotFeatures, ...] = ()
    transitions: tuple[TransitionFeatures, ...] = ()
    slope: float | None = None  # indicative change pct per 5 minutes (least squares)
    acceleration: float | None = None
    bid_pressure: float | None = None
    ask_pressure: float | None = None
    order_arrival: float | None = None  # new orders per 5 minutes
    cancellation_pressure: float | None = None  # cancelled / new, buy-side minus sell-side
    component_score: float | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _num(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if isfinite(value) else None


def _sum(*values):
    clean = [v for v in values if v is not None]
    return sum(clean) if len(clean) == len(values) else None


def _slot_of(ts: datetime) -> str | None:
    local = ts.astimezone(TZ)
    for slot in SLOTS:
        hour, minute = (int(x) for x in slot.split(":"))
        target = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if abs(local - target) <= SLOT_TOLERANCE:
            return slot
    return None


def snapshot_features(obs: AuctionObservation, slot: str) -> SnapshotFeatures:
    indicative, reference = _num(obs.indicative_price), _num(obs.reference_price)
    bid, ask = _num(obs.bid_qty), _num(obs.ask_qty)
    change = (indicative / reference - 1.0) * 100.0 if indicative and reference else None
    depth = _sum(bid, ask)
    imbalance = (bid - ask) / depth if depth else None
    ratio = bid / ask if bid is not None and ask else None
    return SnapshotFeatures(
        slot=slot,
        observed_at=obs.observed_at.isoformat(),
        indicative_price=indicative,
        reference_price=reference,
        indicative_change_pct=None if change is None else round(change, 4),
        bid_qty=bid,
        ask_qty=ask,
        imbalance=None if imbalance is None else round(imbalance, 6),
        bid_ask_ratio=None if ratio is None else round(ratio, 6),
        order_count=_sum(obs.bid_orders, obs.ask_orders),
        new_orders=_sum(obs.new_buy_orders, obs.new_sell_orders),
        cancelled_orders=_sum(obs.cancelled_buy_orders, obs.cancelled_sell_orders),
        volume=_num(obs.matched_volume),
        depth=depth,
        depth_levels=_sum(obs.bid_depth_levels, obs.ask_depth_levels),
    )


def _delta(a, b):
    return None if a is None or b is None else round(b - a, 6)


def _pct(a, b):
    return None if a is None or b is None or a == 0 else round((b / a - 1.0) * 100.0, 4)


def _transition(a: SnapshotFeatures, b: SnapshotFeatures) -> TransitionFeatures:
    return TransitionFeatures(
        a.slot, b.slot,
        _delta(a.indicative_change_pct, b.indicative_change_pct),
        _delta(a.imbalance, b.imbalance),
        _pct(a.bid_qty, b.bid_qty),
        _pct(a.ask_qty, b.ask_qty),
        _delta(a.volume, b.volume),
        None if a.order_count is None or b.order_count is None else b.order_count - a.order_count,
    )


def _least_squares_slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    return None if denom == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def _blocked(symbol: str, date: str, status: str, reason: str) -> AuctionTrajectoryFeatures:
    return AuctionTrajectoryFeatures(symbol, date, status, reasons=(reason,))


def build_auction_features(
    observations: list[AuctionObservation], *, now: datetime
) -> AuctionTrajectoryFeatures:
    """Validate snapshots and derive per-slot, transition and trajectory features."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not observations:
        return _blocked("", "", "BLOCKED_MISSING_AUCTION_DATA", "no auction snapshots")
    symbols = {o.symbol.upper() for o in observations}
    symbol = sorted(symbols)[0]
    if len(symbols) != 1:
        return _blocked(symbol, "", "BLOCKED_CONTRACT_VIOLATION", "mixed symbols")
    if any(o.observed_at.tzinfo is None for o in observations):
        return _blocked(symbol, "", "BLOCKED_INVALID_TIMESTAMP", "naive timestamp")
    dates = {o.observed_at.astimezone(TZ).date().isoformat() for o in observations}
    date = sorted(dates)[0]
    if len(dates) != 1:
        return _blocked(symbol, date, "BLOCKED_CONTRACT_VIOLATION", "mixed session dates")
    if any(o.observed_at > now for o in observations):
        return _blocked(symbol, date, "BLOCKED_FUTURE_TIMESTAMP", "snapshot after decision time")
    slotted: dict[str, AuctionObservation] = {}
    for obs in observations:
        slot = _slot_of(obs.observed_at)
        if slot is None:
            return _blocked(symbol, date, "BLOCKED_INVALID_TIMESTAMP", f"off-schedule {obs.observed_at}")
        if slot in slotted:
            return _blocked(symbol, date, "BLOCKED_DUPLICATE_EVENT", f"duplicate {slot}")
        for price in (obs.indicative_price, obs.reference_price):
            if price is not None and _num(price) is not None and price <= 0:
                return _blocked(symbol, date, "BLOCKED_INVALID_PRICE", "non-positive price")
        slotted[slot] = obs

    snaps = tuple(snapshot_features(slotted[s], s) for s in SLOTS if s in slotted)
    if all(s.indicative_change_pct is None and s.imbalance is None for s in snaps):
        return _blocked(symbol, date, "BLOCKED_MISSING_AUCTION_DATA", "no price or depth fields")

    # Transitions only between consecutive official slots that are both present.
    transitions = tuple(
        _transition(a, b) for a, b in zip(snaps, snaps[1:]) if SLOTS.index(b.slot) - SLOTS.index(a.slot) == 1
    )
    points = [(SLOTS.index(s.slot), s.indicative_change_pct) for s in snaps if s.indicative_change_pct is not None]
    slope = _least_squares_slope([p[0] for p in points], [p[1] for p in points])
    acceleration = None
    if len(points) >= 3:
        first = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0])
        last = (points[-1][1] - points[-2][1]) / (points[-1][0] - points[-2][0])
        acceleration = round(last - first, 6)

    last = snaps[-1]
    bid_pressure = ask_pressure = None
    if last.bid_qty is not None and last.ask_qty is not None and last.depth:
        bid_pressure = round(last.bid_qty / last.depth, 6)
        ask_pressure = round(last.ask_qty / last.depth, 6)

    arrivals = [s.new_orders for s in snaps if s.new_orders is not None]
    order_arrival = round(sum(arrivals) / len(arrivals), 4) if arrivals else None

    cancellation = None
    buy_new = [slotted[s.slot].new_buy_orders for s in snaps]
    buy_cancel = [slotted[s.slot].cancelled_buy_orders for s in snaps]
    sell_new = [slotted[s.slot].new_sell_orders for s in snaps]
    sell_cancel = [slotted[s.slot].cancelled_sell_orders for s in snaps]
    if all(v is not None for v in buy_new + buy_cancel + sell_new + sell_cancel):
        buy_ratio = sum(buy_cancel) / sum(buy_new) if sum(buy_new) else None
        sell_ratio = sum(sell_cancel) / sum(sell_new) if sum(sell_new) else None
        if buy_ratio is not None and sell_ratio is not None:
            cancellation = round(buy_ratio - sell_ratio, 6)

    score, reasons = _component(last, slope, cancellation)
    status = "COMPLETE" if len(snaps) == len(SLOTS) else "PARTIAL"
    return AuctionTrajectoryFeatures(
        symbol, date, status, snaps, transitions,
        None if slope is None else round(slope, 6), acceleration,
        bid_pressure, ask_pressure, order_arrival, cancellation, score, reasons,
    )


def _bounded(value: float) -> float:
    return max(0.0, min(100.0, value))


def _component(last: SnapshotFeatures, slope, cancellation) -> tuple[float | None, tuple[str, ...]]:
    """Renormalized 0-100 auction component; missing parts are skipped, not zeroed."""
    parts: list[tuple[float, float]] = []
    reasons: list[str] = []
    if last.imbalance is not None:
        parts.append((0.40, _bounded(50.0 + last.imbalance * 50.0)))
        if last.imbalance >= 0.35:
            reasons.append("strong_bid_imbalance")
    if last.indicative_change_pct is not None:
        parts.append((0.25, _bounded(50.0 + last.indicative_change_pct * 5.0)))
        if last.indicative_change_pct >= 5.0:
            reasons.append("strong_indicative_price")
    if slope is not None:
        parts.append((0.25, _bounded(50.0 + slope * 10.0)))
        if slope > 0:
            reasons.append("rising_trajectory")
    if cancellation is not None:
        # Buyers cancelling more than sellers is negative.
        parts.append((0.10, _bounded(50.0 - cancellation * 50.0)))
        if cancellation > 0.2:
            reasons.append("buy_cancel_pressure")
    if not parts:
        return None, ()
    weight = sum(w for w, _ in parts)
    return round(sum(w * v for w, v in parts) / weight, 2), tuple(reasons)


def observation_from_record(record: dict, *, symbol: str) -> AuctionObservation:
    """Map a provider record to an observation without inventing values."""

    def pick(*names):
        for name in names:
            if record.get(name) is not None:
                return record[name]
        return None

    def as_int(value):
        return None if value is None else int(value)

    observed = pick("observed_at", "timestamp")
    if observed is None:
        raise ValueError("record has no timestamp")
    ts = observed if isinstance(observed, datetime) else datetime.fromisoformat(str(observed))
    return AuctionObservation(
        symbol=symbol.upper(),
        observed_at=ts,
        indicative_price=_num(pick("indicative_price", "auction_price")),
        reference_price=_num(pick("reference_price", "previous_close")),
        bid_qty=_num(pick("bid_qty", "total_bid_qty")),
        ask_qty=_num(pick("ask_qty", "total_ask_qty")),
        bid_orders=as_int(pick("bid_orders", "bid_order_count")),
        ask_orders=as_int(pick("ask_orders", "ask_order_count")),
        new_buy_orders=as_int(pick("new_buy_orders")),
        new_sell_orders=as_int(pick("new_sell_orders")),
        cancelled_buy_orders=as_int(pick("cancelled_buy_orders")),
        cancelled_sell_orders=as_int(pick("cancelled_sell_orders")),
        matched_volume=_num(pick("matched_volume", "volume")),
        bid_depth_levels=as_int(pick("bid_depth_levels")),
        ask_depth_levels=as_int(pick("ask_depth_levels")),
    )
