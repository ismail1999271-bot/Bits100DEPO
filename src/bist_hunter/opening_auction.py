"""Point-in-time opening-auction pressure engine.

This module only consumes provider-supplied observations. It never invents
order-book values. Missing live fields produce an explicit blocked state.
"""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class AuctionSnapshot:
    symbol: str
    observed_at: str
    indicative_price: float
    reference_price: float
    bid_qty: float
    ask_qty: float
    order_arrival_buy: float = 0.0
    order_arrival_sell: float = 0.0
    order_cancel_buy: float = 0.0
    order_cancel_sell: float = 0.0
    volume: float = 0.0
    baseline_volume: float = 0.0
    tavan_dna: float | None = None
    kap_score: float | None = None
    news_score: float | None = None
    fund_score: float | None = None
    institutional_score: float | None = None
    broker_score: float | None = None


@dataclass(frozen=True, slots=True)
class AuctionSignal:
    symbol: str
    observed_at: str
    status: str
    score: float | None
    indicative_change_pct: float | None
    imbalance_pct: float | None
    queue_strength: float | None
    order_flow_score: float | None
    volume_anomaly: float | None
    reasons: tuple[str, ...]


def _bounded(value: float) -> float:
    return max(0.0, min(100.0, value))


def _optional_mean(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and isfinite(float(v))]
    return sum(clean) / len(clean) if clean else None


def score_opening_auction(snapshot: AuctionSnapshot) -> AuctionSignal:
    required = (
        snapshot.indicative_price,
        snapshot.reference_price,
        snapshot.bid_qty,
        snapshot.ask_qty,
    )
    if any(not isfinite(float(v)) for v in required):
        return AuctionSignal(
            snapshot.symbol, snapshot.observed_at, "BLOCKED_MISSING_MARKET_DATA",
            None, None, None, None, None, None, ("missing required auction fields",)
        )
    if snapshot.indicative_price <= 0 or snapshot.reference_price <= 0:
        return AuctionSignal(
            snapshot.symbol, snapshot.observed_at, "BLOCKED_INVALID_PRICE",
            None, None, None, None, None, None, ("non-positive price",)
        )

    total_depth = snapshot.bid_qty + snapshot.ask_qty
    imbalance = ((snapshot.bid_qty - snapshot.ask_qty) / total_depth * 100.0) if total_depth else 0.0
    queue = _bounded(50.0 + imbalance / 2.0)

    arrivals = snapshot.order_arrival_buy + snapshot.order_arrival_sell
    cancels = snapshot.order_cancel_buy + snapshot.order_cancel_sell
    net = (snapshot.order_arrival_buy - snapshot.order_arrival_sell) - (
        snapshot.order_cancel_buy - snapshot.order_cancel_sell
    )
    flow_score = _bounded(50.0 + (net / arrivals * 50.0 if arrivals else 0.0))

    volume_anomaly = None
    if snapshot.baseline_volume > 0:
        volume_anomaly = snapshot.volume / snapshot.baseline_volume
    volume_score = _bounded((volume_anomaly or 1.0) * 50.0)

    indicative_change = (snapshot.indicative_price / snapshot.reference_price - 1.0) * 100.0
    microstructure = _bounded(
        0.40 * queue + 0.35 * flow_score + 0.15 * volume_score + 0.10 * _bounded(50 + indicative_change * 5)
    )

    external = _optional_mean([
        snapshot.tavan_dna,
        snapshot.kap_score,
        snapshot.news_score,
        snapshot.fund_score,
        snapshot.institutional_score,
        snapshot.broker_score,
    ])
    score = microstructure if external is None else 0.80 * microstructure + 0.20 * external

    reasons: list[str] = []
    if imbalance >= 35:
        reasons.append("strong_buy_imbalance")
    elif imbalance <= -35:
        reasons.append("strong_sell_imbalance")
    if flow_score >= 65:
        reasons.append("positive_order_flow")
    if volume_anomaly is not None and volume_anomaly >= 2:
        reasons.append("volume_anomaly")
    if indicative_change >= 5:
        reasons.append("strong_indicative_price")

    return AuctionSignal(
        snapshot.symbol,
        snapshot.observed_at,
        "SIGNAL",
        round(_bounded(score), 2),
        round(indicative_change, 4),
        round(imbalance, 2),
        round(queue, 2),
        round(flow_score, 2),
        None if volume_anomaly is None else round(volume_anomaly, 4),
        tuple(reasons),
    )
