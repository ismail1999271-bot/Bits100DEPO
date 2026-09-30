"""KAP, news, fund-flow, broker (AKD) and institutional features - point in time.

* Every event is normalized to ``symbol, event_id, published_at, source,
  event_type, headline``.
* Only events with ``published_at <= as_of`` are ever used (no future leakage).
* Duplicates are removed by (source, event_id) and by the same normalized
  headline for the same symbol inside a short window (cross-source copies).
* Absent data returns status ``MISSING_DATA`` and a ``None`` score - it is never
  turned into 0 or a neutral placeholder.
* Keyword sentiment is a transparent baseline; its value must be proven by the
  backtest.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping

from .market_contracts import BrokerFlowRow, InstitutionalFlowRow
from .provider_contracts import ContractError, parse_timestamp

MISSING_DATA = "MISSING_DATA"
OK = "OK"

BULLISH = ("yatırım", "sözleşme", "sipariş", "kâr", "kar artışı", "temettü", "geri alım", "ihale",
           "bedelsiz", "iş ilişkisi", "anlaşma", "ihracat", "kapasite artışı")
BEARISH = ("zarar", "iflas", "dava", "soruşturma", "bedelli sermaye artırımı", "ceza", "haciz",
           "konkordato", "işlem yasağı", "tedbir")


@dataclass(frozen=True, slots=True)
class NormalizedEvent:
    symbol: str
    event_id: str
    published_at: datetime
    source: str
    event_type: str
    headline: str


def _headline(row: Mapping[str, Any]) -> str:
    for key in ("headline", "title", "subject", "summary", "text"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ContractError("missing required field: headline")


def normalize_event(row: Mapping[str, Any], *, default_source: str, default_type: str) -> NormalizedEvent:
    symbol = str(row.get("symbol", "")).strip().upper().removesuffix(".IS")
    event_id = str(row.get("event_id", row.get("id", ""))).strip()
    if not symbol or not event_id:
        raise ContractError("event requires symbol and event_id")
    published = parse_timestamp(row.get("published_at"))
    event_type = str(row.get("event_type", row.get("disclosure_type", row.get("type", default_type)))).strip()
    source = str(row.get("source", default_source)).strip() or default_source
    return NormalizedEvent(symbol, event_id, published, source, event_type.upper() or default_type,
                           _headline(row))


def normalize_kap(rows: Iterable[Mapping[str, Any]]) -> list[NormalizedEvent]:
    return [normalize_event(r, default_source="KAP", default_type="KAP_DISCLOSURE") for r in rows]


def normalize_news(rows: Iterable[Mapping[str, Any]]) -> list[NormalizedEvent]:
    return [normalize_event(r, default_source="NEWS", default_type="NEWS") for r in rows]


def _norm_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def deduplicate(events: Iterable[NormalizedEvent], *, window: timedelta = timedelta(hours=24)) -> list[NormalizedEvent]:
    """Keep the earliest copy of each event (by id and by headline within ``window``)."""
    seen_ids: set[tuple[str, str]] = set()
    last_by_text: dict[tuple[str, str], datetime] = {}
    result: list[NormalizedEvent] = []
    for event in sorted(events, key=lambda e: (e.published_at, e.source, e.event_id)):
        id_key = (event.source, event.event_id)
        text_key = (event.symbol, _norm_text(event.headline))
        previous = last_by_text.get(text_key)
        if id_key in seen_ids or (previous is not None and event.published_at - previous <= window):
            continue
        seen_ids.add(id_key)
        last_by_text[text_key] = event.published_at
        result.append(event)
    return result


def point_in_time(events: Iterable[NormalizedEvent], as_of: datetime) -> list[NormalizedEvent]:
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    return [e for e in events if e.published_at <= as_of]


def headline_polarity(headline: str) -> int:
    text = headline.casefold()
    return sum(term in text for term in BULLISH) - sum(term in text for term in BEARISH)


@dataclass(frozen=True, slots=True)
class EventScore:
    symbol: str
    status: str
    score: float | None
    events: int
    latest: str | None
    reasons: tuple[str, ...] = ()


def event_score(
    events: Iterable[NormalizedEvent], symbol: str, as_of: datetime, *,
    lookback: timedelta = timedelta(days=3), half_life_hours: float = 24.0,
) -> EventScore:
    symbol = symbol.upper()
    window = [e for e in point_in_time(events, as_of)
              if e.symbol == symbol and as_of - e.published_at <= lookback]
    if not window:
        return EventScore(symbol, MISSING_DATA, None, 0, None)
    weighted = total = 0.0
    for event in window:
        age_h = (as_of - event.published_at).total_seconds() / 3600
        weight = 0.5 ** (age_h / half_life_hours)
        weighted += weight * headline_polarity(event.headline)
        total += weight
    mean = weighted / total
    score = round(max(0.0, min(100.0, 50.0 + 25.0 * mean)), 2)
    reasons = ("positive_events",) if mean > 0 else ("negative_events",) if mean < 0 else ()
    return EventScore(symbol, OK, score, len(window), max(e.published_at for e in window).isoformat(), reasons)


# ---------------------------------------------------------------------------
# Fund flow
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FundFlowFeatures:
    symbol: str
    status: str
    inflow: float | None = None
    outflow: float | None = None
    net_flow: float | None = None
    concentration: float | None = None  # HHI of |flow| across funds
    funds: int = 0
    score: float | None = None


def fund_flow_features(rows: Iterable[Mapping[str, Any]], symbol: str, as_of: datetime,
                       *, scale_try: float = 50_000_000.0) -> FundFlowFeatures:
    """Rows: symbol, fund_code, net_flow (TRY), observed_at|published_at."""
    symbol = symbol.upper()
    by_fund: dict[str, float] = {}
    for row in rows:
        if str(row.get("symbol", "")).upper().removesuffix(".IS") != symbol:
            continue
        ts = parse_timestamp(row.get("observed_at", row.get("published_at")))
        if ts > as_of:
            continue
        value = float(row["net_flow"])
        if not math.isfinite(value):
            raise ContractError("non-finite net_flow")
        fund = str(row.get("fund_code", row.get("fund", "UNKNOWN")))
        by_fund[fund] = by_fund.get(fund, 0.0) + value
    if not by_fund:
        return FundFlowFeatures(symbol, MISSING_DATA)
    inflow = sum(v for v in by_fund.values() if v > 0)
    outflow = -sum(v for v in by_fund.values() if v < 0)
    gross = inflow + outflow
    hhi = sum((abs(v) / gross) ** 2 for v in by_fund.values()) if gross else None
    net = inflow - outflow
    score = round(50.0 + 50.0 * math.tanh(net / scale_try), 2)
    return FundFlowFeatures(symbol, OK, inflow, outflow, net, None if hhi is None else round(hhi, 6),
                            len(by_fund), score)


# ---------------------------------------------------------------------------
# Broker distribution (AKD)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BrokerFeatures:
    symbol: str
    status: str
    broker_buying: float | None = None
    broker_selling: float | None = None
    net_broker_flow: float | None = None
    top5_buy_share: float | None = None
    top5_sell_share: float | None = None
    concentration: float | None = None  # HHI of net positions
    brokers: int = 0
    score: float | None = None


def broker_features(rows: Iterable[BrokerFlowRow], symbol: str, as_of: datetime) -> BrokerFeatures:
    symbol = symbol.upper()
    data = [r for r in rows if r.symbol == symbol and r.timestamp <= as_of]
    if not data:
        return BrokerFeatures(symbol, MISSING_DATA)
    buys: dict[str, float] = {}
    sells: dict[str, float] = {}
    for row in data:
        buys[row.broker] = buys.get(row.broker, 0.0) + row.buy_quantity
        sells[row.broker] = sells.get(row.broker, 0.0) + row.sell_quantity
    total_buy, total_sell = sum(buys.values()), sum(sells.values())
    net = total_buy - total_sell
    top_buy = sum(sorted(buys.values(), reverse=True)[:5]) / total_buy if total_buy else None
    top_sell = sum(sorted(sells.values(), reverse=True)[:5]) / total_sell if total_sell else None
    nets = {b: buys.get(b, 0.0) - sells.get(b, 0.0) for b in set(buys) | set(sells)}
    gross = sum(abs(v) for v in nets.values())
    hhi = sum((abs(v) / gross) ** 2 for v in nets.values()) if gross else None
    # Concentrated buying (few brokers accumulating) scores higher than dispersed buying.
    base = 50.0 + 50.0 * (net / (total_buy + total_sell) if total_buy + total_sell else 0.0)
    tilt = ((top_buy or 0) - (top_sell or 0)) * 20.0
    score = round(max(0.0, min(100.0, base + tilt)), 2)
    return BrokerFeatures(symbol, OK, total_buy, total_sell, net,
                          None if top_buy is None else round(top_buy, 6),
                          None if top_sell is None else round(top_sell, 6),
                          None if hhi is None else round(hhi, 6), len(nets), score)


# ---------------------------------------------------------------------------
# Institutional flow
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InstitutionalFeatures:
    symbol: str
    status: str
    institutional_flow: float | None = None
    accumulation_days: int | None = None
    observation_days: int = 0
    accumulation_ratio: float | None = None
    by_investor_type: tuple[tuple[str, float], ...] = ()
    score: float | None = None


def institutional_features(rows: Iterable[InstitutionalFlowRow], symbol: str, as_of: datetime,
                           *, lookback: timedelta = timedelta(days=20),
                           scale_try: float = 100_000_000.0) -> InstitutionalFeatures:
    symbol = symbol.upper()
    data = [r for r in rows if r.symbol == symbol and as_of - lookback <= r.timestamp <= as_of]
    if not data:
        return InstitutionalFeatures(symbol, MISSING_DATA)
    by_day: dict[str, float] = {}
    by_type: dict[str, float] = {}
    for row in data:
        day = row.timestamp.date().isoformat()
        by_day[day] = by_day.get(day, 0.0) + row.net_value
        by_type[row.investor_type] = by_type.get(row.investor_type, 0.0) + row.net_value
    total = sum(by_day.values())
    positive_days = sum(v > 0 for v in by_day.values())
    ratio = positive_days / len(by_day)
    score = round(max(0.0, min(100.0, 0.6 * (50 + 50 * math.tanh(total / scale_try)) + 0.4 * 100 * ratio)), 2)
    return InstitutionalFeatures(symbol, OK, total, positive_days, len(by_day), round(ratio, 6),
                                 tuple(sorted(by_type.items())), score)


def quant_components(
    *,
    kap: EventScore | None = None,
    news: EventScore | None = None,
    fund: FundFlowFeatures | None = None,
    broker: BrokerFeatures | None = None,
    institutional: InstitutionalFeatures | None = None,
) -> dict[str, float]:
    """Map available feature scores to Quant Score component names (missing stay absent)."""
    out: dict[str, float] = {}
    for name, item in (("kap", kap), ("news", news), ("fund", fund), ("broker", broker),
                       ("institutional", institutional)):
        if item is not None and item.status == OK and item.score is not None:
            out[name] = float(item.score)
    return out
