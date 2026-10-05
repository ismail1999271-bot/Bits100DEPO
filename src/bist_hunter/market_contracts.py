"""Normalized, fail-closed contracts for real BIST market-data payloads.

Covers OHLCV, tick/trade, Level-1 quotes, Level-2 / Level-2+ order-book
snapshots, market-by-order events (arrival / cancel / modify / trade), broker
distribution (AKD) and institutional flow rows.

Every parser raises :class:`ContractError` on a violation. Callers convert that
into an ``INVALID`` / ``BLOCKED`` state; nothing here repairs, estimates or
synthesizes values. Timestamps must be timezone-aware ISO-8601 and are checked
against the decision time (no future data) and, optionally, a maximum age.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Iterable, Mapping

from .provider_contracts import ContractError, parse_timestamp

FUTURE_TOLERANCE_SECONDS = 5.0
BOOK_LEVELS = {"L2", "L2+"}
ORDER_ACTIONS = {"NEW", "CANCEL", "MODIFY", "TRADE"}
SIDES = {"BUY", "SELL"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _symbol(row: Mapping[str, Any]) -> str:
    value = row.get("symbol")
    if value is None or not str(value).strip():
        raise ContractError("missing required field: symbol")
    return str(value).strip().upper().removesuffix(".IS")


def _number(row: Mapping[str, Any], name: str, *, positive: bool = False, optional: bool = False):
    value = row.get(name)
    if value is None:
        if optional:
            return None
        raise ContractError(f"missing required field: {name}")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"invalid numeric field: {name}") from exc
    if not isfinite(value):
        raise ContractError(f"non-finite field: {name}")
    if value < 0 or (positive and value == 0):
        raise ContractError(f"invalid {'non-positive' if positive else 'negative'} field: {name}")
    return value


def _text(row: Mapping[str, Any], name: str, *, optional: bool = False) -> str | None:
    value = row.get(name)
    if value is None or not str(value).strip():
        if optional:
            return None
        raise ContractError(f"missing required field: {name}")
    return str(value).strip()


def check_time(
    ts: datetime, *, now: datetime | None, max_age_seconds: float | None
) -> None:
    """Reject future observations and (optionally) stale ones."""
    if now is None:
        return
    current = now.astimezone(timezone.utc)
    age = (current - ts).total_seconds()
    if age < -FUTURE_TOLERANCE_SECONDS:
        raise ContractError("future timestamp")
    if max_age_seconds is not None and age > max_age_seconds:
        raise ContractError("stale observation")


def _ts(row: Mapping[str, Any], *names: str, now, max_age) -> datetime:
    for name in names:
        if row.get(name) is not None:
            ts = parse_timestamp(row[name])
            check_time(ts, now=now, max_age_seconds=max_age)
            return ts
    raise ContractError(f"missing timestamp ({'/'.join(names)})")


# ---------------------------------------------------------------------------
# contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OhlcvBar:
    symbol: str
    timestamp: datetime
    interval: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class Tick:
    symbol: str
    timestamp: datetime
    price: float
    quantity: float
    trade_id: str
    aggressor: str | None = None


@dataclass(frozen=True, slots=True)
class Level1Quote:
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last: float | None = None


@dataclass(frozen=True, slots=True)
class BookLevel:
    price: float
    quantity: float
    orders: int | None = None


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    symbol: str
    timestamp: datetime
    level: str  # "L2" or "L2+"
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    sequence: int | None = None


@dataclass(frozen=True, slots=True)
class OrderEvent:
    symbol: str
    timestamp: datetime
    order_id: str
    side: str
    action: str
    price: float | None
    quantity: float


@dataclass(frozen=True, slots=True)
class BrokerFlowRow:
    symbol: str
    timestamp: datetime
    broker: str
    buy_quantity: float
    sell_quantity: float
    buy_value: float | None
    sell_value: float | None
    source: str
    event_id: str


@dataclass(frozen=True, slots=True)
class InstitutionalFlowRow:
    symbol: str
    timestamp: datetime
    investor_type: str
    net_value: float
    source: str
    event_id: str
    buy_value: float | None = None
    sell_value: float | None = None


# ---------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------


def parse_ohlcv(row: Mapping[str, Any], *, now: datetime | None = None, max_age_seconds: float | None = None,
                interval: str | None = None) -> OhlcvBar:
    bar = OhlcvBar(
        symbol=_symbol(row),
        timestamp=_ts(row, "timestamp", "observed_at", now=now, max_age=max_age_seconds),
        interval=interval or _text(row, "interval", optional=True) or "1D",
        open=_number(row, "open", positive=True),
        high=_number(row, "high", positive=True),
        low=_number(row, "low", positive=True),
        close=_number(row, "close", positive=True),
        volume=_number(row, "volume"),
    )
    if bar.high < max(bar.open, bar.close, bar.low) or bar.low > min(bar.open, bar.close):
        raise ContractError("inconsistent OHLC range")
    return bar


def parse_tick(row: Mapping[str, Any], *, now: datetime | None = None,
               max_age_seconds: float | None = None) -> Tick:
    aggressor = _text(row, "aggressor", optional=True)
    if aggressor is not None and aggressor.upper() not in SIDES:
        raise ContractError("aggressor must be BUY or SELL")
    return Tick(
        symbol=_symbol(row),
        timestamp=_ts(row, "timestamp", "observed_at", now=now, max_age=max_age_seconds),
        price=_number(row, "price", positive=True),
        quantity=_number(row, "quantity", positive=True),
        trade_id=_text(row, "trade_id"),
        aggressor=None if aggressor is None else aggressor.upper(),
    )


def parse_level1(row: Mapping[str, Any], *, now: datetime | None = None,
                 max_age_seconds: float | None = None) -> Level1Quote:
    quote = Level1Quote(
        symbol=_symbol(row),
        timestamp=_ts(row, "timestamp", "observed_at", now=now, max_age=max_age_seconds),
        bid=_number(row, "bid", positive=True),
        ask=_number(row, "ask", positive=True),
        bid_size=_number(row, "bid_size"),
        ask_size=_number(row, "ask_size"),
        last=_number(row, "last", positive=True, optional=True),
    )
    if quote.bid >= quote.ask:
        raise ContractError("crossed or locked Level-1 quote")
    return quote


def _levels(raw: Any, side: str) -> tuple[BookLevel, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ContractError(f"{side} side must be a non-empty list")
    levels: list[BookLevel] = []
    for item in raw:
        if isinstance(item, Mapping):
            price, qty, orders = item.get("price"), item.get("quantity", item.get("qty")), item.get("orders")
        elif isinstance(item, (list, tuple)) and len(item) in (2, 3):
            price, qty, orders = item[0], item[1], item[2] if len(item) == 3 else None
        else:
            raise ContractError(f"invalid {side} level: {item!r}")
        level = BookLevel(
            _number({"price": price}, "price", positive=True),
            _number({"quantity": qty}, "quantity", positive=True),
            None if orders is None else int(_number({"orders": orders}, "orders", positive=True)),
        )
        levels.append(level)
    prices = [lv.price for lv in levels]
    ordered = sorted(prices, reverse=(side == "bids"))
    if prices != ordered or len(set(prices)) != len(prices):
        raise ContractError(f"{side} levels must be strictly sorted by price")
    return tuple(levels)


def parse_order_book(row: Mapping[str, Any], *, now: datetime | None = None,
                     max_age_seconds: float | None = None) -> OrderBookSnapshot:
    level = (_text(row, "level", optional=True) or "L2").upper()
    if level not in BOOK_LEVELS:
        raise ContractError("level must be L2 or L2+")
    bids = _levels(row.get("bids"), "bids")
    asks = _levels(row.get("asks"), "asks")
    if bids[0].price >= asks[0].price:
        raise ContractError("crossed or locked order book")
    sequence = row.get("sequence")
    return OrderBookSnapshot(
        symbol=_symbol(row),
        timestamp=_ts(row, "timestamp", "observed_at", now=now, max_age=max_age_seconds),
        level=level,
        bids=bids,
        asks=asks,
        sequence=None if sequence is None else int(sequence),
    )


def parse_order_event(row: Mapping[str, Any], *, now: datetime | None = None,
                      max_age_seconds: float | None = None) -> OrderEvent:
    side = (_text(row, "side") or "").upper()
    action = (_text(row, "action") or "").upper()
    if side not in SIDES:
        raise ContractError("side must be BUY or SELL")
    if action not in ORDER_ACTIONS:
        raise ContractError(f"action must be one of {sorted(ORDER_ACTIONS)}")
    return OrderEvent(
        symbol=_symbol(row),
        timestamp=_ts(row, "timestamp", "observed_at", now=now, max_age=max_age_seconds),
        order_id=_text(row, "order_id"),
        side=side,
        action=action,
        price=_number(row, "price", positive=True, optional=action == "CANCEL"),
        quantity=_number(row, "quantity", positive=action != "CANCEL"),
    )


def parse_broker_flow(row: Mapping[str, Any], *, now: datetime | None = None) -> BrokerFlowRow:
    return BrokerFlowRow(
        symbol=_symbol(row),
        timestamp=_ts(row, "published_at", "observed_at", now=now, max_age=None),
        broker=_text(row, "broker").upper(),
        buy_quantity=_number(row, "buy_quantity"),
        sell_quantity=_number(row, "sell_quantity"),
        buy_value=_number(row, "buy_value", optional=True),
        sell_value=_number(row, "sell_value", optional=True),
        source=_text(row, "source"),
        event_id=_text(row, "event_id"),
    )


def parse_institutional_flow(row: Mapping[str, Any], *, now: datetime | None = None) -> InstitutionalFlowRow:
    net = row.get("net_value")
    try:
        net_value = float(net)
    except (TypeError, ValueError) as exc:
        raise ContractError("invalid numeric field: net_value") from exc
    if not isfinite(net_value):
        raise ContractError("non-finite field: net_value")
    return InstitutionalFlowRow(
        symbol=_symbol(row),
        timestamp=_ts(row, "published_at", "observed_at", now=now, max_age=None),
        investor_type=_text(row, "investor_type").upper(),
        net_value=net_value,
        source=_text(row, "source"),
        event_id=_text(row, "event_id"),
        buy_value=_number(row, "buy_value", optional=True),
        sell_value=_number(row, "sell_value", optional=True),
    )


def parse_many(parser, rows: Iterable[Mapping[str, Any]], *, key, **kwargs) -> list:
    """Parse rows and reject duplicates by ``key(parsed)``; fail closed on the first error."""
    parsed = []
    seen = set()
    for row in rows:
        item = parser(row, **kwargs)
        k = key(item)
        if k in seen:
            raise ContractError(f"duplicate observation: {k}")
        seen.add(k)
        parsed.append(item)
    return parsed
