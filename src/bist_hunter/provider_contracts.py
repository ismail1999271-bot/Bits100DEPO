"""Strict contracts for real provider observations.

The contracts are deliberately vendor-neutral: adapters translate a licensed
provider schema into these records before data enters the research pipeline.
Missing required fields, stale timestamps, invalid quantities and duplicate
observations are rejected rather than silently downgraded to synthetic data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


class ContractError(ValueError):
    """Provider payload violates a required normalized contract."""


DOMAINS = ("market", "kap", "news", "fund_flow", "broker_consensus", "institutional")


@dataclass(frozen=True, slots=True)
class MarketObservation:
    symbol: str
    observed_at: str
    indicative_price: float
    reference_price: float
    bid_qty: float
    ask_qty: float
    order_arrival_buy: float | None = None
    order_arrival_sell: float | None = None
    order_cancel_buy: float | None = None
    order_cancel_sell: float | None = None
    volume: float | None = None
    baseline_volume: float | None = None


@dataclass(frozen=True, slots=True)
class EventObservation:
    symbol: str
    published_at: str
    source: str
    event_id: str
    payload: Mapping[str, Any]


def _text(row: Mapping[str, Any], name: str) -> str:
    value = row.get(name)
    if value is None or not str(value).strip():
        raise ContractError(f"missing required field: {name}")
    return str(value).strip()


def _positive(row: Mapping[str, Any], name: str, *, allow_zero: bool = False) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"invalid numeric field: {name}") from exc
    if value < 0 or (not allow_zero and value <= 0):
        raise ContractError(f"invalid non-positive field: {name}")
    return value


def _optional_nonnegative(row: Mapping[str, Any], name: str) -> float | None:
    value = row.get(name)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"invalid numeric field: {name}") from exc
    if value < 0:
        raise ContractError(f"negative field: {name}")
    return value


def parse_timestamp(value: Any) -> datetime:
    if value is None:
        raise ContractError("missing required field: observed_at")
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ContractError("timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def parse_market_observation(row: Mapping[str, Any], *, max_age_seconds: float | None = None, now: datetime | None = None) -> MarketObservation:
    observed = parse_timestamp(row.get("observed_at"))
    if max_age_seconds is not None:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        age = (current - observed).total_seconds()
        if age > max_age_seconds:
            raise ContractError("stale market observation")
        if age < -5:
            raise ContractError("future market observation")
    symbol = _text(row, "symbol").upper()
    return MarketObservation(
        symbol=symbol,
        observed_at=observed.isoformat(),
        indicative_price=_positive(row, "indicative_price"),
        reference_price=_positive(row, "reference_price"),
        bid_qty=_positive(row, "bid_qty", allow_zero=True),
        ask_qty=_positive(row, "ask_qty", allow_zero=True),
        order_arrival_buy=_optional_nonnegative(row, "order_arrival_buy"),
        order_arrival_sell=_optional_nonnegative(row, "order_arrival_sell"),
        order_cancel_buy=_optional_nonnegative(row, "order_cancel_buy"),
        order_cancel_sell=_optional_nonnegative(row, "order_cancel_sell"),
        volume=_optional_nonnegative(row, "volume"),
        baseline_volume=_optional_nonnegative(row, "baseline_volume"),
    )


def parse_event_observation(row: Mapping[str, Any], *, timestamp_field: str = "published_at") -> EventObservation:
    published = parse_timestamp(row.get(timestamp_field))
    return EventObservation(
        symbol=_text(row, "symbol").upper(),
        published_at=published.isoformat(),
        source=_text(row, "source"),
        event_id=_text(row, "event_id"),
        payload=dict(row),
    )


def validate_unique_keys(rows: list[Mapping[str, Any]], keys: tuple[str, ...]) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = tuple(row.get(item) for item in keys)
        if any(value is None for value in key):
            raise ContractError(f"missing uniqueness key: {keys}")
        if key in seen:
            raise ContractError(f"duplicate provider observation: {key}")
        seen.add(key)
