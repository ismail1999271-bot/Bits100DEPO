"""Provider registry, fail-closed client and CONNECTED/STALE/INVALID/MISSING/BLOCKED status.

* No endpoint for a *required* domain -> BLOCKED; for an optional domain -> MISSING.
* Endpoint without credential -> BLOCKED.
* A fetch whose payload violates the contract -> INVALID (whole batch rejected).
* Newest observation older than the domain's freshness budget -> STALE.
* Otherwise -> CONNECTED.

Only CONNECTED data may feed signals. The client never falls back to
synthetic, cached-as-live or zero-filled data.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from . import market_contracts as mc
from .adapters import HttpJsonProvider, ProviderError, parse_json_records
from .provider_contracts import ContractError, parse_event_observation

CONNECTED, STALE, INVALID, MISSING, BLOCKED = "CONNECTED", "STALE", "INVALID", "MISSING", "BLOCKED"
STATUSES = (CONNECTED, STALE, INVALID, MISSING, BLOCKED)


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    domain: str
    label: str
    endpoint_env: str
    token_env: str
    required: bool
    max_age_seconds: float | None  # None: event feeds, freshness not enforced per row
    parser: Callable[..., Any]
    timestamp_attr: str = "timestamp"


def _event(field: str):
    def parse(row, *, now=None, max_age_seconds=None):
        event = parse_event_observation(row, timestamp_field=field)
        ts = datetime.fromisoformat(event.published_at)
        mc.check_time(ts, now=now, max_age_seconds=max_age_seconds)
        return event
    return parse


REGISTRY: tuple[ProviderSpec, ...] = (
    ProviderSpec("ohlcv", "BIST Market (OHLCV)", "BIST_MARKET_DATA_URL", "BIST_MARKET_DATA_URL_TOKEN",
                 True, 26 * 3600, mc.parse_ohlcv),
    ProviderSpec("tick", "Tick / Trades", "BIST_TICK_DATA_URL", "BIST_TICK_DATA_URL_TOKEN",
                 False, 60, mc.parse_tick),
    ProviderSpec("level1", "Level 1", "BIST_LEVEL1_URL", "BIST_LEVEL1_URL_TOKEN",
                 False, 30, mc.parse_level1),
    ProviderSpec("level2", "Level 2", "BIST_LEVEL2_URL", "BIST_LEVEL2_URL_TOKEN",
                 False, 30, mc.parse_order_book),
    ProviderSpec("level2plus", "Level 2+", "BIST_LEVEL2PLUS_URL", "BIST_LEVEL2PLUS_URL_TOKEN",
                 False, 30, mc.parse_order_book),
    ProviderSpec("order_events", "Order arrival / cancel", "BIST_ORDER_EVENTS_URL",
                 "BIST_ORDER_EVENTS_URL_TOKEN", False, 60, mc.parse_order_event),
    ProviderSpec("kap", "KAP", "KAP_API_URL", "KAP_API_URL_TOKEN", False, None, _event("published_at"),
                 "published_at"),
    ProviderSpec("news", "News", "NEWS_API_URL", "NEWS_API_URL_TOKEN", False, None, _event("published_at"),
                 "published_at"),
    ProviderSpec("fund_flow", "Fund Flow", "FUND_FLOW_API_URL", "FUND_FLOW_API_URL_TOKEN", False, None,
                 _event("observed_at"), "published_at"),
    ProviderSpec("broker", "Broker (AKD)", "BROKER_DATA_URL", "BROKER_DATA_URL_TOKEN", False, None,
                 mc.parse_broker_flow),
    ProviderSpec("institutional", "Institutional", "INSTITUTIONAL_DATA_URL", "INSTITUTIONAL_DATA_URL_TOKEN",
                 False, None, mc.parse_institutional_flow),
)
SPECS = {spec.domain: spec for spec in REGISTRY}


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    domain: str
    label: str
    status: str
    reason: str
    rows: int = 0
    newest: str | None = None


def configuration_status(spec: ProviderSpec, env: Mapping[str, str] | None = None) -> ProviderStatus:
    values = os.environ if env is None else env
    endpoint = values.get(spec.endpoint_env, "").strip()
    token = values.get(spec.token_env, "").strip()
    if not endpoint:
        status = BLOCKED if spec.required else MISSING
        return ProviderStatus(spec.domain, spec.label, status, f"{spec.endpoint_env} not configured")
    if not token:
        return ProviderStatus(spec.domain, spec.label, BLOCKED, f"{spec.token_env} not configured")
    return ProviderStatus(spec.domain, spec.label, CONNECTED, "configured (not yet probed)")


def evaluate_batch(
    spec: ProviderSpec, rows: list[Mapping[str, Any]], *, now: datetime
) -> tuple[ProviderStatus, list[Any]]:
    """Validate a fetched batch; return status and parsed rows (empty unless CONNECTED)."""
    if not rows:
        return ProviderStatus(spec.domain, spec.label, MISSING, "provider returned no rows"), []
    try:
        parsed = [spec.parser(row, now=now) for row in rows]
    except ContractError as exc:
        return ProviderStatus(spec.domain, spec.label, INVALID, f"contract violation: {exc}", len(rows)), []
    keys = [(getattr(p, "symbol", None), str(getattr(p, spec.timestamp_attr, None)),
             getattr(p, "event_id", None) or getattr(p, "trade_id", None) or getattr(p, "order_id", None)
             or getattr(p, "broker", None) or getattr(p, "investor_type", None) or getattr(p, "interval", None))
            for p in parsed]
    if len(set(keys)) != len(keys):
        return ProviderStatus(spec.domain, spec.label, INVALID, "duplicate observations", len(rows)), []
    stamps = [getattr(p, spec.timestamp_attr) for p in parsed]
    newest = max(datetime.fromisoformat(s) if isinstance(s, str) else s for s in stamps)
    if spec.max_age_seconds is not None:
        age = (now.astimezone(timezone.utc) - newest).total_seconds()
        if age > spec.max_age_seconds:
            return ProviderStatus(spec.domain, spec.label, STALE, f"newest row {int(age)}s old",
                                  len(rows), newest.isoformat()), []
    return ProviderStatus(spec.domain, spec.label, CONNECTED, "ok", len(rows), newest.isoformat()), parsed


class ProviderClient:
    """Fetch + validate one domain. Any failure is reported, never papered over."""

    def __init__(self, env: Mapping[str, str] | None = None, timeout_seconds: float = 10.0,
                 transport: Callable[[ProviderSpec, dict[str, str]], Any] | None = None) -> None:
        self.env = os.environ if env is None else env
        self.timeout_seconds = timeout_seconds
        self._transport = transport or self._http

    def _http(self, spec: ProviderSpec, params: dict[str, str]) -> Any:
        provider = HttpJsonProvider(
            self.env[spec.endpoint_env].strip(), self.timeout_seconds,
            (("Authorization", f"Bearer {self.env[spec.token_env].strip()}"),),
        )
        return provider.fetch(params)

    def fetch(self, domain: str, params: dict[str, str], *, now: datetime) -> tuple[ProviderStatus, list[Any]]:
        spec = SPECS[domain]
        configured = configuration_status(spec, self.env)
        if configured.status != CONNECTED:
            return configured, []
        try:
            rows = parse_json_records(self._transport(spec, params))
        except ProviderError as exc:
            return ProviderStatus(domain, spec.label, BLOCKED, f"fetch failed: {exc}"), []
        return evaluate_batch(spec, rows, now=now)


def status_board(env: Mapping[str, str] | None = None) -> tuple[ProviderStatus, ...]:
    """Configuration-level status of every domain (no network calls)."""
    return tuple(configuration_status(spec, env) for spec in REGISTRY)


def signals_allowed(statuses: tuple[ProviderStatus, ...]) -> bool:
    """Signals require every *required* domain to be CONNECTED."""
    by_domain = {s.domain: s for s in statuses}
    return all(by_domain.get(spec.domain) is not None and by_domain[spec.domain].status == CONNECTED
               for spec in REGISTRY if spec.required)


def format_status_board(statuses: tuple[ProviderStatus, ...]) -> str:
    width = max(len(s.label) for s in statuses)
    return "\n".join(f"{s.label.ljust(width)}  {s.status:<9}  {s.reason}" for s in statuses)
