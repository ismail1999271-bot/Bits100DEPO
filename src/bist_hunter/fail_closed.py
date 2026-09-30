"""Central fail-closed gate: no research signal unless the data is trustworthy.

BLOCKED reasons (any one blocks the symbol):
missing market data, stale data, invalid timestamp, future timestamp,
duplicate event, provider contract violation, missing credentials,
insufficient coverage, bad data quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

BLOCKED = "BLOCKED"
PASS = "PASS"

REASONS = (
    "MISSING_MARKET_DATA", "STALE_DATA", "INVALID_TIMESTAMP", "FUTURE_TIMESTAMP", "DUPLICATE_EVENT",
    "PROVIDER_CONTRACT_VIOLATION", "MISSING_CREDENTIALS", "INSUFFICIENT_COVERAGE", "BAD_DATA_QUALITY",
)


@dataclass(frozen=True, slots=True)
class GateInput:
    symbol: str
    now: datetime
    market_timestamp: datetime | str | None
    max_market_age_seconds: float
    coverage: float | None
    min_coverage: float = 0.40
    provider_statuses: dict[str, str] = field(default_factory=dict)  # domain -> CONNECTED/STALE/...
    required_domains: tuple[str, ...] = ("ohlcv",)
    duplicate_events: int = 0
    contract_violations: int = 0
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GateResult:
    symbol: str
    status: str
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == PASS


def _parse(ts) -> datetime | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def evaluate_gate(item: GateInput) -> GateResult:
    reasons: list[str] = []
    if item.now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if item.market_timestamp is None:
        reasons.append("MISSING_MARKET_DATA")
    else:
        ts = _parse(item.market_timestamp)
        if ts is None or ts.tzinfo is None:
            reasons.append("INVALID_TIMESTAMP")
        else:
            age = (item.now.astimezone(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds()
            if age < -5:
                reasons.append("FUTURE_TIMESTAMP")
            elif age > item.max_market_age_seconds:
                reasons.append("STALE_DATA")
    for domain in item.required_domains:
        status = item.provider_statuses.get(domain, "MISSING")
        if status == "BLOCKED":
            reasons.append("MISSING_CREDENTIALS")
        elif status == "INVALID":
            reasons.append("PROVIDER_CONTRACT_VIOLATION")
        elif status == "STALE":
            reasons.append("STALE_DATA")
        elif status != "CONNECTED":
            reasons.append("MISSING_MARKET_DATA")
    if item.duplicate_events:
        reasons.append("DUPLICATE_EVENT")
    if item.contract_violations:
        reasons.append("PROVIDER_CONTRACT_VIOLATION")
    if item.coverage is None or item.coverage < item.min_coverage:
        reasons.append("INSUFFICIENT_COVERAGE")
    if item.quality_flags:
        reasons.append("BAD_DATA_QUALITY")
    unique = tuple(dict.fromkeys(reasons))
    return GateResult(item.symbol.upper(), BLOCKED if unique else PASS, unique)
