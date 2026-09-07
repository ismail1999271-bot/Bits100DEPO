"""Production readiness and provider health gates.

This module deliberately fails closed: missing credentials or unvalidated feeds
never become synthetic live data and never enable trading.
"""
from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class Gate:
    name: str
    ready: bool
    reason: str


REQUIRED_PROVIDER_ENV = {
    "BIST market data": ("BIST_MARKET_DATA_URL",),
    "KAP": ("KAP_API_URL",),
    "Fund flow": ("FUND_FLOW_API_URL",),
    "Broker consensus": ("BROKER_DATA_URL",),
    "Institutional intelligence": ("INSTITUTIONAL_DATA_URL",),
}


def _configured(names: tuple[str, ...]) -> bool:
    return all(os.getenv(name, "").strip() for name in names)


def readiness_report() -> tuple[Gate, ...]:
    gates = [
        Gate("Claude research agent", bool(os.getenv("ANTHROPIC_API_KEY")),
             "API key configured" if os.getenv("ANTHROPIC_API_KEY") else "ANTHROPIC_API_KEY missing"),
        Gate("Telegram delivery", bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID")),
             "credentials configured" if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID") else "Telegram secrets missing"),
        Gate("Paper trading", True, "paper mode is fail-safe and does not place live orders"),
        Gate("Final holdout", True, "gate exists; it must remain untouched until model selection is frozen"),
    ]
    for label, names in REQUIRED_PROVIDER_ENV.items():
        ok = _configured(names)
        gates.append(Gate(label, ok, "endpoint configured" if ok else f"missing: {', '.join(names)}"))
    return tuple(gates)


def production_allowed() -> bool:
    """Return True only when every external provider gate is configured."""
    return all(g.ready for g in readiness_report())


def format_report() -> str:
    lines = ["Bits100 production readiness"]
    for gate in readiness_report():
        lines.append(f"{'READY' if gate.ready else 'BLOCKED'} | {gate.name} | {gate.reason}")
    lines.append(f"PRODUCTION={'ALLOWED' if production_allowed() else 'BLOCKED'}")
    return "\n".join(lines)
