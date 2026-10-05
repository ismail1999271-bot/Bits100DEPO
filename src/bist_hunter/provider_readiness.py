"""Deterministic readiness checks for the external market-data domains."""
from __future__ import annotations

import os
from dataclasses import dataclass


DOMAINS = ("market", "kap", "news", "fund_flow", "broker", "institutional")
ENV_KEYS = {
    "market": "BIST_MARKET_DATA_URL",
    "kap": "KAP_API_URL",
    "news": "NEWS_API_URL",
    "fund_flow": "FUND_FLOW_API_URL",
    "broker": "BROKER_DATA_URL",
    "institutional": "INSTITUTIONAL_DATA_URL",
}
TOKEN_KEYS = {
    domain: key.replace("_URL", "_TOKEN") for domain, key in ENV_KEYS.items()
}


@dataclass(frozen=True, slots=True)
class ProviderReadiness:
    domain: str
    configured: bool
    credentialed: bool
    status: str
    reason: str


def evaluate_provider_readiness(env: dict[str, str] | None = None) -> tuple[ProviderReadiness, ...]:
    values = os.environ if env is None else env
    result = []
    for domain in DOMAINS:
        endpoint = bool(values.get(ENV_KEYS[domain], "").strip())
        token = bool(values.get(TOKEN_KEYS[domain], "").strip())
        configured = endpoint
        credentialed = endpoint and token
        if credentialed:
            status, reason = "CONFIGURED", "endpoint and credential configured"
        elif configured:
            status, reason = "BLOCKED_MISSING_CREDENTIAL", "endpoint configured but credential is missing"
        else:
            status, reason = "BLOCKED_MISSING_ENDPOINT", "licensed provider endpoint is not configured"
        result.append(ProviderReadiness(domain, configured, credentialed, status, reason))
    return tuple(result)


def production_data_ready(env: dict[str, str] | None = None) -> bool:
    """Return true only when every required external data domain is credentialed."""
    return all(item.credentialed for item in evaluate_provider_readiness(env))
