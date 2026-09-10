"""Provider-neutral live data gateway for the Bits100 pipeline.

The gateway wires six real-data domains into one point-in-time envelope:
market/order-book, KAP, news, fund flow, broker consensus and institutional
research. Providers are configured by environment variables; absent credentials
remain BLOCKED rather than becoming synthetic data.
"""
from dataclasses import dataclass
import os
from typing import Any

from .adapters import HttpJsonProvider, ProviderError, parse_json_records


@dataclass(frozen=True, slots=True)
class LiveDataEnvelope:
    symbol: str
    observed_at: str
    market: list[dict[str, Any]]
    kap: list[dict[str, Any]]
    news: list[dict[str, Any]]
    fund_flow: list[dict[str, Any]]
    broker_consensus: list[dict[str, Any]]
    institutional: list[dict[str, Any]]

    @property
    def complete(self) -> bool:
        return bool(self.market) and bool(self.kap or self.news or self.fund_flow or self.broker_consensus or self.institutional)


class LiveDataGateway:
    """Fetch and normalize all configured real-data domains.

    The gateway intentionally does not guess vendor schemas. Every provider
    returns JSON records, which are validated as lists of objects. This makes
    Matriks/Finnet/ForInvest or another licensed vendor pluggable without
    changing the downstream research engine.
    """

    ENV = {
        "market": "BIST_MARKET_DATA_URL",
        "kap": "KAP_API_URL",
        "news": "NEWS_API_URL",
        "fund_flow": "FUND_FLOW_API_URL",
        "broker_consensus": "BROKER_DATA_URL",
        "institutional": "INSTITUTIONAL_DATA_URL",
    }

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    def _provider(self, env_name: str) -> HttpJsonProvider:
        endpoint = os.getenv(env_name, "").strip()
        if not endpoint:
            raise ProviderError(f"{env_name} is not configured")
        token = os.getenv(f"{env_name}_TOKEN", "").strip()
        headers = (("Authorization", f"Bearer {token}"),) if token else ()
        return HttpJsonProvider(endpoint, self.timeout_seconds, headers)

    def fetch_domain(self, domain: str, params: dict[str, str]) -> list[dict[str, Any]]:
        if domain not in self.ENV:
            raise ProviderError(f"unknown live-data domain: {domain}")
        payload = self._provider(self.ENV[domain]).fetch(params)
        return parse_json_records(payload)

    def snapshot(self, symbol: str, start: str, end: str, observed_at: str) -> LiveDataEnvelope:
        params = {"symbol": symbol, "start": start, "end": end}
        domains: dict[str, list[dict[str, Any]]] = {}
        errors: list[str] = []
        for domain in self.ENV:
            try:
                domains[domain] = self.fetch_domain(domain, params)
            except ProviderError as exc:
                errors.append(f"{domain}: {exc}")
                domains[domain] = []
        if not domains["market"]:
            raise ProviderError("LIVE_DATA_BLOCKED: market feed unavailable")
        return LiveDataEnvelope(
            symbol=symbol,
            observed_at=observed_at,
            market=domains["market"],
            kap=domains["kap"],
            news=domains["news"],
            fund_flow=domains["fund_flow"],
            broker_consensus=domains["broker_consensus"],
            institutional=domains["institutional"],
        )
