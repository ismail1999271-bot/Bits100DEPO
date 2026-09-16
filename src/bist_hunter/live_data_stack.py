"""Provider-neutral live data gateway for the Bits100 pipeline.

Provider adapters translate licensed vendor payloads into strict normalized
records. Missing credentials/data remain BLOCKED; no synthetic observations are
created.
"""
from dataclasses import dataclass
import os
from typing import Any

from .adapters import HttpJsonProvider, ProviderError, parse_json_records
from .provider_contracts import ContractError, parse_market_observation, validate_unique_keys


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
    """Fetch six real-data domains and enforce point-in-time market integrity."""

    ENV = {
        "market": "BIST_MARKET_DATA_URL",
        "kap": "KAP_API_URL",
        "news": "NEWS_API_URL",
        "fund_flow": "FUND_FLOW_API_URL",
        "broker_consensus": "BROKER_DATA_URL",
        "institutional": "INSTITUTIONAL_DATA_URL",
    }

    def __init__(self, timeout_seconds: float = 10.0, market_max_age_seconds: float | None = 30.0) -> None:
        self.timeout_seconds = timeout_seconds
        self.market_max_age_seconds = market_max_age_seconds

    def _provider(self, env_name: str) -> HttpJsonProvider:
        endpoint = os.getenv(env_name, "").strip()
        if not endpoint:
            raise ProviderError(f"{env_name} is not configured")
        token = os.getenv(f"{env_name}_TOKEN", "").strip()
        if not token:
            raise ProviderError(f"{env_name}_TOKEN is not configured")
        headers = (("Authorization", f"Bearer {token}"),)
        return HttpJsonProvider(endpoint, self.timeout_seconds, headers)

    def fetch_domain(self, domain: str, params: dict[str, str]) -> list[dict[str, Any]]:
        if domain not in self.ENV:
            raise ProviderError(f"unknown live-data domain: {domain}")
        payload = self._provider(self.ENV[domain]).fetch(params)
        return parse_json_records(payload)

    def _validate_market(self, rows: list[dict[str, Any]], symbol: str) -> list[dict[str, Any]]:
        if not rows:
            return []
        normalized = []
        for row in rows:
            try:
                observation = parse_market_observation(row, max_age_seconds=self.market_max_age_seconds)
            except ContractError as exc:
                raise ProviderError(f"MARKET_CONTRACT_BLOCKED: {exc}") from exc
            if observation.symbol != symbol.upper():
                raise ProviderError("MARKET_CONTRACT_BLOCKED: symbol mismatch")
            normalized.append(row)
        try:
            validate_unique_keys(normalized, ("symbol", "observed_at"))
        except ContractError as exc:
            raise ProviderError(f"MARKET_CONTRACT_BLOCKED: {exc}") from exc
        return normalized

    def snapshot(self, symbol: str, start: str, end: str, observed_at: str) -> LiveDataEnvelope:
        params = {"symbol": symbol, "start": start, "end": end}
        domains: dict[str, list[dict[str, Any]]] = {}
        for domain in self.ENV:
            try:
                domains[domain] = self.fetch_domain(domain, params)
            except ProviderError:
                if domain == "market":
                    raise
                domains[domain] = []
        domains["market"] = self._validate_market(domains["market"], symbol)
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
