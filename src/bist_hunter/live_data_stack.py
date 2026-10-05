"""Provider-neutral live data gateway for the Bits100 pipeline."""
from dataclasses import dataclass
import os
from typing import Any

from .adapters import HttpJsonProvider, ProviderError, parse_json_records
from .provider_contracts import ContractError, parse_market_observation, validate_domain_rows, validate_unique_keys


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
    """Fetch six real-data domains and fail closed on contract violations."""

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
        return HttpJsonProvider(endpoint, self.timeout_seconds, (("Authorization", f"Bearer {token}"),))

    def fetch_domain(self, domain: str, params: dict[str, str]) -> list[dict[str, Any]]:
        if domain not in self.ENV:
            raise ProviderError(f"unknown live-data domain: {domain}")
        return parse_json_records(self._provider(self.ENV[domain]).fetch(params))

    def _validate_market(self, rows: list[dict[str, Any]], symbol: str) -> list[dict[str, Any]]:
        normalized = []
        for row in rows:
            try:
                observation = parse_market_observation(row, max_age_seconds=self.market_max_age_seconds)
                if observation.symbol != symbol.upper():
                    raise ContractError("symbol mismatch")
            except ContractError as exc:
                raise ProviderError(f"MARKET_CONTRACT_BLOCKED: {exc}") from exc
            normalized.append(row)
        try:
            validate_unique_keys(normalized, ("symbol", "observed_at"))
        except ContractError as exc:
            raise ProviderError(f"MARKET_CONTRACT_BLOCKED: {exc}") from exc
        return normalized

    def _validate_domains(self, domains: dict[str, list[dict[str, Any]]]) -> None:
        for domain, rows in domains.items():
            if domain == "market" or not rows:
                continue
            try:
                validate_domain_rows(domain, rows)
            except ContractError as exc:
                raise ProviderError(f"{domain.upper()}_CONTRACT_BLOCKED: {exc}") from exc

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
        self._validate_domains(domains)
        if not domains["market"]:
            raise ProviderError("LIVE_DATA_BLOCKED: market feed unavailable")
        return LiveDataEnvelope(symbol, observed_at, domains["market"], domains["kap"], domains["news"], domains["fund_flow"], domains["broker_consensus"], domains["institutional"])
