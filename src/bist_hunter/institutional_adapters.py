"""Vendor-neutral adapters for institutional datasets.

These adapters define the normalized contract that Bloomberg/LSEG/FactSet/Capital IQ/
AlphaSense/PitchBook/Preqin-like feeds can implement. They do not bypass licensing or
pretend that a premium dataset is available without a configured provider.
"""
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from .adapters import HttpJsonProvider, ProviderError, parse_json_records


class RecordFetcher(Protocol):
    def fetch(self, params: dict[str, str] | None = None) -> Any: ...


@dataclass(frozen=True, slots=True)
class InstitutionalFeedAdapter:
    provider: RecordFetcher
    dataset: str

    def fetch_records(self, start: str, end: str) -> list[dict[str, Any]]:
        payload = self.provider.fetch({"dataset": self.dataset, "start": start, "end": end})
        return parse_json_records(payload)

    def normalize(self, start: str, end: str) -> pd.DataFrame:
        records = self.fetch_records(start, end)
        if not records:
            return pd.DataFrame()
        frame = pd.DataFrame(records)
        required = {"symbol", "observed_at"}
        missing = required.difference(frame.columns)
        if missing:
            raise ProviderError(f"normalized institutional records missing: {sorted(missing)}")
        frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
        if "as_of" in frame:
            frame["as_of"] = pd.to_datetime(frame["as_of"], utc=True)
            if bool((frame["as_of"] > frame["observed_at"]).any()):
                raise ProviderError("institutional feed contains future as_of information")
        frame["source_dataset"] = self.dataset
        return frame


@dataclass(frozen=True, slots=True)
class InstitutionalDataFabric:
    """Named feed families used by the research orchestrator."""
    market: InstitutionalFeedAdapter
    fundamentals: InstitutionalFeedAdapter
    estimates: InstitutionalFeedAdapter
    holdings: InstitutionalFeedAdapter
    flows: InstitutionalFeedAdapter
    research: InstitutionalFeedAdapter
    transactions: InstitutionalFeedAdapter
    alternative: InstitutionalFeedAdapter

    def snapshot(self, start: str, end: str) -> dict[str, pd.DataFrame]:
        return {
            "market": self.market.normalize(start, end),
            "fundamentals": self.fundamentals.normalize(start, end),
            "estimates": self.estimates.normalize(start, end),
            "holdings": self.holdings.normalize(start, end),
            "flows": self.flows.normalize(start, end),
            "research": self.research.normalize(start, end),
            "transactions": self.transactions.normalize(start, end),
            "alternative": self.alternative.normalize(start, end),
        }


def adapter_from_http(endpoint: str, dataset: str, *, token: str | None = None) -> InstitutionalFeedAdapter:
    """Create a feed adapter from an explicit HTTP endpoint.

    Authentication is optional because some data providers use other mechanisms. Secrets are
    passed at runtime and never embedded in repository code.
    """
    headers = (("Authorization", f"Bearer {token}"),) if token else ()
    return InstitutionalFeedAdapter(HttpJsonProvider(endpoint=endpoint, headers=headers), dataset)
