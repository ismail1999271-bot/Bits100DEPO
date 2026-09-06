"""Provider adapters for real market, KAP, news and social feeds.

The adapters intentionally require an explicit endpoint. No fake live data is
created when credentials or a licensed provider are absent.
"""
from dataclasses import dataclass
import json
from typing import Any
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class HttpJsonProvider:
    endpoint: str
    timeout_seconds: float = 10.0
    headers: tuple[tuple[str, str], ...] = ()

    def fetch(self, params: dict[str, str] | None = None) -> Any:
        if not self.endpoint:
            raise ProviderError("provider endpoint is not configured")
        query = ""
        if params:
            from urllib.parse import urlencode
            query = "?" + urlencode(params)
        request = Request(self.endpoint + query, headers=dict(self.headers))
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise ProviderError(f"provider request failed: {exc}") from exc


@dataclass(frozen=True, slots=True)
class BistMarketDataAdapter:
    provider: HttpJsonProvider

    def bars(self, symbol: str, start: str, end: str) -> Any:
        return self.provider.fetch({"symbol": symbol, "start": start, "end": end})


@dataclass(frozen=True, slots=True)
class KapDisclosureAdapter:
    provider: HttpJsonProvider

    def disclosures(self, start: str, end: str) -> Any:
        return self.provider.fetch({"start": start, "end": end})


@dataclass(frozen=True, slots=True)
class NewsAdapter:
    provider: HttpJsonProvider

    def headlines(self, start: str, end: str) -> Any:
        return self.provider.fetch({"start": start, "end": end})


@dataclass(frozen=True, slots=True)
class SocialAdapter:
    provider: HttpJsonProvider

    def posts(self, start: str, end: str) -> Any:
        return self.provider.fetch({"start": start, "end": end})


def parse_json_records(payload: Any) -> list[dict[str, Any]]:
    """Normalize common API payload envelopes without assuming a vendor schema."""
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        records = payload["data"]
    elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
        records = payload["items"]
    else:
        raise ProviderError("payload does not contain a record list")
    if not all(isinstance(item, dict) for item in records):
        raise ProviderError("provider records must be objects")
    return records
