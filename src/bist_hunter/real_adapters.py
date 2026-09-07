"""Concrete internet-backed adapters.

Market data uses Yahoo Finance chart responses as a no-key fallback. KAP and
institutional/fund feeds remain credential/contract driven because licensed
feeds must not be faked or scraped as if they were official APIs.
"""
from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, UTC
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .adapters import ProviderError, parse_json_records


def _get(url: str, headers: dict[str, str] | None = None, timeout: float = 15.0) -> bytes:
    request = Request(url, headers=headers or {"User-Agent": "Bits100DEPO/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except Exception as exc:
        raise ProviderError(f"HTTP request failed: {exc}") from exc


def yahoo_chart(symbol: str, start: date, end: date) -> list[dict[str, object]]:
    """Return daily OHLCV for a BIST ticker using Yahoo's public chart API."""
    ticker = symbol if symbol.endswith(".IS") else f"{symbol}.IS"
    params = urlencode({"period1": int(datetime.combine(start, datetime.min.time(), UTC).timestamp()),
                        "period2": int(datetime.combine(end + timedelta(days=1), datetime.min.time(), UTC).timestamp()),
                        "interval": "1d", "events": "history", "includeAdjustedClose": "true"})
    payload = json.loads(_get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?{params}").decode())
    result = payload.get("chart", {}).get("result")
    if not result:
        raise ProviderError(f"Yahoo returned no chart data for {symbol}")
    chart = result[0]
    quote = chart.get("indicators", {}).get("quote", [{}])[0]
    timestamps = chart.get("timestamp", [])
    rows = []
    for i, stamp in enumerate(timestamps):
        values = {key: quote.get(key, [None] * len(timestamps))[i] for key in ("open", "high", "low", "close", "volume")}
        if all(value is not None for value in values.values()):
            rows.append({"symbol": symbol, "timestamp": datetime.fromtimestamp(stamp, UTC).isoformat(), **values})
    return rows


def load_symbol_universe() -> list[str]:
    """Load the complete eligible universe from the configured provider."""
    raw = os.getenv("BIST_SYMBOLS", "")
    symbols = [item.strip().upper() for item in raw.split(",") if item.strip()]
    if not symbols:
        raise ProviderError("BIST_SYMBOLS is not configured; refusing to scan a partial universe")
    return list(dict.fromkeys(symbols))


def fetch_json_endpoint(endpoint: str, token_env: str | None = None) -> list[dict[str, object]]:
    headers = {"User-Agent": "Bits100DEPO/1.0"}
    if token_env:
        token = os.getenv(token_env, "")
        if not token:
            raise ProviderError(f"{token_env} is not configured")
        headers["Authorization"] = f"Bearer {token}"
    return parse_json_records(json.loads(_get(endpoint, headers).decode("utf-8")))


def fetch_kap_disclosures(start: date, end: date) -> list[dict[str, object]]:
    endpoint = os.getenv("KAP_API_URL", "")
    if not endpoint:
        raise ProviderError("KAP_API_URL is not configured")
    query = urlencode({"start": start.isoformat(), "end": end.isoformat()})
    return fetch_json_endpoint(f"{endpoint}?{query}", "KAP_API_KEY")


def fetch_news_rss(url: str) -> list[dict[str, object]]:
    root = ET.fromstring(_get(url))
    rows = []
    for item in root.findall(".//item"):
        rows.append({
            "source": "rss",
            "title": (item.findtext("title") or "").strip(),
            "url": (item.findtext("link") or "").strip(),
            "published_at": (item.findtext("pubDate") or "").strip(),
        })
    return rows


def fetch_fund_flow(start: date, end: date) -> list[dict[str, object]]:
    endpoint = os.getenv("FUND_FLOW_API_URL", "")
    if not endpoint:
        raise ProviderError("FUND_FLOW_API_URL is not configured")
    query = urlencode({"start": start.isoformat(), "end": end.isoformat()})
    return fetch_json_endpoint(f"{endpoint}?{query}", "FUND_FLOW_API_KEY")


def source_healthcheck() -> dict[str, bool]:
    """Expose configuration health without making a network call."""
    return {
        "market": bool(os.getenv("BIST_SYMBOLS")),
        "kap": bool(os.getenv("KAP_API_URL") and os.getenv("KAP_API_KEY")),
        "fund_flow": bool(os.getenv("FUND_FLOW_API_URL") and os.getenv("FUND_FLOW_API_KEY")),
        "news_rss": bool(os.getenv("NEWS_RSS_URL")),
        "telegram": bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID")),
    }
