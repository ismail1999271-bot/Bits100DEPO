"""Concrete internet-backed adapters with fail-closed source contracts."""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime, timedelta
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .adapters import ProviderError, parse_json_records


def _get(url: str, headers: dict[str, str] | None = None, timeout: float = 20.0) -> bytes:
    request = Request(url, headers=headers or {"User-Agent": "Bits100DEPO/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except Exception as exc:
        raise ProviderError(f"HTTP request failed: {exc}") from exc


def yahoo_chart(symbol: str, start: date, end: date, interval: str = "1d") -> list[dict[str, object]]:
    ticker = symbol if symbol.endswith(".IS") else f"{symbol}.IS"
    params = urlencode({
        "period1": int(datetime.combine(start, datetime.min.time(), UTC).timestamp()),
        "period2": int(datetime.combine(end + timedelta(days=1), datetime.min.time(), UTC).timestamp()),
        "interval": interval, "events": "history", "includeAdjustedClose": "true",
    })
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


def historical_bars(symbol: str, start: date, end: date) -> list[dict[str, object]]:
    """Use the configured licensed BIST history provider, otherwise Yahoo as a public fallback."""
    endpoint = os.getenv("BIST_HISTORICAL_API_URL", "")
    if endpoint:
        rows = _records_from_endpoint(endpoint, "BIST_DATA_API_KEY", {
            "symbol": symbol, "start": start.isoformat(), "end": end.isoformat(), "interval": "1d",
        })
        normalized = []
        for row in rows:
            try:
                normalized.append({
                    "symbol": symbol,
                    "timestamp": row.get("timestamp", row.get("date")),
                    "open": float(row["open"]), "high": float(row["high"]),
                    "low": float(row["low"]), "close": float(row["close"]),
                    "volume": float(row.get("volume", 0)),
                })
            except (KeyError, TypeError, ValueError):
                continue
        if not normalized:
            raise ProviderError(f"historical provider returned no valid rows for {symbol}")
        return normalized
    return yahoo_chart(symbol, start, end)


def intraday_bars(symbol: str, start: datetime, end: datetime, interval: str = "5m") -> list[dict[str, object]]:
    endpoint = os.getenv("BIST_INTRADAY_API_URL", "")
    if endpoint:
        rows = _records_from_endpoint(endpoint, "BIST_DATA_API_KEY", {
            "symbol": symbol, "start": start.isoformat(), "end": end.isoformat(), "interval": interval,
        })
        return [{
            "symbol": symbol, "timestamp": row.get("timestamp", row.get("date")),
            "open": float(row["open"]), "high": float(row["high"]), "low": float(row["low"]),
            "close": float(row["close"]), "volume": float(row.get("volume", 0)),
        } for row in rows]
    return yahoo_chart(symbol, start.date(), end.date(), interval=interval)


def _records_from_endpoint(endpoint: str, token_env: str | None = None, params: dict[str, str] | None = None) -> list[dict[str, object]]:
    headers = {"User-Agent": "Bits100DEPO/1.0", "Accept": "application/json"}
    if token_env:
        token = os.getenv(token_env, "")
        if not token:
            raise ProviderError(f"{token_env} is not configured")
        headers["Authorization"] = f"Bearer {token}"
        headers["X-API-KEY"] = token
    url = endpoint
    if params:
        url += ("&" if "?" in url else "?") + urlencode(params)
    try:
        return parse_json_records(json.loads(_get(url, headers).decode("utf-8")))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProviderError(f"endpoint did not return valid JSON: {exc}") from exc


def load_symbol_universe() -> list[str]:
    endpoint = os.getenv("BIST_UNIVERSE_URL", "")
    if endpoint:
        rows = _records_from_endpoint(endpoint, "BIST_DATA_API_KEY")
        symbols = [str(row.get("symbol", row.get("code", row.get("ticker", "")))).upper().strip().removesuffix(".IS") for row in rows]
        symbols = list(dict.fromkeys(symbol for symbol in symbols if symbol))
        if not symbols:
            raise ProviderError("BIST_UNIVERSE_URL returned an empty universe")
        return symbols
    raw = os.getenv("BIST_SYMBOLS", "")
    symbols = list(dict.fromkeys(item.strip().upper().removesuffix(".IS") for item in raw.split(",") if item.strip()))
    if not symbols:
        raise ProviderError("BIST_UNIVERSE_URL/BIST_SYMBOLS is not configured; refusing to scan a partial universe")
    return symbols


def fetch_json_endpoint(endpoint: str, token_env: str | None = None) -> list[dict[str, object]]:
    return _records_from_endpoint(endpoint, token_env)


def fetch_kap_disclosures(start: date, end: date) -> list[dict[str, object]]:
    endpoint = os.getenv("KAP_API_URL", "")
    if not endpoint:
        raise ProviderError("KAP_API_URL is not configured")
    rows = _records_from_endpoint(endpoint, "KAP_API_KEY", {"start": start.isoformat(), "end": end.isoformat()})
    return [{"symbol": str(row.get("symbol", row.get("stockCode", row.get("memberCode", "")))).upper(), "title": str(row.get("title", row.get("subject", ""))), "text": str(row.get("text", row.get("summary", ""))), "published_at": row.get("published_at", row.get("publishDate")), "source_id": row.get("id", row.get("disclosureId"))} for row in rows]


def fetch_news_rss(url: str) -> list[dict[str, object]]:
    root = ET.fromstring(_get(url))
    rows = []
    for item in root.findall(".//item"):
        categories = [((node.text or "").strip().upper()) for node in item.findall("category")]
        rows.append({"source": "rss", "symbol": next((x for x in categories if 2 < len(x) <= 6 and x.isascii() and x.isalnum()), ""), "title": (item.findtext("title") or "").strip(), "description": (item.findtext("description") or "").strip(), "url": (item.findtext("link") or "").strip(), "published_at": (item.findtext("pubDate") or "").strip()})
    return rows


def fetch_fund_flow(start: date, end: date) -> list[dict[str, object]]:
    endpoint = os.getenv("FUND_FLOW_API_URL", "")
    if not endpoint:
        raise ProviderError("FUND_FLOW_API_URL is not configured")
    rows = _records_from_endpoint(endpoint, "FUND_FLOW_API_KEY", {"start": start.isoformat(), "end": end.isoformat()})
    return [{"symbol": str(row.get("symbol", row.get("stockCode", row.get("ticker", "")))).upper(), "score": row.get("score"), "net_flow": row.get("net_flow", row.get("netFlow", 0)), "source_id": row.get("id", row.get("fundId"))} for row in rows]


def source_healthcheck() -> dict[str, bool]:
    return {"market": bool(os.getenv("BIST_UNIVERSE_URL") or os.getenv("BIST_SYMBOLS")), "historical": bool(os.getenv("BIST_HISTORICAL_API_URL") or os.getenv("BIST_SYMBOLS")), "intraday": bool(os.getenv("BIST_INTRADAY_API_URL") or os.getenv("BIST_SYMBOLS")), "kap": bool(os.getenv("KAP_API_URL") and os.getenv("KAP_API_KEY")), "fund_flow": bool(os.getenv("FUND_FLOW_API_URL") and os.getenv("FUND_FLOW_API_KEY")), "news_rss": bool(os.getenv("NEWS_RSS_URL")), "telegram": bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"))}
