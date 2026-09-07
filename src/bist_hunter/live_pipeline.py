"""Single-universe provider-backed research pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from .adapters import (
    BistMarketDataAdapter,
    HttpJsonProvider,
    KapDisclosureAdapter,
    NewsAdapter,
    parse_json_records,
)
from .daily_ranker import rank_latest


@dataclass(frozen=True, slots=True)
class PipelineSnapshot:
    as_of: date
    universe_size: int
    rows: int
    ranked: pd.DataFrame
    disclosures: int
    news_items: int
    data_sources: tuple[str, ...]


def _records_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    aliases = {"time": "timestamp", "datetime": "timestamp", "ticker": "symbol", "code": "symbol"}
    frame = frame.rename(columns={k: v for k, v in aliases.items() if k in frame.columns})
    required = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"market payload missing columns: {missing}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for column in required[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=required).sort_values(["symbol", "timestamp"])


def run_universe_pipeline(
    market_adapter: BistMarketDataAdapter,
    symbols: list[str],
    as_of: date,
    kap_adapter: KapDisclosureAdapter | None = None,
    news_adapter: NewsAdapter | None = None,
) -> PipelineSnapshot:
    """Fetch every eligible symbol before ranking the complete universe."""
    records: list[dict[str, Any]] = []
    for symbol in symbols:
        payload = market_adapter.bars(symbol, "history", as_of.isoformat())
        records.extend(parse_json_records(payload))
    frame = _records_frame(records)
    ranked = rank_latest(frame)
    disclosures = 0
    news_items = 0
    sources = ["market"]
    if kap_adapter is not None:
        disclosures = len(parse_json_records(kap_adapter.disclosures(as_of.isoformat(), as_of.isoformat())))
        sources.append("KAP")
    if news_adapter is not None:
        news_items = len(parse_json_records(news_adapter.headlines(as_of.isoformat(), as_of.isoformat())))
        sources.append("news")
    return PipelineSnapshot(as_of, len(symbols), len(frame), ranked, disclosures, news_items, tuple(sources))


def build_http_market_adapter(endpoint: str) -> BistMarketDataAdapter:
    return BistMarketDataAdapter(HttpJsonProvider(endpoint))
