"""06:00 real-data research runner; transport stays optional until Telegram activation."""
from __future__ import annotations

import math
import os
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from .adapters import ProviderError
from .daily_ranker import rank_latest
from .real_adapters import fetch_fund_flow, fetch_kap_disclosures, fetch_news_rss, historical_bars, load_symbol_universe
from .telegram_report import send_message

BULLISH_TERMS = ("yatırım", "sözleşme", "sipariş", "kar", "kâr", "temettü", "geri alım", "ihale")
BEARISH_TERMS = ("zarar", "iflas", "dava", "soruşturma", "sermaye artırımı")


def _collect_market(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    failures = 0
    for symbol in symbols:
        try:
            rows.extend(historical_bars(symbol, start, end))
        except ProviderError:
            failures += 1
    if not rows:
        raise ProviderError(f"no market rows returned; failed symbols={failures}")
    return pd.DataFrame(rows)


def _event_scores(rows: list[dict[str, object]], text_keys: tuple[str, ...]) -> dict[str, float]:
    """Score only records carrying an explicit symbol; never infer tickers from prose."""
    grouped: dict[str, list[float]] = {}
    for row in rows:
        symbol = str(row.get("symbol", "")).upper().strip().removesuffix(".IS")
        if not symbol:
            continue
        text = " ".join(str(row.get(key, "")) for key in text_keys).lower()
        value = sum(term in text for term in BULLISH_TERMS) - sum(term in text for term in BEARISH_TERMS)
        grouped.setdefault(symbol, []).append(float(value))
    return {s: max(0.0, min(100.0, 50.0 + 12.5 * sum(v) / max(1, len(v)))) for s, v in grouped.items()}


def _fund_scores(rows: list[dict[str, object]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        symbol = str(row.get("symbol", "")).upper().strip().removesuffix(".IS")
        if not symbol:
            continue
        raw_value = row.get("score")
        if raw_value is None:
            try:
                raw_value = 50.0 + 50.0 * math.tanh(float(row.get("net_flow", 0.0)) / 100_000_000)
            except (TypeError, ValueError):
                continue
        try:
            out[symbol] = max(0.0, min(100.0, float(raw_value)))
        except (TypeError, ValueError):
            continue
    return out


def run_real_morning(as_of: date | None = None, *, send_telegram: bool = False) -> str:
    day = as_of or datetime.now(UTC).date()
    symbols = load_symbol_universe()
    frame = _collect_market(symbols, day - timedelta(days=90), day)
    kap_rows = fetch_kap_disclosures(day - timedelta(days=1), day) if os.getenv("KAP_API_URL") else []
    fund_rows = fetch_fund_flow(day - timedelta(days=1), day) if os.getenv("FUND_FLOW_API_URL") else []
    news_rows = fetch_news_rss(os.environ["NEWS_RSS_URL"]) if os.getenv("NEWS_RSS_URL") else []

    frame["smart_money_score"] = frame["symbol"].str.upper().map(_fund_scores(fund_rows))
    frame["fundamental_score"] = frame["symbol"].str.upper().map(_event_scores(kap_rows, ("title", "text")))
    frame["research_score"] = frame["symbol"].str.upper().map(_event_scores(news_rows, ("title", "description")))
    ranked = rank_latest(frame)
    lines = ["🌅 BITS100 — 06:00 GERÇEK VERİ", f"Evren: {len(symbols)} | OHLCV satırı: {len(frame)}"]
    if ranked.empty:
        lines.append("🔴 BUGÜN KALİTELİ TAVAN SİNYALİ YOK.")
    else:
        for index, row in ranked.head(20).iterrows():
            score = float(row.get("institutional_score", row.get("score", 0.0)))
            coverage = float(row.get("institutional_data_coverage", 0.0))
            lines.append(f"{index + 1}. {row['symbol']} | skor={score:.2f} | veri-kapsamı={coverage:.0%}")
    lines.append(f"Kaynaklar: HISTORICAL + KAP:{len(kap_rows)} + NEWS:{len(news_rows)} + FUND:{len(fund_rows)}")
    message = "\n".join(lines)
    if send_telegram:
        send_message(message)
    return message
