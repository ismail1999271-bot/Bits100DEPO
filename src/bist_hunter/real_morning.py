"""06:00 real-data morning runner: universe -> intelligence -> Top-K -> Telegram."""
from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from .adapters import ProviderError
from .daily_ranker import rank_latest
from .real_adapters import fetch_fund_flow, fetch_kap_disclosures, fetch_news_rss, load_symbol_universe, yahoo_chart
from .telegram_report import send_message


def _collect_market(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        rows.extend(yahoo_chart(symbol, start, end))
    if not rows:
        raise ProviderError("no market rows returned for the configured universe")
    return pd.DataFrame(rows)


def _symbol_scores(records: list[dict[str, object]], key: str = "score") -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in records:
        symbol = str(row.get("symbol", "")).upper()
        if not symbol:
            continue
        value = row.get(key)
        try:
            scores[symbol] = float(value)
        except (TypeError, ValueError):
            continue
    return scores


def run_real_morning(as_of: date | None = None) -> str:
    day = as_of or datetime.now(UTC).date()
    symbols = load_symbol_universe()
    frame = _collect_market(symbols, day - timedelta(days=45), day)

    # Intelligence feeds are additive. Missing optional feeds never become fake scores.
    kap_rows: list[dict[str, object]] = []
    fund_rows: list[dict[str, object]] = []
    news_rows: list[dict[str, object]] = []
    if os.getenv("KAP_API_URL"):
        kap_rows = fetch_kap_disclosures(day - timedelta(days=1), day)
    if os.getenv("FUND_FLOW_API_URL"):
        fund_rows = fetch_fund_flow(day - timedelta(days=1), day)
    if os.getenv("NEWS_RSS_URL"):
        news_rows = fetch_news_rss(os.environ["NEWS_RSS_URL"])

    # Feed-specific schemas stay provider-neutral: only explicit symbol/score fields are merged.
    fund_scores = _symbol_scores(fund_rows)
    if fund_scores:
        frame["smart_money_score"] = frame["symbol"].str.upper().map(fund_scores)
    ranked = rank_latest(frame)
    lines = ["🌅 BITS100 — 06:00 GERÇEK VERİ", f"Evren: {len(symbols)} | OHLCV satırı: {len(frame)}"]
    if ranked.empty:
        lines.append("🔴 BUGÜN KALİTELİ TAVAN SİNYALİ YOK.")
    else:
        for index, row in ranked.head(20).iterrows():
            lines.append(f"{index + 1}. {row['symbol']} | skor={row.get('score', row.get('institutional_score', 0)):.2f}")
    lines.append(f"Kaynaklar: OHLCV + KAP:{len(kap_rows)} + NEWS:{len(news_rows)} + FUND:{len(fund_rows)}")
    message = "\n".join(lines)
    send_message(message)
    return message
