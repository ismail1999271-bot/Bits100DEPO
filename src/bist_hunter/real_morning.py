"""06:00 real-data runner: universe -> intelligence -> Top-K -> Telegram."""
from __future__ import annotations

import math
import os
import re
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from .adapters import ProviderError
from .daily_ranker import rank_latest
from .real_adapters import fetch_fund_flow, fetch_kap_disclosures, fetch_news_rss, load_symbol_universe, yahoo_chart
from .telegram_report import send_message

BULLISH_TERMS = ("yatırım", "sözleşme", "sipariş", "kar", "temettü", "geri alım", "ihale")
BEARISH_TERMS = ("zarar", "iflas", "dava", "soruşturma", "sermaye artırımı")


def _collect_market(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        rows.extend(yahoo_chart(symbol, start, end))
    if not rows:
        raise ProviderError("no market rows returned for the configured universe")
    return pd.DataFrame(rows)


def _event_scores(rows: list[dict[str, object]], text_keys: tuple[str, ...]) -> dict[str, float]:
    scores: dict[str, list[float]] = {}
    for row in rows:
        text = " ".join(str(row.get(key, "")) for key in text_keys).lower()
        value = sum(term in text for term in BULLISH_TERMS) - sum(term in text for term in BEARISH_TERMS)
        for symbol in re.findall(r"\b[A-ZÇĞİÖŞÜ]{3,6}\b", text.upper()):
            if symbol not in {"KAP", "BIST", "RSS"}:
                scores.setdefault(symbol, []).append(float(value))
    return {symbol: max(0.0, min(100.0, 50.0 + 12.5 * sum(values) / max(1, len(values)))) for symbol, values in scores.items()}


def _fund_scores(rows: list[dict[str, object]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        symbol = str(row.get("symbol", "")).upper()
        if not symbol:
            continue
        try:
            raw = float(row.get("score", row.get("net_flow", 0.0)))
        except (TypeError, ValueError):
            continue
        if "score" not in row:
            raw = 50.0 + 50.0 * math.tanh(raw / 100_000_000)
        out[symbol] = max(0.0, min(100.0, raw))
    return out


def run_real_morning(as_of: date | None = None) -> str:
    day = as_of or datetime.now(UTC).date()
    symbols = load_symbol_universe()
    frame = _collect_market(symbols, day - timedelta(days=45), day)
    kap_rows: list[dict[str, object]] = []
    fund_rows: list[dict[str, object]] = []
    news_rows: list[dict[str, object]] = []
    if os.getenv("KAP_API_URL"):
        kap_rows = fetch_kap_disclosures(day - timedelta(days=1), day)
    if os.getenv("FUND_FLOW_API_URL"):
        fund_rows = fetch_fund_flow(day - timedelta(days=1), day)
    if os.getenv("NEWS_RSS_URL"):
        news_rows = fetch_news_rss(os.environ["NEWS_RSS_URL"])

    scores = (
        ("smart_money_score", _fund_scores(fund_rows)),
        ("fundamental_score", _event_scores(kap_rows, ("title", "headline", "subject", "text"))),
        ("research_score", _event_scores(news_rows, ("title", "description", "summary"))),
    )
    for column, mapping in scores:
        if mapping:
            frame[column] = frame["symbol"].str.upper().map(mapping)
    ranked = rank_latest(frame)
    lines = ["🌅 BITS100 — 06:00 GERÇEK VERİ", f"Evren: {len(symbols)} | OHLCV satırı: {len(frame)}"]
    if ranked.empty:
        lines.append("🔴 BUGÜN KALİTELİ TAVAN SİNYALİ YOK.")
    else:
        for index, row in ranked.head(20).iterrows():
            score = float(row.get("institutional_score", row.get("score", 0.0)))
            lines.append(f"{index + 1}. {row['symbol']} | skor={score:.2f}")
    lines.append(f"Kaynaklar: OHLCV + KAP:{len(kap_rows)} + NEWS:{len(news_rows)} + FUND:{len(fund_rows)}")
    message = "\n".join(lines)
    send_message(message)
    return message
