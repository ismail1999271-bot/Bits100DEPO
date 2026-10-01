"""Two-stage news/KAP classification and the market-wide bypass (research only).

Stage 1 filters irrelevant items, stage 2 classifies category, polarity and
materiality. Without an LLM the stage runs on transparent keyword rules. An LLM
may be injected as ``llm(prompt) -> str`` (any provider); its answer is accepted
only if it is strict JSON that satisfies the schema, otherwise the rule result
is used and flagged ``llm_rejected``. The classifier never creates events, never
sees data newer than ``as_of`` and its output is a *feature*, not a trade order.

Market bypass: a point-in-time check for market-wide stress headlines. When it
fires, the ranking shows every symbol as BLOCKED (reason MARKET_BYPASS) until
``until`` - the research equivalent of "do not act on today's signals".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Iterable

from .event_features import BEARISH, BULLISH, NormalizedEvent, point_in_time

CATEGORIES = ("EARNINGS", "CONTRACT", "CAPITAL", "LEGAL", "DIVIDEND", "OWNERSHIP", "MACRO", "OTHER")
CATEGORY_TERMS = {
    "EARNINGS": ("kâr", "kar artışı", "zarar", "bilanço", "finansal sonuç", "ciro"),
    "CONTRACT": ("sözleşme", "sipariş", "ihale", "anlaşma", "iş ilişkisi"),
    "CAPITAL": ("bedelli", "bedelsiz", "sermaye artırımı", "geri alım", "tahsisli"),
    "LEGAL": ("dava", "soruşturma", "ceza", "haciz", "konkordato", "işlem yasağı", "tedbir", "iflas"),
    "DIVIDEND": ("temettü", "kâr payı"),
    "OWNERSHIP": ("pay alım", "pay satım", "ortaklık yapısı", "yönetim kurulu", "devralma"),
    "MACRO": ("faiz", "enflasyon", "merkez bankası", "kur", "savaş", "kriz"),
}
IRRELEVANT_TERMS = ("sorumluluk reddi", "genel kurul çağrısı", "bağımsız denetim kuruluşu seçimi")
MARKET_STRESS_TERMS = ("devre kesici", "piyasa çöktü", "savaş", "acil toplantı", "döviz krizi", "borsa kapatıldı",
                       "işlemler durduruldu", "ani faiz artışı", "kur şoku")
PROMPT = (
    "Aşağıdaki Türkçe KAP/haber başlığını sınıflandır. SADECE JSON döndür: "
    '{{"relevant": bool, "category": one of {cats}, "polarity": -1..1, "materiality": 0..1, '
    '"confidence": 0..1}}. Başlık: {headline}'
)


@dataclass(frozen=True, slots=True)
class Classification:
    event_id: str
    symbol: str
    relevant: bool
    category: str
    polarity: float
    materiality: float
    confidence: float
    method: str  # RULES / LLM / RULES(llm_rejected)


def _rules(event: NormalizedEvent) -> Classification:
    text = event.headline.casefold()
    if any(t in text for t in IRRELEVANT_TERMS):
        return Classification(event.event_id, event.symbol, False, "OTHER", 0.0, 0.0, 0.5, "RULES")
    category = "OTHER"
    best = 0
    for cat, terms in CATEGORY_TERMS.items():
        hits = sum(t in text for t in terms)
        if hits > best:
            category, best = cat, hits
    bulls = sum(t in text for t in BULLISH)
    bears = sum(t in text for t in BEARISH)
    total = bulls + bears
    polarity = 0.0 if total == 0 else (bulls - bears) / total
    materiality = min(1.0, 0.25 * (best + total))
    confidence = round(min(0.8, 0.3 + 0.15 * (best + total)), 2)  # rules never claim high confidence
    return Classification(event.event_id, event.symbol, True, category, round(polarity, 3),
                          round(materiality, 3), confidence, "RULES")


def _parse_llm(text: str, event: NormalizedEvent) -> Classification | None:
    try:
        data = json.loads(text)
        relevant = data["relevant"]
        category = data["category"]
        polarity, materiality, confidence = (float(data[k]) for k in ("polarity", "materiality", "confidence"))
    except Exception:
        return None
    if not isinstance(relevant, bool) or category not in CATEGORIES:
        return None
    if not (-1 <= polarity <= 1 and 0 <= materiality <= 1 and 0 <= confidence <= 1):
        return None
    return Classification(event.event_id, event.symbol, relevant, category, polarity, materiality,
                          confidence, "LLM")


def classify_event(event: NormalizedEvent, llm: Callable[[str], str] | None = None) -> Classification:
    base = _rules(event)
    if llm is None:
        return base
    if not base.relevant:  # stage 1: obviously irrelevant items never reach the LLM
        return base
    try:
        answer = llm(PROMPT.format(cats="/".join(CATEGORIES), headline=event.headline))
    except Exception:
        return _flag(base)
    parsed = _parse_llm(answer, event)
    return parsed if parsed is not None else _flag(base)


def _flag(base: Classification) -> Classification:
    return Classification(base.event_id, base.symbol, base.relevant, base.category, base.polarity,
                          base.materiality, base.confidence, "RULES(llm_rejected)")


def classify_events(events: Iterable[NormalizedEvent], as_of: datetime,
                    llm: Callable[[str], str] | None = None) -> list[Classification]:
    return [classify_event(e, llm) for e in point_in_time(events, as_of)]


def news_score(classes: Iterable[Classification], symbol: str) -> float | None:
    """0..100 from relevant, confidence-and-materiality weighted polarity; None when no evidence."""
    rel = [c for c in classes if c.symbol == symbol.upper() and c.relevant]
    if not rel:
        return None
    weights = [max(1e-6, c.materiality * c.confidence) for c in rel]
    mean = sum(w * c.polarity for w, c in zip(weights, rel)) / sum(weights)
    return round(max(0.0, min(100.0, 50.0 + 50.0 * mean)), 2)


@dataclass(frozen=True, slots=True)
class MarketBypass:
    active: bool
    reasons: tuple[str, ...]
    until: datetime | None


def market_bypass(events: Iterable[NormalizedEvent], as_of: datetime, *, hold: timedelta = timedelta(hours=24),
                  min_hits: int = 2) -> MarketBypass:
    """Active when >= ``min_hits`` distinct market-stress headlines appeared within ``hold`` before ``as_of``."""
    hits: list[tuple[datetime, str]] = []
    for e in point_in_time(events, as_of):
        if as_of - e.published_at > hold:
            continue
        text = e.headline.casefold()
        for term in MARKET_STRESS_TERMS:
            if term in text:
                hits.append((e.published_at, f"{term}: {e.headline[:80]}"))
                break
    if len(hits) < min_hits:
        return MarketBypass(False, (), None)
    return MarketBypass(True, tuple(h[1] for h in hits[:5]), max(h[0] for h in hits) + hold)
