import json
from datetime import datetime, timedelta, timezone

from bist_hunter.event_features import NormalizedEvent
from bist_hunter.news_classifier import (
    classify_event, classify_events, market_bypass, news_score,
)

NOW = datetime(2026, 9, 30, 10, 0, tzinfo=timezone.utc)


def ev(i, text, hours_ago=1, symbol="AAA"):
    return NormalizedEvent(symbol, str(i), NOW - timedelta(hours=hours_ago), "KAP", "X", text)


def test_rules_classify_and_filter():
    c = classify_event(ev(1, "Yeni sipariş ve sözleşme imzalandı"))
    assert c.relevant and c.category == "CONTRACT" and c.polarity > 0 and c.method == "RULES"
    assert not classify_event(ev(2, "Sorumluluk reddi beyanı")).relevant
    assert classify_event(ev(3, "Şirkete dava ve haciz")).polarity < 0


def test_llm_accepted_only_if_strict_schema():
    good = json.dumps({"relevant": True, "category": "EARNINGS", "polarity": 0.5,
                       "materiality": 0.7, "confidence": 0.9})
    assert classify_event(ev(1, "bilanço açıklandı"), llm=lambda p: good).method == "LLM"
    bad = classify_event(ev(1, "bilanço açıklandı"), llm=lambda p: "kesinlikle al!")
    assert bad.method == "RULES(llm_rejected)"
    out_of_range = json.dumps({"relevant": True, "category": "EARNINGS", "polarity": 5,
                               "materiality": 0.7, "confidence": 0.9})
    assert classify_event(ev(1, "bilanço"), llm=lambda p: out_of_range).method == "RULES(llm_rejected)"

    def boom(p):
        raise RuntimeError("down")
    assert classify_event(ev(1, "bilanço"), llm=boom).method == "RULES(llm_rejected)"


def test_irrelevant_never_reaches_llm():
    calls = []
    classify_event(ev(1, "sorumluluk reddi"), llm=lambda p: calls.append(p) or "")
    assert calls == []


def test_future_events_excluded_and_score_none_without_evidence():
    future = ev(9, "temettü", hours_ago=-2)
    assert classify_events([future], NOW) == []
    assert news_score([], "AAA") is None
    classes = classify_events([ev(1, "sipariş sözleşme"), ev(2, "dava haciz", symbol="BBB")], NOW)
    assert news_score(classes, "AAA") > 50 > news_score(classes, "BBB")


def test_market_bypass_needs_two_hits_and_expires():
    one = [ev(1, "Devre kesici devreye girdi", symbol="MARKET")]
    assert not market_bypass(one, NOW).active
    two = one + [ev(2, "Ani faiz artışı kararı", symbol="MARKET")]
    b = market_bypass(two, NOW)
    assert b.active and b.until > NOW
    assert not market_bypass(two, NOW + timedelta(hours=30)).active
