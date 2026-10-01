from datetime import date, datetime, timezone

from bist_hunter.tavan_risk import (
    TavanRiskInputs, assess_tavan_risk, hype_count, limit_up_streak, lock_strength, manipulation_risk,
)

NOW = datetime(2026, 9, 11, 7, 0, tzinfo=timezone.utc)


def test_streak_and_lock_strength():
    closes = (100, 101, 111.1, 122.2, 134.4)  # last three closes are +10%
    assert limit_up_streak(closes, 0.10) == 3
    assert limit_up_streak((100, 100.5), 0.10) == 0
    assert lock_strength(9_000, 1_000) == 90.0
    assert lock_strength(None, 1_000) is None
    assert lock_strength(0, 0) is None


def test_hard_blocks():
    base = dict(symbol="aaa", now=NOW)
    assert assess_tavan_risk(TavanRiskInputs(**base, vbts_measures=("tek_fiyat",))).status == "BLOCKED"
    assert "TRADING_HALT_OR_CIRCUIT_BREAKER" in assess_tavan_risk(
        TavanRiskInputs(**base, trading_state="circuit_breaker", vbts_measures=())).block_reasons
    assert "IPO_LOCKUP" in assess_tavan_risk(
        TavanRiskInputs(**base, vbts_measures=(), ipo_lockup_until=date(2026, 9, 20))).block_reasons
    ok = assess_tavan_risk(TavanRiskInputs(**base, vbts_measures=()))
    assert ok.status == "OK" and ok.next_day_break_risk == "UNKNOWN" and ok.lock_strength is None


def test_warnings_unknowns_and_calendar():
    risk = assess_tavan_risk(TavanRiskInputs(
        symbol="AAA", now=NOW, vbts_measures=("brut_takas",), free_float_ratio=0.08,
        daily_closes=(100, 110, 121, 133.1, 146.4), bid_queue_qty=1_000, ask_qty_at_limit=2_000,
        upcoming_events=(("BEDELSIZ", date(2026, 9, 12)), ("TEMETTU", date(2026, 10, 30)))))
    assert risk.status == "OK" and risk.next_day_break_risk == "HIGH" and risk.free_float_flag == "LOW_FLOAT"
    assert {"vbts_brut_takas", "weak_limit_lock", "low_free_float", "corporate_event_soon"} <= set(risk.warnings)
    assert risk.calendar_flags == ("BEDELSIZ@2026-09-12",)
    assert risk.lock_probability_hint == round(1 / 3, 4)
    assert assess_tavan_risk(TavanRiskInputs(symbol="A", now=NOW)).warnings == ("vbts_unknown",)


def test_manipulation_risk_needs_two_inputs_and_hype_count():
    assert manipulation_risk(3.0, None, None, None) is None
    assert manipulation_risk(4.0, 10, 5.0, 0.0) == 1.0
    assert hype_count(["Bu hisse ROKET olacak", "bilanço açıklandı", "kesin tavan"]) == 2


def test_ranking_blocks_vbts_even_with_high_score():
    from datetime import timedelta

    from bist_hunter.fail_closed import GateInput
    from bist_hunter.research_ranking import SymbolResearch, rank_research
    from bist_hunter.universe import build_bist100_plus_universe

    universe = build_bist100_plus_universe(["AAA", "BBB"], [], as_of="2026-09-11", source="t")
    gate = lambda s: GateInput(s, NOW, NOW - timedelta(seconds=5), 60, None,  # noqa: E731
                               provider_statuses={"ohlcv": "CONNECTED"})
    comps = {"auction": 95, "tavan_dna": 95}
    table = rank_research(universe, [
        SymbolResearch("AAA", gate("AAA"), comps,
                       tavan_risk=assess_tavan_risk(TavanRiskInputs("AAA", NOW, vbts_measures=("TEK_FIYAT",)))),
        SymbolResearch("BBB", gate("BBB"), {"auction": 75, "tavan_dna": 75},
                       tavan_risk=assess_tavan_risk(TavanRiskInputs("BBB", NOW, vbts_measures=()))),
    ])
    assert list(table["Symbol"]) == ["BBB", "AAA"]
    assert table.loc[1, "Status"] == "BLOCKED" and "BLOCK:VBTS_TEK_FIYAT" in table.loc[1, "Reasons"]


def test_market_bypass_blocks_whole_ranking():
    from datetime import datetime, timezone

    from bist_hunter.fail_closed import GateInput
    from bist_hunter.news_classifier import MarketBypass
    from bist_hunter.research_ranking import SymbolResearch, rank_research
    from bist_hunter.universe import Universe

    now = datetime(2026, 9, 30, 10, tzinfo=timezone.utc)
    uni = Universe("T", ("AAA",), "2026-09-30", "test")
    item = SymbolResearch("AAA", GateInput("AAA", now, now, 3600, None), {"technical": 90.0})
    out = rank_research(uni, [item], bypass=MarketBypass(True, ("x",), now))
    assert out.iloc[0]["Status"] == "BLOCKED" and "MARKET_BYPASS" in out.iloc[0]["Reasons"]
