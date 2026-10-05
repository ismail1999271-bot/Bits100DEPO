from datetime import datetime, timedelta, timezone

from bist_hunter.fail_closed import GateInput, evaluate_gate
from bist_hunter.research_ranking import COLUMNS, SymbolResearch, rank_research
from bist_hunter.risk import RiskInputs, assess_risk
from bist_hunter.universe import build_bist100_plus_universe

NOW = datetime(2026, 9, 11, 7, 0, tzinfo=timezone.utc)
OK_PROVIDERS = {"ohlcv": "CONNECTED"}


def gate(symbol, age=10, providers=OK_PROVIDERS, **kw):
    return GateInput(symbol, NOW, NOW - timedelta(seconds=age), 60, None, provider_statuses=providers, **kw)


def test_gate_reasons():
    assert evaluate_gate(GateInput("A", NOW, NOW, 60, 0.5, provider_statuses=OK_PROVIDERS)).passed
    cases = {
        "MISSING_MARKET_DATA": GateInput("A", NOW, None, 60, 0.5, provider_statuses=OK_PROVIDERS),
        "STALE_DATA": GateInput("A", NOW, NOW - timedelta(hours=1), 60, 0.5, provider_statuses=OK_PROVIDERS),
        "FUTURE_TIMESTAMP": GateInput("A", NOW, NOW + timedelta(minutes=5), 60, 0.5, provider_statuses=OK_PROVIDERS),
        "INVALID_TIMESTAMP": GateInput("A", NOW, "not-a-date", 60, 0.5, provider_statuses=OK_PROVIDERS),
        "MISSING_CREDENTIALS": GateInput("A", NOW, NOW, 60, 0.5, provider_statuses={"ohlcv": "BLOCKED"}),
        "PROVIDER_CONTRACT_VIOLATION": GateInput("A", NOW, NOW, 60, 0.5, provider_statuses={"ohlcv": "INVALID"}),
        "DUPLICATE_EVENT": GateInput("A", NOW, NOW, 60, 0.5, provider_statuses=OK_PROVIDERS, duplicate_events=1),
        "INSUFFICIENT_COVERAGE": GateInput("A", NOW, NOW, 60, 0.1, provider_statuses=OK_PROVIDERS),
        "BAD_DATA_QUALITY": GateInput("A", NOW, NOW, 60, 0.5, provider_statuses=OK_PROVIDERS,
                                      quality_flags=("ohlc_gap",)),
    }
    for reason, item in cases.items():
        result = evaluate_gate(item)
        assert result.status == "BLOCKED" and reason in result.reasons, reason


def test_risk_references():
    report = assess_risk(RiskInputs(last_price=100, atr=2, support=98, reference_close=95, spread_bps=10,
                                    avg_daily_value_try=50_000_000, top_of_book_depth_try=500_000))
    assert report.entry_reference == 100
    assert report.stop_reference == 97.51  # max(100-3, 98*0.995)
    assert report.target_reference == 104.5  # min(100+6, 95*1.10)
    assert report.risk_reward == round(4.5 / 2.49, 4)
    assert report.spread_risk == "LOW" and report.liquidity_risk == "LOW"
    assert assess_risk(RiskInputs(last_price=None)).status == "BLOCKED_MISSING_MARKET_DATA"
    unknown = assess_risk(RiskInputs(last_price=10))
    assert unknown.overall == "UNKNOWN" and "liquidity_unknown" in unknown.reasons


def test_ranking_orders_pass_before_blocked_and_covers_universe():
    universe = build_bist100_plus_universe(["AAA", "BBB", "CCC"], ["DDD"], as_of="2026-09-11", source="t")
    items = [
        SymbolResearch("AAA", gate("AAA"), {"auction": 80, "tavan_dna": 70, "technical": 60},
                       relative_volume=2.1, risk=RiskInputs(last_price=10, atr=0.3)),
        SymbolResearch("BBB", gate("BBB"), {"auction": 95, "tavan_dna": 90}),
        SymbolResearch("CCC", gate("CCC", age=9999), {"auction": 99, "tavan_dna": 99}),
    ]
    table = rank_research(universe, items)
    assert tuple(table.columns) == COLUMNS
    assert list(table["Symbol"]) == ["BBB", "AAA", "CCC", "DDD"]
    assert list(table["Rank"]) == [1, 2, 3, 4]
    assert table.loc[2, "Status"] == "BLOCKED" and "STALE_DATA" in table.loc[2, "Reasons"]
    assert table.loc[2, "Quant Score"] is None or table.loc[2, "Quant Score"] != table.loc[2, "Quant Score"]
    assert table.loc[3, "Reasons"] == "MISSING_MARKET_DATA"
    assert "tahmin" in table.attrs["disclaimer"]
