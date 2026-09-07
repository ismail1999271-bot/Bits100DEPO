from datetime import date, timedelta

from bist_hunter.features import FeatureSnapshot
from bist_hunter.fund_flow import FundFlow, detect_flow_anomalies
from bist_hunter.ranking import rank_candidates
from bist_hunter.smart_money import FundHolding, estimate_sector_flows


def test_flow_anomaly_uses_previous_history_only():
    start = date(2026, 8, 1)
    rows = [FundFlow(start + timedelta(days=i), "THF", "Tera Hisse", 100_000_000, "test") for i in range(4)]
    rows.append(FundFlow(start + timedelta(days=4), "THF", "Tera Hisse", 900_000_000, "test"))
    anomalies = detect_flow_anomalies(rows, lookback=4, z_threshold=2.0)
    assert anomalies[-1].is_anomalous
    assert anomalies[-1].z_score > 2


def test_sector_flow_is_allocated_from_holdings():
    day = date(2026, 9, 4)
    flows = [FundFlow(day, "THF", "Tera Hisse", 1_000_000_000, "test")]
    holdings = [
        FundHolding(day, "THF", "AAA", 0.60, "BANKING"),
        FundHolding(day, "THF", "BBB", 0.20, "DEFENSE"),
    ]
    result = estimate_sector_flows(flows, holdings)
    assert result[0].sector == "BANKING"
    assert result[0].estimated_flow_try == 600_000_000


def test_ranking_blends_positive_and_negative_smart_money():
    f = FeatureSnapshot(0, 0.04, 2.0, .02, .9, .08, .05, 1.0, .02)
    result = rank_candidates(
        [("AAA", f), ("BBB", f)],
        min_score=0,
        smart_money_flows={"AAA": 500_000_000, "BBB": -500_000_000},
    )
    assert result.candidates[0].symbol == "AAA"
    assert "smart money accumulation" in result.candidates[0].reasons
    assert "smart money distribution" in result.candidates[1].reasons
