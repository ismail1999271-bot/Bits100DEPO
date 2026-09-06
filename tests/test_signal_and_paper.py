from bist_hunter.paper_trading import PaperBroker, RiskLimits
from bist_hunter.scoring import CandidateFeatures
from bist_hunter.signal_engine import SignalConfig, build_opportunity, rank_opportunities


def test_quality_gate_allows_strong_and_rejects_risky():
    strong = CandidateFeatures(volume_anomaly=1, price_momentum=1, flow_strength=1, catalyst_strength=1)
    risky = CandidateFeatures(volume_anomaly=1, price_momentum=1, flow_strength=1, catalyst_strength=1, manipulation_risk=1)
    assert build_opportunity("AAA", strong, SignalConfig(min_score=70)).quality
    assert not build_opportunity("BBB", risky, SignalConfig(max_manipulation_risk=0.65)).quality


def test_ranker_does_not_force_low_quality_trade():
    weak = CandidateFeatures()
    result = rank_opportunities([("AAA", weak)])
    assert result[0].quality is False


def test_paper_broker_enforces_position_limit_and_audit():
    broker = PaperBroker(100_000, RiskLimits(max_position_pct=0.10, max_order_notional_try=20_000))
    assert broker.buy("AAA", 100, 100, "test")
    assert not broker.buy("BBB", 300, 100, "too large")
    assert broker.sell("AAA", 110, "target")
    assert broker.equity_try == 101_000
    assert [x.action for x in broker.audit] == ["BUY", "SELL"]
