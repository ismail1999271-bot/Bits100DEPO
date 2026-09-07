from bist_hunter.broker_consensus import BrokerView, consensus
from bist_hunter.production_readiness import readiness_report


def test_broker_consensus_counts_each_broker_once():
    views = [
        BrokerView("A", "AAA", "BUY"),
        BrokerView("A", "AAA", "SELL"),  # latest view replaces same broker
        BrokerView("B", "AAA", "BUY"),
    ]
    assert consensus(views)["AAA"] == 50.0


def test_readiness_fails_closed_without_secrets(monkeypatch):
    for name in (
        "BIST_MARKET_DATA_URL", "KAP_API_URL", "FUND_FLOW_API_URL",
        "BROKER_DATA_URL", "INSTITUTIONAL_DATA_URL", "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    gates = {gate.name: gate for gate in readiness_report()}
    assert not gates["BIST market data"].ready
    assert not gates["Claude research agent"].ready
