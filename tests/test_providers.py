from datetime import date

import pytest

from bist_hunter.data.providers import InjectedFundFlowProvider
from bist_hunter.fund_flow import FundFlow


def test_injected_fund_flow_provider_returns_normalized_rows():
    day = date(2026, 9, 2)
    provider = InjectedFundFlowProvider(lambda start, end: [FundFlow(day, "THF", "Fon A", 100.0, "test")])
    rows = provider.fund_flows(day, day)
    assert rows[0].fund_code == "THF"
    assert rows[0].net_flow_try == 100.0


def test_fund_flow_provider_requires_fetcher():
    provider = InjectedFundFlowProvider()
    with pytest.raises(RuntimeError, match="fund-flow fetcher"):
        provider.fund_flows(date(2026, 9, 2), date(2026, 9, 2))
