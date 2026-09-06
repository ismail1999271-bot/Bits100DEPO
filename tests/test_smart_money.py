from datetime import date

import pytest

from bist_hunter.fund_flow import FundFlow
from bist_hunter.smart_money import FundHolding, estimate_stock_flows, stock_smart_money_score


def test_estimate_stock_flows_allocates_fund_flow_by_holdings():
    day = date(2026, 9, 2)
    flows = [
        FundFlow(day, "THF", "Fon A", 1_000_000_000, "test"),
        FundFlow(day, "DFI", "Fon B", -500_000_000, "test"),
    ]
    holdings = [
        FundHolding(day, "THF", "AAA", 0.50),
        FundHolding(day, "THF", "BBB", 0.20),
        FundHolding(day, "DFI", "AAA", 0.40),
    ]
    result = estimate_stock_flows(flows, holdings)
    assert result[0].symbol == "AAA"
    assert result[0].estimated_flow_try == 300_000_000
    assert result[0].contributing_funds == ("DFI", "THF")
    assert result[1].symbol == "BBB"
    assert result[1].estimated_flow_try == 200_000_000


def test_stock_flow_requires_matching_date_and_bounded_score():
    day = date(2026, 9, 2)
    flows = [FundFlow(day, "THF", "Fon A", 1_000_000_000, "test")]
    holdings = [FundHolding(date(2026, 9, 1), "THF", "AAA", 1.0)]
    assert estimate_stock_flows(flows, holdings) == []
    assert -1 < stock_smart_money_score(-1_000_000_000) < 0
    assert 0 < stock_smart_money_score(1_000_000_000) < 1


def test_invalid_inputs_fail_fast():
    day = date(2026, 9, 2)
    with pytest.raises(ValueError):
        estimate_stock_flows([], [])
    with pytest.raises(ValueError):
        stock_smart_money_score(1, 0)
    with pytest.raises(ValueError):
        FundHolding(day, "THF", "AAA", 1.1)
