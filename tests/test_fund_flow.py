from datetime import date

import pytest

from bist_hunter.fund_flow import FundFlow, smart_money_score, summarize_fund_flows


def test_fund_flow_summary_separates_inflows_and_outflows():
    day = date(2026, 9, 2)
    rows = [
        FundFlow(day, "THF", "Fon A", 2_890_000_000, "test"),
        FundFlow(day, "DFI", "Fon B", -3_050_000_000, "test"),
        FundFlow(day, "DOH", "Fon C", 860_000_000, "test"),
    ]
    summary = summarize_fund_flows(rows)
    assert summary.total_inflow_try == 3_750_000_000
    assert summary.total_outflow_try == 3_050_000_000
    assert summary.net_flow_try == 700_000_000
    assert summary.inflow_funds == ("THF", "DOH")
    assert summary.outflow_funds == ("DFI",)


def test_smart_money_score_is_bounded_and_monotonic():
    assert smart_money_score(-10_000_000_000) < smart_money_score(0) < smart_money_score(10_000_000_000)
    assert -1 < smart_money_score(-10_000_000_000) < 0
    assert 0 < smart_money_score(10_000_000_000) < 1


def test_invalid_scale_and_empty_data_fail_fast():
    with pytest.raises(ValueError):
        smart_money_score(1, 0)
    with pytest.raises(ValueError):
        summarize_fund_flows([])
