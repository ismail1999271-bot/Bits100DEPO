from bist_hunter.e2e import run_smoke


def test_end_to_end_smoke():
    result = run_smoke()
    assert result["rows"] > 0
    assert result["labeled_rows"] > 0
    assert result["positive_events"] > 0
    assert 0 < result["example_trade_net_return"] < 0.10
