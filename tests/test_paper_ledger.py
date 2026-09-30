import json
from datetime import datetime, timedelta, timezone

import pytest

from bist_hunter.paper_ledger import CostModel, PaperLedger
from bist_hunter.paper_trading import RiskLimits

T0 = datetime(2026, 9, 11, 7, 0, tzinfo=timezone.utc)
ADV = 1_000_000_000.0


def ledger():
    return PaperLedger(1_000_000, RiskLimits(max_position_pct=0.2, max_order_notional_try=500_000),
                       CostModel(commission_bps=4, slippage_bps=5, min_spread_bps=10))


def test_round_trip_records_costs_and_pnl():
    book = ledger()
    assert book.buy(T0, "thyao", 1000, 100.0, "quant_score>=70", avg_daily_value_try=ADV)
    buy = book.entries[-1]
    assert buy.fill_price == pytest.approx(100 * (1 + 10 / 10_000))  # 5 bps half-spread + 5 bps slippage
    assert buy.fees == pytest.approx(buy.notional * 4 / 10_000, rel=1e-6)
    assert buy.cash_after == pytest.approx(1_000_000 - buy.notional - buy.fees, rel=1e-9)
    assert buy.position_after == 1000 and buy.reason == "quant_score>=70"
    assert book.sell(T0 + timedelta(hours=3), "THYAO", 110.0, "target")
    sell = book.entries[-1]
    expected = (sell.fill_price - buy.fill_price) * 1000 - sell.fees - buy.fees
    assert sell.realized_pnl == pytest.approx(expected, abs=0.01)
    assert sell.exit_price == sell.fill_price and sell.entry_price == pytest.approx(buy.fill_price)
    summary = book.summary()
    assert summary["closed_trades"] == 1 and summary["open_positions"] == 0
    assert summary["cash"] == pytest.approx(1_000_000 + expected, abs=0.01)
    assert summary["fees"] > 0 and summary["slippage"] > 0


def test_limits_liquidity_and_backdating():
    book = ledger()
    assert not book.buy(T0, "A", 10, 100.0, "x")  # unknown liquidity blocks
    assert book.entries[-1].reason == "LIQUIDITY_UNKNOWN"
    assert not book.buy(T0, "A", 1000, 100.0, "x", avg_daily_value_try=1_000_000)
    assert book.entries[-1].reason == "LIQUIDITY_LIMIT"
    assert not book.buy(T0, "A", 6000, 100.0, "x", avg_daily_value_try=ADV)
    assert book.entries[-1].reason == "ORDER_NOTIONAL_LIMIT"
    assert not book.sell(T0, "ZZZ", 10.0, "x")
    with pytest.raises(ValueError):
        book.buy(T0 - timedelta(seconds=1), "A", 1, 1.0, "x", avg_daily_value_try=ADV)
    with pytest.raises(ValueError):
        book.buy(datetime(2026, 9, 11, 10), "A", 1, 1.0, "x", avg_daily_value_try=ADV)


def test_jsonl_export(tmp_path):
    book = ledger()
    book.buy(T0, "A", 10, 100.0, "x", avg_daily_value_try=ADV)
    path = tmp_path / "ledger.jsonl"
    book.write_jsonl(path)
    row = json.loads(path.read_text().splitlines()[0])
    for key in ("cash_after", "position_after", "entry_price", "exit_price", "realized_pnl", "fees",
                "slippage", "timestamp", "reason"):
        assert key in row
