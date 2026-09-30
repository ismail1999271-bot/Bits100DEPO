from datetime import datetime, timedelta, timezone

from bist_hunter.paper_ledger import CostModel, PaperLedger
from bist_hunter.paper_trading import RiskLimits
from bist_hunter.trade_review import TradePlan, build_round_trips, review_trades

T0 = datetime(2026, 9, 11, 7, 0, tzinfo=timezone.utc)
ADV = 5_000_000_000.0


def ledger():
    return PaperLedger(1_000_000, RiskLimits(max_position_pct=0.5, max_order_notional_try=900_000),
                       CostModel(commission_bps=0, slippage_bps=0, min_spread_bps=0))


def trade(book, sym, minutes, qty, buy, sell, hold=30, reason_out="manual"):
    t = T0 + timedelta(minutes=minutes)
    assert book.buy(t, sym, qty, buy, "quant_score", avg_daily_value_try=ADV)
    assert book.sell(t + timedelta(minutes=hold), sym, sell, reason_out)
    return t


def test_round_trips_fifo_pnl_matches_ledger():
    book = ledger()
    trade(book, "AAA", 0, 100, 100.0, 110.0)
    trips = build_round_trips(book.entries)
    assert len(trips) == 1 and round(trips[0].pnl, 2) == 1000.0
    assert round(sum(t.pnl for t in trips), 2) == round(book.summary()["realized_pnl"], 2)


def test_patterns_and_process_vs_outcome():
    book = ledger()
    t1 = trade(book, "AAA", 0, 100, 100.0, 110.0)        # win, exits before half target, chased
    trade(book, "BBB", 20000, 100, 50.0, 45.0, hold=10)   # loss
    t3 = trade(book, "CCC", 20020, 300, 50.0, 49.0)       # revenge: soon after loss, bigger, loses
    t4 = trade(book, "DDD", 40000, 10, 10.0, 9.0)         # clean small loss
    plans = {("AAA", t1.isoformat()): TradePlan(target=130.0, prior_return=0.09),
             ("CCC", t3.isoformat()): TradePlan(stop=49.5),
             ("DDD", t4.isoformat()): TradePlan(stop=8.9, prior_return=0.0)}
    report = review_trades(book.entries, initial_equity=1_000_000, plans=plans, max_position_pct=0.5)
    flags = {v.trip.symbol: v.flags for v in report.verdicts}
    assert flags["AAA"] == ("CHASE", "EARLY_EXIT")
    assert "REVENGE_TRADE" in flags["CCC"] and "STOP_VIOLATION" in flags["CCC"]
    assert flags["BBB"] == () and flags["DDD"] == ("STOP_VIOLATION",) or flags["DDD"] == ()
    by = {v.trip.symbol: v.classification for v in report.verdicts}
    assert by["AAA"] == "BAD_PROCESS_LUCKY_OUTCOME" and by["BBB"] == "GOOD_PROCESS_BAD_OUTCOME"
    assert report.good_process_bad_outcome >= 1 and report.bad_process_good_outcome == 1
    assert 0 <= report.process_score <= 100 and len(report.rules) == 5
    assert any("5 günde" in r for r in report.rules)


def test_no_trades():
    book = ledger()
    assert review_trades(book.entries, initial_equity=1_000_000).status == "NO_CLOSED_TRADES"
