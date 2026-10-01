from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from bist_hunter.live_board import alerts, build_board, change_histogram, fetch_quotes, movers, summarize
from bist_hunter.live_feed import load_live_snapshot

NOW = datetime(2026, 9, 30, 10, 0, tzinfo=timezone.utc)


class FakeTicker:
    """TEST DOUBLE ONLY: synthetic quotes to exercise the code paths."""
    DATA = {
        "AAA": dict(last_price=11.0, previous_close=10.0, open=10.4, day_high=11.0, day_low=10.3, volume=5000.0),
        "BBB": dict(last_price=9.0, previous_close=10.0, open=9.8, day_high=9.9, day_low=9.0, volume=2000.0),
        "CCC": dict(last_price=10.05, previous_close=10.0, open=10.0, day_high=10.1, day_low=9.95, volume=900.0),
        "BAD": dict(last_price=None, previous_close=10.0),
    }

    def __init__(self, symbol):
        self.symbol = symbol
        if symbol == "BOOM":
            raise RuntimeError("boom")

    @property
    def fast_info(self):
        return self.DATA[self.symbol]

    def history(self, **kw):
        n = 80
        close = np.linspace(8, 10, n)
        idx = pd.date_range(end="2026-09-29", periods=n)
        return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                             "Volume": np.full(n, 1000.0)}, index=idx)


def test_fetch_quotes_skips_unusable_and_failed():
    q, skipped = fetch_quotes(["AAA", "BAD", "BOOM", "ZZZ"], FakeTicker, NOW)
    assert list(q["symbol"]) == ["AAA"]
    assert set(skipped) == {"BAD", "BOOM", "ZZZ"}


def test_board_metrics_tavan_taban_and_staleness():
    q, _ = fetch_quotes(["AAA", "BBB", "CCC"], FakeTicker, NOW)
    q.loc[q["symbol"] == "CCC", "timestamp"] = NOW - timedelta(hours=2)
    b = build_board(q, None, now=NOW).set_index("Sembol")
    assert b.loc["AAA", "Tavan"] and not b.loc["AAA", "Taban"]
    assert b.loc["BBB", "Taban"] is np.True_ or bool(b.loc["BBB", "Taban"])
    assert round(b.loc["AAA", "Değişim %"], 4) == 10.0
    assert abs(b.loc["AAA", "Gün İçi Konum"] - 1.0) < 1e-9
    assert b.loc["CCC", "Durum"] == "STALE" and b.loc["AAA", "Durum"] == "CANLI"
    assert "RSI" not in b.columns  # no history -> no invented indicators
    s = summarize(b.reset_index())
    assert s["count"] == 3 and s["tavan"] == 1 and s["taban"] == 1 and s["stale"] == 1


def test_indicators_alerts_and_helpers():
    q, _ = fetch_quotes(["AAA", "BBB"], FakeTicker, NOW)
    bars = []
    for sym in ("AAA", "BBB"):
        h = FakeTicker(sym).history()
        h = h.reset_index(names="timestamp").rename(columns=str.lower)
        h["symbol"] = sym
        h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
        bars.append(h)
    board = build_board(q, pd.concat(bars), now=NOW)
    assert {"RSI", "Rel. Hacim", "Trend", "Direnç Kırılımı"} <= set(board.columns)
    texts = [a["text"] for a in alerts(board)]
    assert any("AAA TAVANDA" in t for t in texts) and any("BBB TABANDA" in t for t in texts)
    assert movers(board, "Değişim %").iloc[0]["Sembol"] == "AAA"
    assert change_histogram(board)["Adet"].sum() == 2
    h = change_histogram(board)
    assert list(h["Sıra"]) == sorted(h["Sıra"]) and h.iloc[0]["Aralık"].startswith("≤")
    assert alerts(pd.DataFrame()) == [] and summarize(pd.DataFrame())["status"] == "MISSING"


def test_snapshot_blocked_without_source_and_ok_with_injected_source():
    blocked = load_live_snapshot({}, now=NOW)
    assert blocked.status == "BLOCKED" and blocked.board is None
    snap = load_live_snapshot({"BIST_DATA_BACKEND": "borsapy"}, now=NOW, symbols_fn=lambda: ["AAA", "BBB", "CCC"],
                              ticker_factory=FakeTicker)
    assert snap.status == "OK" and len(snap.board) == 3 and snap.bars is not None
    assert "gecikmeli" in snap.note and snap.summary["tavan"] == 1
    none = load_live_snapshot({"BIST_DATA_BACKEND": "borsapy"}, now=NOW, symbols_fn=lambda: ["BAD"],
                              ticker_factory=FakeTicker)
    assert none.status == "BLOCKED"
