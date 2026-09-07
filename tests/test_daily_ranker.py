import pandas as pd

from bist_hunter.daily_ranker import RankingConfig, rank_latest


def bars():
    rows = []
    for symbol, close in [("ISMEN", 33.22), ("AAA", 40.0)]:
        for day in range(25):
            rows.append({
                "symbol": symbol,
                "timestamp": f"2026-09-{day + 1:02d}",
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close * (1 + day * 0.002),
                "volume": 1000 + day * 10,
            })
    frame = pd.DataFrame(rows)
    frame.loc[frame["symbol"] == "ISMEN", ["last_price", "entry_low", "entry_high", "target", "stop"]] = [33.22, 32.62, 33.50, 36.40, 32.00]
    return frame


def test_ranker_uses_technical_gate_and_score():
    result = rank_latest(bars(), RankingConfig(min_score=0))
    row = result[result["symbol"] == "ISMEN"].iloc[0]
    assert row["technical_risk_gate"] is True or bool(row["technical_risk_gate"])
    assert row["risk_reward"] > 1
    assert pd.notna(row["technical_watch_score"])


def test_partial_watchlist_does_not_break_uncovered_symbols():
    result = rank_latest(bars(), RankingConfig(min_score=0))
    assert "AAA" in set(result["symbol"])
