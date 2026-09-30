import pandas as pd
import pytest

from bist_hunter.portfolio_backtest import (
    NOT_EVIDENCE, ExecutionConfig, assert_point_in_time, run_portfolio_backtest,
)
from bist_hunter.synthetic import make_dataset
from bist_hunter.technical import add_technical_features, technical_score
from bist_hunter.universe import universe_history_from_records


def frame():
    f = pd.DataFrame(make_dataset(symbols=("AAA", "BBB", "CCC", "DDD"), days=120))
    f["volume"] = f["volume"] * 100  # enough ADV for the liquidity gate
    return f


def score_fn(f):
    feats = add_technical_features(f)
    feats.index = f.sort_values(["symbol", "timestamp"]).index
    return feats.apply(technical_score, axis=1).reindex(f.index).astype(float)


def test_backtest_segments_metrics_and_evidence_flag():
    f = frame()
    result = run_portfolio_backtest(f, score_fn(f), provenance="synthetic",
                                    config=ExecutionConfig(score_threshold=50, top_k=2))
    assert result.evidence == NOT_EVIDENCE
    names = [s.segment for s in result.segments]
    assert names == ["train", "validation", "holdout"]
    assert result.segments[0].period[1] < result.segments[1].period[0] < result.segments[2].period[0]
    filled = result.trades[result.trades["status"] == "FILLED"]
    assert not filled.empty
    assert (filled["net_return"] < filled["gross_return"]).all()  # costs always applied
    for seg in result.segments:
        for field in ("total_return", "win_rate", "expectancy", "max_drawdown", "trade_count", "turnover",
                      "costs_try", "slippage_try"):
            assert getattr(seg, field) is not None


def test_limit_up_open_and_illiquid_names_are_not_filled():
    f = frame()
    f["volume"] = 1.0  # tiny ADV
    result = run_portfolio_backtest(f, score_fn(f), provenance="synthetic",
                                    config=ExecutionConfig(score_threshold=0))
    assert set(result.trades["status"]) <= {"SKIP_LIQUIDITY", "SKIP_LIMIT_UP"}


def test_survivorship_filter_excludes_non_members():
    f = frame()
    history = universe_history_from_records([{"symbol": s, "valid_from": "2020-01-01"} for s in ("AAA", "BBB")],
                                            source="t")
    result = run_portfolio_backtest(f, score_fn(f), provenance="synthetic", universe_history=history,
                                    config=ExecutionConfig(score_threshold=0))
    assert set(result.trades["symbol"]) <= {"AAA", "BBB"}


def test_point_in_time_checker_detects_leak():
    f = frame()
    assert_point_in_time(score_fn, f)

    def leaky(g):
        nxt = g.sort_values(["symbol", "timestamp"]).groupby("symbol")["close"].shift(-1)
        return (nxt / g["close"]).reindex(g.index)

    with pytest.raises(AssertionError):
        assert_point_in_time(leaky, f)
