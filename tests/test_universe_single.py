from datetime import date

import pandas as pd
import pytest

from bist_hunter.adapters import ProviderError
from bist_hunter.daily_ranker import RankingConfig, rank_latest
from bist_hunter.universe import (
    build_bist100_plus_universe, filter_point_in_time, load_bist100_plus_universe,
    universe_history_from_records,
)

HISTORY = universe_history_from_records(
    [
        {"symbol": "AAA", "valid_from": "2020-01-01"},
        {"symbol": "DEAD.IS", "valid_from": "2020-01-01", "valid_to": "2021-06-30"},
        {"symbol": "NEW", "listed_at": "2023-01-02"},
    ],
    source="test",
)


def test_history_is_point_in_time_and_keeps_delisted():
    assert HISTORY.symbols_on("2021-01-01") == ("AAA", "DEAD")
    assert HISTORY.symbols_on(date(2022, 1, 1)) == ("AAA",)
    assert HISTORY.symbols_on("2024-01-01") == ("AAA", "NEW")
    assert HISTORY.all_symbols() == ("AAA", "DEAD", "NEW")


def test_filter_point_in_time_avoids_survivorship_and_lookahead():
    frame = pd.DataFrame({
        "symbol": ["AAA", "DEAD", "DEAD", "NEW", "NEW"],
        "timestamp": ["2021-01-04", "2021-01-04", "2022-01-03", "2022-06-01", "2023-06-01"],
    })
    kept = filter_point_in_time(frame, HISTORY)
    assert list(zip(kept.symbol, kept.timestamp)) == [
        ("AAA", "2021-01-04"), ("DEAD", "2021-01-04"), ("NEW", "2023-06-01")]


def test_invalid_history_rejected():
    with pytest.raises(ValueError):
        universe_history_from_records([{"symbol": "X"}], source="t")
    with pytest.raises(ValueError):
        universe_history_from_records([{"symbol": "X", "valid_from": "2022-01-01", "valid_to": "2021-01-01"}], source="t")


def test_ranker_only_ranks_universe_members():
    rows = []
    for symbol in ("AAA", "zzz.is", "OUT"):
        for day in range(25):
            rows.append({"symbol": symbol, "timestamp": f"2026-09-{day + 1:02d}", "open": 10, "high": 10.1,
                         "low": 9.9, "close": 10 * (1 + day * 0.003), "volume": 1000 + day})
    universe = build_bist100_plus_universe(["AAA"], ["ZZZ"], as_of="2026-09-25", source="t")
    ranked = rank_latest(pd.DataFrame(rows), RankingConfig(min_score=0), universe=universe)
    assert set(ranked["symbol"]) == {"AAA", "ZZZ"}


def test_single_universe_blocks_without_configuration(monkeypatch):
    for name in ("BIST_UNIVERSE_URL", "BIST_SYMBOLS", "BIST100_MEMBERSHIP_URL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ProviderError):
        load_bist100_plus_universe()


def test_single_universe_from_env(monkeypatch):
    monkeypatch.delenv("BIST_UNIVERSE_URL", raising=False)
    monkeypatch.delenv("BIST100_MEMBERSHIP_URL", raising=False)
    monkeypatch.setenv("BIST_SYMBOLS", "thyao, AKBNK.IS,thyao")
    universe = load_bist100_plus_universe("2026-09-30")
    assert universe.symbols == ("AKBNK", "THYAO")
    assert universe.as_of == "2026-09-30" and universe.source == "BIST_SYMBOLS"
