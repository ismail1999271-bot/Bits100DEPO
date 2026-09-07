from datetime import datetime, timedelta

import pytest

from bist_hunter.domain import MarketBar
from bist_hunter.performance_board import build_performance_board, format_performance_board


def bars(symbol: str, closes: list[float]) -> list[MarketBar]:
    start = datetime(2026, 1, 1)
    return [
        MarketBar(symbol, start + timedelta(days=i), close, close, close, close, 1_000)
        for i, close in enumerate(closes)
    ]


def test_builds_month_style_leaderboard_and_rank_change() -> None:
    rows = build_performance_board(
        {
            "AAA": bars("AAA", [100, 110, 120, 130, 150]),
            "BBB": bars("BBB", [100, 90, 95, 100, 105]),
            "CCC": bars("CCC", [100, 100, 100, 100, 100]),
        },
        period_bars=2,
        top_k=3,
    )

    assert [row.symbol for row in rows] == ["AAA", "BBB", "CCC"]
    assert rows[0].period_return_pct == 25.0
    assert rows[0].previous_period_return_pct == 20.0
    assert rows[1].rank_change == 1
    assert rows[1].acceleration_pct == 15.0


def test_excludes_symbols_without_two_full_periods() -> None:
    rows = build_performance_board(
        {"AAA": bars("AAA", [100, 110, 120])},
        period_bars=2,
    )
    assert rows == ()


def test_validates_configuration() -> None:
    with pytest.raises(ValueError):
        build_performance_board({}, period_bars=0)
    with pytest.raises(ValueError):
        build_performance_board({}, top_k=0)


def test_formats_social_style_board() -> None:
    rows = build_performance_board(
        {
            "AAA": bars("AAA", [100, 110, 120, 130, 150]),
            "BBB": bars("BBB", [100, 90, 95, 100, 105]),
        },
        period_bars=2,
    )
    text = format_performance_board(rows, title="AĞUSTOS LİDERLERİ")
    assert "AĞUSTOS LİDERLERİ" in text
    assert "AAA" in text
    assert "%  25.00" in text
    assert "BBB" in text
