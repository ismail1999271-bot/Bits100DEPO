from datetime import date, datetime

from bist_hunter.backtest.engine import CostModel, simulate_trade
from bist_hunter.data.schema import NormalizedNews
from bist_hunter.news.normalize import deduplicate


def test_news_deduplication():
    a = NormalizedNews(datetime(2026, 1, 1, 10), "KAP", "ABC yeni yatırım", "https://x/1", ("ABC",))
    b = NormalizedNews(datetime(2026, 1, 1, 10), "KAP", "ABC yeni yatırım", "https://x/2", ("ABC",))
    assert len(deduplicate([a, b])) == 1


def test_backtest_costs_reduce_return():
    result = simulate_trade("ABC", 100, 110, 10_000, CostModel(10, 5))
    assert result.gross_return == 0.10
    assert result.net_return < result.gross_return
