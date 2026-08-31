from datetime import datetime

from bist_hunter.domain import MarketBar
from bist_hunter.tavan_dna import LimitRule, first_touch_time, is_limit_up_close


def bar(close: float, high: float, minute: int = 0) -> MarketBar:
    return MarketBar(
        symbol="TEST",
        timestamp=datetime(2026, 1, 2, 10, minute),
        open=100.0,
        high=high,
        low=99.0,
        close=close,
        volume=1_000,
    )


def test_limit_up_close():
    rule = LimitRule(0.10)
    assert is_limit_up_close(bar(110.0, 110.0), 100.0, rule)
    assert not is_limit_up_close(bar(109.9, 109.9), 100.0, rule)


def test_first_touch_time():
    rule = LimitRule(0.10)
    bars = [bar(101.0, 105.0, 0), bar(106.0, 110.0, 1), bar(110.0, 110.0, 2)]
    assert first_touch_time(bars, 100.0, rule) == datetime(2026, 1, 2, 10, 1)
