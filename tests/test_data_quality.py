import pandas as pd

from bist_hunter.data.quality import validate_ohlcv


def test_ohlcv_quality_accepts_valid_multi_symbol_data():
    frame = pd.DataFrame([
        {"symbol": "AAA", "timestamp": "2026-01-01", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100},
        {"symbol": "BBB", "timestamp": "2026-01-01", "open": 20, "high": 21, "low": 19, "close": 20, "volume": 200},
        {"symbol": "AAA", "timestamp": "2026-01-02", "open": 10, "high": 12, "low": 9, "close": 11, "volume": 120},
    ])
    assert validate_ohlcv(frame) == []


def test_ohlcv_quality_rejects_impossible_bar():
    frame = pd.DataFrame([{"symbol": "AAA", "timestamp": "2026-01-01", "open": 10, "high": 9, "low": 8, "close": 10, "volume": 100}])
    assert "high_below_ohlc" in validate_ohlcv(frame)
