from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from bist_hunter.historical_backtest import load_ohlcv_csv, run_historical_backtest


def _frame(rows: int = 120) -> pd.DataFrame:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    records = []
    for i in range(rows):
        close = 100 + (i % 17) * 0.4
        high = close * (1.12 if i % 9 == 0 else 1.02)
        records.append({
            "symbol": "TEST",
            "timestamp": start + timedelta(days=i),
            "open": close * 0.995,
            "high": high,
            "low": close * 0.98,
            "close": close,
            "volume": 1_000_000 + (i % 11) * 50_000,
        })
    return pd.DataFrame(records)


def test_historical_backtest_is_three_way_and_cost_aware():
    report = run_historical_backtest(_frame())
    assert report.train_rows > 0
    assert report.validation_rows > 0
    assert report.holdout_rows > 0
    assert report.train_rows + report.validation_rows + report.holdout_rows == report.labeled_rows
    assert report.holdout_returns.sample_size >= 0


def test_csv_loader_rejects_duplicate_observations(tmp_path):
    path = tmp_path / "bars.csv"
    _frame(10).to_csv(path, index=False)
    frame = pd.read_csv(path)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_ohlcv_csv(path)
