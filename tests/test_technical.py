import numpy as np
import pandas as pd
import pytest

from bist_hunter.synthetic import make_dataset
from bist_hunter.technical import (
    FEATURE_COLUMNS, add_technical_features, feature_usefulness, multi_timeframe_features,
    resample_bars, technical_score,
)


def intraday(days=6):
    rng = np.random.default_rng(3)
    rows = []
    for symbol in ("AAA", "BBB", "CCC"):
        price = 100.0
        for d in range(days):
            start = pd.Timestamp("2026-09-07 10:00") + pd.Timedelta(days=d)
            for m in range(0, 8 * 60, 5):
                price *= 1 + rng.normal(0, 0.002)
                rows.append({"symbol": symbol, "timestamp": start + pd.Timedelta(minutes=m), "open": price,
                             "high": price * 1.001, "low": price * 0.999, "close": price,
                             "volume": float(rng.integers(100, 1000))})
    return pd.DataFrame(rows)


def test_daily_features_exist_and_are_point_in_time():
    frame = pd.DataFrame(make_dataset(days=80))
    feats = add_technical_features(frame)
    assert set(FEATURE_COLUMNS) <= set(feats.columns)
    cut = feats["timestamp"].sort_values().unique()[60]
    truncated = add_technical_features(frame[pd.to_datetime(frame["timestamp"]) <= cut])
    a = feats[feats["timestamp"] == cut].set_index("symbol")[list(FEATURE_COLUMNS)]
    b = truncated[truncated["timestamp"] == cut].set_index("symbol")[list(FEATURE_COLUMNS)]
    pd.testing.assert_frame_equal(a, b)  # future rows never change past features
    last = feats.dropna(subset=["rsi_14"]).iloc[-1]
    assert 0 <= last["rsi_14"] <= 100


def test_resample_drops_incomplete_bucket():
    bars = intraday(days=1)
    as_of = pd.Timestamp("2026-09-07 11:30")
    hourly = resample_bars(bars, "1H", as_of=as_of)
    assert hourly["timestamp"].max() == pd.Timestamp("2026-09-07 10:00")  # 11:00 bucket incomplete
    with pytest.raises(ValueError):
        resample_bars(bars, "2H")


def test_multi_timeframe_supports_required_frames():
    out = multi_timeframe_features(intraday(), pd.Timestamp("2026-09-12 18:00"))
    assert set(out["timeframe"]) == {"5M", "15M", "1H", "4H", "1D"}
    assert (out["timestamp"] <= pd.Timestamp("2026-09-12 18:00")).all()


def test_technical_score_needs_enough_features():
    assert technical_score({"rsi_14": 60}) is None
    score = technical_score({"rsi_14": 62, "macd_hist": 1, "atr_14": 2, "relative_volume": 2.0,
                             "breakout_strength": 1.0, "momentum_10": 0.05})
    assert 60 < score <= 100


def test_feature_usefulness_ic():
    frame = pd.DataFrame(make_dataset(symbols=("A", "B", "C", "D", "E"), days=60))
    feats = add_technical_features(frame)
    feats["oracle"] = feats.groupby("symbol")["close"].shift(-1) / feats["close"]  # leak on purpose
    good = feature_usefulness(feats, "oracle")
    assert good.mean_ic > 0.9
    noise = feature_usefulness(feats, "rsi_14")
    assert noise.observations > 0 and abs(noise.mean_ic) < 0.5
