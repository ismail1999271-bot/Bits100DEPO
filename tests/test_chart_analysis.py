import numpy as np
import pandas as pd

from bist_hunter.chart_analysis import chart_report, detect_patterns, pivots, render_chart, support_resistance


def _bars(closes, symbol="AAA", spread=0.01):
    c = np.array(closes, float)
    ts = pd.date_range("2025-01-01", periods=len(c), freq="D", tz="UTC")
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"symbol": symbol, "timestamp": ts, "open": o, "high": np.maximum(o, c) * (1 + spread),
                         "low": np.minimum(o, c) * (1 - spread), "close": c, "volume": 1000.0})


def _wave(n=120, seed=3):
    rng = np.random.default_rng(seed)
    return 100 + 8 * np.sin(np.arange(n) / 6) + np.cumsum(rng.normal(0, 0.3, n))


def test_levels_are_point_in_time():
    df = _bars(_wave())
    full = support_resistance(df)
    truncated = support_resistance(df.iloc[:-10])
    # confirmed pivots can only change by adding later bars, never by editing history:
    ph_full, _ = pivots(df)
    ph_part, _ = pivots(df.iloc[:-10])
    assert set(ph_part) <= set(ph_full)
    assert all(r > df["close"].iloc[-1] for r in full.resistances)
    assert all(s < df["close"].iloc[-1] for s in full.supports)
    assert truncated is not None


def test_last_confirm_bars_never_make_pivots():
    df = _bars(_wave())
    ph, pl = pivots(df, left=3, confirm=3)
    assert max(ph + pl) <= len(df) - 1 - 3


def test_patterns_limit_up_and_breakout():
    closes = [10.0] * 40 + [11.0]
    df = _bars(closes)
    flags = detect_patterns(df)
    assert "LIMIT_UP" in flags and "BREAKOUT_20" in flags
    assert detect_patterns(_bars([10.0] * 10)) == []


def test_render_and_report(tmp_path):
    df = _bars(_wave())
    out = render_chart(df, "AAA", tmp_path / "AAA.png")
    assert out is not None and out.stat().st_size > 5000
    assert render_chart(_bars([10.0] * 10), "AAA", tmp_path / "x.png") is None
    rep = chart_report(df, "AAA")
    assert rep["status"] == "OK" and "supports" in rep
    assert chart_report(_bars([10.0] * 10), "AAA")["status"] == "MISSING"
