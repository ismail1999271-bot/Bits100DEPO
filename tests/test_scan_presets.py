import numpy as np
import pandas as pd

from bist_hunter.scan_presets import SCREENS, matches, run_screens
from bist_hunter.technical import add_technical_features
from bist_hunter.telegram_notify import SignalChangeTracker, change_report


def _bars(symbol, closes, vols):
    ts = pd.date_range("2025-01-01", periods=len(closes), freq="D", tz="UTC")
    c = np.array(closes, float)
    return pd.DataFrame({"symbol": symbol, "timestamp": ts, "open": c, "high": c * 1.01, "low": c * 0.99,
                         "close": c, "volume": vols})


def test_screens_flag_spike_and_do_not_impute():
    flat = _bars("AAA", [10.0] * 80, [1000.0] * 80)
    spike = _bars("BBB", [10.0] * 79 + [10.5], [1000.0] * 79 + [5000.0])
    short = _bars("CCC", [10.0] * 5, [1000.0] * 5)
    tech = add_technical_features(pd.concat([flat, spike, short], ignore_index=True))
    table = run_screens(tech)
    assert table.loc["BBB", "volume_spike"] and not table.loc["AAA", "volume_spike"]
    assert not table.loc["CCC"].any()  # insufficient history -> no match, not imputed
    assert set(table.columns) == set(SCREENS)
    assert "BBB" in matches(tech, ["volume_spike"]).index


def test_signal_change_tracker_dedupes_and_survives_corrupt_state(tmp_path):
    path = tmp_path / "s.json"
    t = SignalChangeTracker(str(path))
    r1 = pd.DataFrame({"Symbol": ["A", "B"], "Status": ["SIGNAL", "BLOCKED"]})
    assert len(t.changes(r1)) == 2
    t.save()
    t2 = SignalChangeTracker(str(path))
    assert t2.changes(r1) == []
    r2 = pd.DataFrame({"Symbol": ["A", "B"], "Status": ["WATCH", "BLOCKED"]})
    ch = t2.changes(r2)
    assert ch == [("A", "SIGNAL", "WATCH")] and change_report(ch) is None
    path.write_text("{not json")
    assert SignalChangeTracker(str(path)).state == {}
