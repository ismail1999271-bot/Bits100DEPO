from datetime import datetime, timezone

import pandas as pd

from bist_hunter.event_features import NormalizedEvent
from bist_hunter.event_study import event_study, priced_in_ratio, rank_events_by_impact, type_impact_table


def bars(days=40):
    ts = pd.date_range("2026-01-05", periods=days, freq="B", tz="UTC") + pd.Timedelta(hours=7)
    close = [100 * (1 + 0.01 * i) for i in range(days)]
    return pd.DataFrame({"symbol": "AAA", "timestamp": ts, "open": [c * 0.999 for c in close],
                         "high": close, "low": close, "close": close, "volume": 1000})


def event(i, day, etype="IS_ILISKISI"):
    when = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc) + pd.Timedelta(days=day)
    return NormalizedEvent("AAA", str(i), when, "KAP", etype, "h")


def test_entry_is_first_bar_after_publication():
    b = bars()
    study = event_study([event(1, 7)], b)  # published Jan 12 12:00 UTC; bars stamped 07:00 UTC
    row = study.iloc[0]
    assert row["entry_time"] > row["published_at"]
    entry_bar = b[b["timestamp"] == row["entry_time"]].iloc[0]
    assert row["ret_1"] == entry_bar["close"] / entry_bar["open"] - 1
    assert row["runup"] is not None and row["runup"] > 0


def test_as_of_excludes_unrealized_outcomes_and_future_events():
    b = bars()
    ev = [event(1, 7)]
    as_of = pd.Timestamp("2026-01-14 23:00", tz="UTC")
    study = event_study(ev, b, as_of=as_of)
    assert pd.notna(study.loc[0, "ret_1"]) and pd.isna(study.loc[0, "ret_20"])
    assert event_study([event(1, 20)], b, as_of=as_of).empty


def test_type_table_sample_gate_and_ranking():
    b = bars(80)
    events = [event(i, i, "IS_ILISKISI") for i in range(1, 30)] + [event(100, 10, "DAVA")]
    study = event_study(events, b)
    table = type_impact_table(study, min_events=20)
    status = dict(zip(table["event_type"], table["status"]))
    assert status["IS_ILISKISI"] == "OK" and status["DAVA"] == "INSUFFICIENT_SAMPLE"
    ranked = rank_events_by_impact([event(200, 40, "DAVA"), event(201, 40, "IS_ILISKISI")], table)
    assert ranked[0]["event_type"] == "IS_ILISKISI" and ranked[0]["historical_mean"] > 0
    assert ranked[1]["historical_mean"] is None


def test_priced_in():
    assert priced_in_ratio(0.12, 0.01) == "LIKELY_PRICED_IN"
    assert priced_in_ratio(0.0, 0.05) == "MARKET_SURPRISED"
    assert priced_in_ratio(None, 0.05) == "UNKNOWN"
