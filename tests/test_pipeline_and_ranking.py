from datetime import datetime, timezone
import pandas as pd

from bist_hunter.pipeline import normalize_ohlcv, time_split
from bist_hunter.ranking import rank_candidates
from bist_hunter.features import FeatureSnapshot


def test_normalize_and_split():
    rows = [{"symbol":"abc","timestamp":"2026-01-02T10:00:00Z","open":100,"high":101,"low":99,"close":100,"volume":1000},
            {"symbol":"abc","timestamp":"2026-01-03T10:00:00Z","open":100,"high":105,"low":100,"close":104,"volume":2000}]
    x = normalize_ohlcv(rows)
    assert x.symbol.tolist() == ["ABC", "ABC"]
    split = time_split(x, datetime(2026, 1, 3, tzinfo=timezone.utc))
    assert len(split.train) == 1 and len(split.test) == 1


def test_no_quality_signal_is_allowed():
    f = FeatureSnapshot(0, 0, 1, .01, .5, 0, 0, .9, .02)
    result = rank_candidates([("ABC", f)], min_score=99)
    assert result.status == "NO_QUALITY_SIGNAL"
    assert result.candidates == ()
