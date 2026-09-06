import pandas as pd
import pytest

from bist_hunter.adapters import ProviderError, parse_json_records
from bist_hunter.evaluation import evaluate_opportunities


def test_parse_provider_envelopes():
    assert parse_json_records([{"symbol": "AAA"}])[0]["symbol"] == "AAA"
    assert parse_json_records({"data": [{"symbol": "BBB"}]})[0]["symbol"] == "BBB"
    with pytest.raises(ProviderError):
        parse_json_records({"error": "no data"})


def test_opportunity_metrics_top_k_and_drawdown():
    frame = pd.DataFrame(
        [
            {"timestamp": "2026-01-01", "symbol": "A", "score": 0.9, "future_return": 0.12},
            {"timestamp": "2026-01-01", "symbol": "B", "score": 0.8, "future_return": 0.01},
            {"timestamp": "2026-01-02", "symbol": "C", "score": 0.95, "future_return": -0.05},
            {"timestamp": "2026-01-02", "symbol": "D", "score": 0.7, "future_return": 0.15},
        ]
    )
    metrics = evaluate_opportunities(frame, top_k=1)
    assert metrics.top_k_precision == 0.5
    assert metrics.hit_day_rate == 1.0
    assert metrics.recall == 0.5
    assert metrics.max_drawdown < 0
