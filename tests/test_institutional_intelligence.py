from datetime import datetime, timezone

import pandas as pd
import pytest

from bist_hunter.institutional_intelligence import (
    InstitutionalWeights,
    analyst_revision_score,
    build_institutional_score,
    estimate_exposure,
    institutional_consensus,
    portfolio_risk_summary,
    research_signal_score,
    validate_point_in_time,
)


def test_point_in_time_rejects_future_information():
    frame = pd.DataFrame(
        [{"as_of": "2026-09-08", "observed_at": "2026-09-07"}]
    )
    with pytest.raises(ValueError, match="lookahead"):
        validate_point_in_time(frame)


def test_consensus_counts_manager_not_funds():
    frame = pd.DataFrame(
        [
            {"symbol": "AAA", "manager": "Ak", "weight": 0.08, "change": 0.01, "as_of": "2026-09-07"},
            {"symbol": "AAA", "manager": "Ak", "weight": 0.06, "change": 0.01, "as_of": "2026-09-07"},
            {"symbol": "AAA", "manager": "Is", "weight": 0.05, "change": 0.00, "as_of": "2026-09-07"},
        ]
    )
    result = institutional_consensus(frame)
    row = result.iloc[0]
    assert row["manager_count"] == 2
    assert row["fund_count"] == 3
    assert 0 <= row["consensus_score"] <= 100


def test_exposure_is_nan_without_aum():
    frame = pd.DataFrame([{"symbol": "AAA", "manager": "Ak", "weight": 0.10}])
    assert pd.isna(estimate_exposure(frame).iloc[0]["estimated_exposure_try"])


def test_exposure_uses_aum_times_weight():
    frame = pd.DataFrame([{"symbol": "AAA", "manager": "Ak", "weight": 0.10, "aum_try": 1_000_000}])
    assert estimate_exposure(frame).iloc[0]["estimated_exposure_try"] == 100_000


def test_analyst_revision_is_point_in_time_sequence():
    frame = pd.DataFrame(
        [
            {"symbol": "AAA", "period": "2027", "estimate": 10.0, "observed_at": "2026-09-01"},
            {"symbol": "AAA", "period": "2027", "estimate": 11.0, "observed_at": "2026-09-07"},
        ]
    )
    result = analyst_revision_score(frame)
    assert result.iloc[0]["analyst_revision_score"] > 50


def test_research_score_respects_confidence():
    frame = pd.DataFrame(
        [
            {"symbol": "AAA", "sentiment": 1.0, "confidence": 1.0, "observed_at": "2026-09-07"},
            {"symbol": "AAA", "sentiment": -1.0, "confidence": 0.0, "observed_at": "2026-09-07"},
        ]
    )
    assert research_signal_score(frame).iloc[0]["research_score"] > 50


def test_composite_score_and_coverage():
    frame = pd.DataFrame(
        [{
            "symbol": "AAA",
            "score": 80,
            "smart_money_score": 70,
            "consensus_score": 90,
            "research_score": 60,
            "fundamental_score": 75,
        }]
    )
    result = build_institutional_score(frame)
    assert 0 <= result.iloc[0]["institutional_score"] <= 100
    assert result.iloc[0]["institutional_data_coverage"] == 1


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        InstitutionalWeights(market=1.0, smart_money=0.0, consensus=0.0, research=0.0, fundamentals=0.1)


def test_portfolio_risk_summary_returns_effective_positions():
    frame = pd.DataFrame(
        [
            {"portfolio": "P1", "symbol": "AAA", "weight": 0.6},
            {"portfolio": "P1", "symbol": "BBB", "weight": 0.4},
        ]
    )
    result = portfolio_risk_summary(frame)
    assert result.iloc[0]["positions"] == 2
    assert result.iloc[0]["top1_weight"] == 0.6
    assert result.iloc[0]["effective_positions"] > 1
