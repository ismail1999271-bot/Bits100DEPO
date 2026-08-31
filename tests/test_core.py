from datetime import datetime, timedelta

from bist_hunter.domain import MarketBar
from bist_hunter.features import build_features
from bist_hunter.market_regime_v2 import RegimeInputs, fear_greed_score, regime_label
from bist_hunter.pipeline import CandidateEvidence, rank_candidates, score_candidate


def test_features_are_based_on_prior_bars():
    bars = []
    for i in range(22):
        bars.append(MarketBar("AAA", datetime(2026, 1, 1) + timedelta(days=i), 100+i, 102+i, 99+i, 101+i, 1000))
    bars[-1] = MarketBar("AAA", bars[-1].timestamp, 120, 130, 119, 129, 3000)
    f = build_features(bars)
    assert f.return_1 > 0
    assert f.volume_ratio > 1
    assert 0 <= f.close_location <= 1


def test_fear_greed_bounds_and_label():
    score = fear_greed_score(RegimeInputs(index_momentum=1, breadth=1, volume_pressure=1, volatility=1, global_risk=1, fx_pressure=1, sentiment=1))
    assert score == 100
    assert regime_label(score) == "EXTREME_GREED"


def test_candidate_ranking():
    strong = CandidateEvidence("STRONG", pattern=1, catalyst=1, volume_anomaly=1)
    weak = CandidateEvidence("WEAK")
    assert score_candidate(strong) > score_candidate(weak)
    assert rank_candidates([weak, strong])[0][0] == "STRONG"
