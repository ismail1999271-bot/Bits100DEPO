"""Daily cross-sectional opportunity ranking.

The engine never forces a trade. If the best candidate does not clear the
configured quality threshold, it explicitly returns NO_QUALITY_SIGNAL.
"""

from dataclasses import dataclass

from .features import FeatureSnapshot, score_snapshot


@dataclass(frozen=True, slots=True)
class Candidate:
    symbol: str
    score: float
    confidence: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RankingResult:
    candidates: tuple[Candidate, ...]
    status: str


def rank_candidates(items: list[tuple[str, FeatureSnapshot]], min_score: float = 70.0, top_k: int = 20) -> RankingResult:
    scored: list[Candidate] = []
    for symbol, features in items:
        score = score_snapshot(features)
        confidence = min(0.99, max(0.01, 0.35 + score / 150))
        reasons: list[str] = []
        if features.volume_ratio >= 2:
            reasons.append("volume anomaly")
        if features.breakout_20 >= 0.98:
            reasons.append("near 20-bar high")
        if features.return_5 > 0.03:
            reasons.append("positive 5-bar momentum")
        scored.append(Candidate(symbol, score, confidence, tuple(reasons)))
    scored.sort(key=lambda x: x.score, reverse=True)
    selected = tuple(x for x in scored if x.score >= min_score)[:top_k]
    return RankingResult(selected, "SIGNAL" if selected else "NO_QUALITY_SIGNAL")
