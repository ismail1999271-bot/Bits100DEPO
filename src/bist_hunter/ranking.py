"""Daily cross-sectional opportunity ranking.

The engine never forces a trade. If the best candidate does not clear the
configured quality threshold, it explicitly returns NO_QUALITY_SIGNAL.
"""

from dataclasses import dataclass
from math import tanh

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


def _smart_money_0_100(estimated_flow_try: float, scale_try: float) -> float:
    if scale_try <= 0:
        raise ValueError("smart_money_scale_try must be positive")
    return (tanh(estimated_flow_try / scale_try) + 1.0) * 50.0


def rank_candidates(
    items: list[tuple[str, FeatureSnapshot]],
    min_score: float = 70.0,
    top_k: int = 20,
    smart_money_flows: dict[str, float] | None = None,
    smart_money_weight: float = 0.20,
    smart_money_scale_try: float = 100_000_000,
) -> RankingResult:
    """Rank candidates, optionally blending institutional fund-flow exposure.

    ``smart_money_flows`` should contain *estimated* stock-level exposure
    derived from fund flows and the latest known fund holdings. It is never
    treated as a trade ledger. The default 20% weight leaves market structure
    as the dominant signal while allowing strong institutional flow to move a
    candidate materially up or down the list.
    """
    if not 0 <= smart_money_weight <= 1:
        raise ValueError("smart_money_weight must be between 0 and 1")
    if smart_money_scale_try <= 0:
        raise ValueError("smart_money_scale_try must be positive")

    scored: list[Candidate] = []
    flows = smart_money_flows or {}
    for symbol, features in items:
        base_score = score_snapshot(features)
        score = base_score
        reasons: list[str] = []
        if features.volume_ratio >= 2:
            reasons.append("volume anomaly")
        if features.breakout_20 >= 0.98:
            reasons.append("near 20-bar high")
        if features.return_5 > 0.03:
            reasons.append("positive 5-bar momentum")

        if symbol in flows:
            flow = flows[symbol]
            sm_score = _smart_money_0_100(flow, smart_money_scale_try)
            score = round(base_score * (1 - smart_money_weight) + sm_score * smart_money_weight, 2)
            if flow >= smart_money_scale_try:
                reasons.append("smart money accumulation")
            elif flow <= -smart_money_scale_try:
                reasons.append("smart money distribution")

        score = round(max(0.0, min(100.0, score)), 2)
        confidence = min(0.99, max(0.01, 0.35 + score / 150))
        scored.append(Candidate(symbol, score, confidence, tuple(reasons)))
    scored.sort(key=lambda x: x.score, reverse=True)
    selected = tuple(x for x in scored if x.score >= min_score)[:top_k]
    return RankingResult(selected, "SIGNAL" if selected else "NO_QUALITY_SIGNAL")
