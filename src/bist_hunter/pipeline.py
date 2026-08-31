"""Pure ranking pipeline that combines market and stock-level evidence."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    symbol: str
    volume_anomaly: float = 0.0
    momentum: float = 0.0
    catalyst: float = 0.0
    sector: float = 0.0
    orderbook: float = 0.0
    flow: float = 0.0
    social: float = 0.0
    pattern: float = 0.0
    manipulation_risk: float = 0.0


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_candidate(e: CandidateEvidence) -> float:
    weights = {
        "volume_anomaly": 0.15,
        "momentum": 0.15,
        "catalyst": 0.15,
        "sector": 0.10,
        "orderbook": 0.10,
        "flow": 0.10,
        "social": 0.05,
        "pattern": 0.20,
    }
    values = vars(e)
    score = sum(weights[k] * _clip(values[k]) for k in weights)
    score -= 0.10 * _clip(e.manipulation_risk)
    return round(100 * _clip(score), 2)


def rank_candidates(candidates: list[CandidateEvidence]) -> list[tuple[str, float]]:
    return sorted(((c.symbol, score_candidate(c)) for c in candidates), key=lambda x: x[1], reverse=True)
