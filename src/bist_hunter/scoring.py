"""Explainable baseline scoring engine.

This is a research baseline, not a claim of predictive edge. We will replace
weights with walk-forward learned weights after the first historical dataset.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandidateFeatures:
    volume_anomaly: float = 0.0
    price_momentum: float = 0.0
    flow_strength: float = 0.0
    catalyst_strength: float = 0.0
    sector_strength: float = 0.0
    orderbook_strength: float = 0.0
    social_attention: float = 0.0
    manipulation_risk: float = 0.0

    def values(self) -> dict[str, float]:
        return {
            "volume_anomaly": self.volume_anomaly,
            "price_momentum": self.price_momentum,
            "flow_strength": self.flow_strength,
            "catalyst_strength": self.catalyst_strength,
            "sector_strength": self.sector_strength,
            "orderbook_strength": self.orderbook_strength,
            "social_attention": self.social_attention,
            "manipulation_risk": self.manipulation_risk,
        }


DEFAULT_WEIGHTS = {
    "volume_anomaly": 0.15,
    "price_momentum": 0.15,
    "flow_strength": 0.15,
    "catalyst_strength": 0.15,
    "sector_strength": 0.10,
    "orderbook_strength": 0.10,
    "social_attention": 0.10,
    "manipulation_risk": -0.10,
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_candidate(features: CandidateFeatures, weights: dict[str, float] | None = None) -> tuple[float, tuple[str, ...]]:
    """Return a 0-100 score and human-readable positive drivers."""
    weights = weights or DEFAULT_WEIGHTS
    values = features.values()
    score = 50.0 + 50.0 * sum(weights[k] * (_clamp(v) - 0.5) for k, v in values.items())
    score = max(0.0, min(100.0, score))
    reasons = tuple(
        name
        for name, value in values.items()
        if value >= 0.70 and weights.get(name, 0) > 0
    )
    return round(score, 2), reasons
