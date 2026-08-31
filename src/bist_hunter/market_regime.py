"""Market regime and Fear/Greed baseline.

Inputs are normalized to 0..1. Production adapters will populate these values
from authoritative market data; no external data source is assumed here.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RegimeFeatures:
    momentum: float = 0.5
    breadth: float = 0.5
    volume_risk_appetite: float = 0.5
    volatility_inverse: float = 0.5
    global_risk_appetite: float = 0.5
    flow: float = 0.5


def fear_greed_score(features: RegimeFeatures) -> float:
    """Return a 0..100 composite market sentiment score."""
    vals = (
        features.momentum,
        features.breadth,
        features.volume_risk_appetite,
        features.volatility_inverse,
        features.global_risk_appetite,
        features.flow,
    )
    if any(not 0 <= value <= 1 for value in vals):
        raise ValueError("all regime features must be between 0 and 1")
    return round(sum(vals) / len(vals) * 100, 2)


def regime_label(score: float) -> str:
    if not 0 <= score <= 100:
        raise ValueError("score must be between 0 and 100")
    if score < 20:
        return "EXTREME_FEAR"
    if score < 35:
        return "FEAR"
    if score < 50:
        return "NEUTRAL"
    if score < 65:
        return "GREED"
    if score < 80:
        return "STRONG_GREED"
    return "EXTREME_GREED"
