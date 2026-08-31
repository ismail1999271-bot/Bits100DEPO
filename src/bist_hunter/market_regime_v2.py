"""Market-wide fear/greed regime score.

This is a composite research indicator, not a claim that BIST has an official
Fear & Greed index. Inputs are normalized to [-1, 1] before aggregation.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RegimeInputs:
    index_momentum: float = 0.0
    breadth: float = 0.0
    volume_pressure: float = 0.0
    volatility: float = 0.0
    global_risk: float = 0.0
    fx_pressure: float = 0.0
    sentiment: float = 0.0


def fear_greed_score(x: RegimeInputs) -> float:
    weights = {
        "index_momentum": 0.20,
        "breadth": 0.20,
        "volume_pressure": 0.15,
        "volatility": 0.10,
        "global_risk": 0.10,
        "fx_pressure": 0.10,
        "sentiment": 0.15,
    }
    values = vars(x)
    raw = sum(weights[k] * max(-1.0, min(1.0, values[k])) for k in weights)
    return round(50.0 + 50.0 * raw, 2)


def regime_label(score: float) -> str:
    if score < 20:
        return "EXTREME_FEAR"
    if score < 35:
        return "FEAR"
    if score < 50:
        return "CAUTIOUS"
    if score < 65:
        return "NEUTRAL"
    if score < 80:
        return "GREED"
    return "EXTREME_GREED"
