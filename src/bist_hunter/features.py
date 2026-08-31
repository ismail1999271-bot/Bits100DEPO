"""Leakage-safe feature engineering for pre-tavan and daily ranking research."""

from dataclasses import dataclass
from math import isfinite

from .domain import MarketBar


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    return_1: float
    return_5: float
    volume_ratio: float
    range_pct: float
    close_location: float
    return_20: float
    distance_ma20: float
    breakout_20: float
    volatility_20: float


def _safe_div(a: float, b: float) -> float:
    if b == 0 or not isfinite(b):
        return 0.0
    return a / b


def build_features(bars: list[MarketBar], volume_lookback: int = 20) -> FeatureSnapshot:
    if len(bars) < 2:
        raise ValueError("at least two bars are required")
    current = bars[-1]
    prev = bars[-2]
    return_1 = _safe_div(current.close, prev.close) - 1.0
    return_5 = _safe_div(current.close, bars[max(0, len(bars) - 6)].close) - 1.0
    anchor20 = bars[max(0, len(bars) - 21)].close
    return_20 = _safe_div(current.close, anchor20) - 1.0
    history = bars[-volume_lookback - 1 : -1]
    avg_volume = sum(x.volume for x in history) / len(history) if history else current.volume
    volume_ratio = _safe_div(current.volume, avg_volume)
    range_pct = _safe_div(current.high - current.low, current.close)
    close_location = _safe_div(current.close - current.low, current.high - current.low) if current.high != current.low else 0.5
    ma20 = sum(x.close for x in bars[-20:]) / min(20, len(bars))
    distance_ma20 = _safe_div(current.close, ma20) - 1.0
    high20 = max(x.high for x in bars[-20:])
    breakout_20 = _safe_div(current.close, high20)
    returns = [_safe_div(bars[i].close, bars[i - 1].close) - 1.0 for i in range(max(1, len(bars) - 20), len(bars))]
    mean = sum(returns) / len(returns)
    volatility_20 = (sum((x - mean) ** 2 for x in returns) / max(1, len(returns) - 1)) ** 0.5
    return FeatureSnapshot(return_1, return_5, volume_ratio, range_pct, close_location, return_20, distance_ma20, breakout_20, volatility_20)


def score_snapshot(f: FeatureSnapshot) -> float:
    """Interpretable baseline score from 0-100; not a claim of predictive power."""
    score = 50.0
    score += max(-10, min(10, f.return_5 * 100))
    score += max(-10, min(10, f.return_20 * 60))
    score += max(-12, min(18, (f.volume_ratio - 1) * 12))
    score += max(-8, min(8, f.distance_ma20 * 50))
    score += max(-8, min(8, (f.breakout_20 - 1) * 80))
    score += max(-5, min(5, (f.close_location - 0.5) * 10))
    return round(max(0.0, min(100.0, score)), 2)
