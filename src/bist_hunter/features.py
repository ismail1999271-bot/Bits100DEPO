"""Leakage-safe feature engineering for pre-tavan research."""

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
    history = bars[-volume_lookback - 1 : -1]
    avg_volume = sum(x.volume for x in history) / len(history) if history else current.volume
    volume_ratio = _safe_div(current.volume, avg_volume)
    range_pct = _safe_div(current.high - current.low, current.close)
    close_location = _safe_div(current.close - current.low, current.high - current.low) if current.high != current.low else 0.5
    return FeatureSnapshot(return_1, return_5, volume_ratio, range_pct, close_location)
