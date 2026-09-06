"""Unified opportunity score and quality gate.

All inputs are normalized to [-1, 1] (or [0, 1] for risk). The engine is
provider-neutral: live BIST/KAP/news/social adapters feed it normalized data.
"""
from dataclasses import dataclass

from .scoring import CandidateFeatures, score_candidate


@dataclass(frozen=True, slots=True)
class Opportunity:
    symbol: str
    score: float
    confidence: float
    quality: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SignalConfig:
    min_score: float = 70.0
    min_confidence: float = 0.55
    max_manipulation_risk: float = 0.65


def _positive(value: float) -> float:
    return max(0.0, min(1.0, (value + 1.0) / 2.0))


def build_opportunity(symbol: str, features: CandidateFeatures, config: SignalConfig = SignalConfig()) -> Opportunity:
    score, reasons = score_candidate(features)
    confidence = max(0.0, min(1.0, 0.55 + 0.35 * (score / 100.0) - 0.25 * max(0.0, features.manipulation_risk)))
    quality = (
        score >= config.min_score
        and confidence >= config.min_confidence
        and features.manipulation_risk <= config.max_manipulation_risk
    )
    return Opportunity(symbol, score, round(confidence, 4), quality, reasons)


def rank_opportunities(items: list[tuple[str, CandidateFeatures]], config: SignalConfig = SignalConfig()) -> list[Opportunity]:
    opportunities = [build_opportunity(symbol, features, config) for symbol, features in items]
    return sorted(opportunities, key=lambda x: (x.quality, x.score, x.confidence), reverse=True)


def tavan_probability_proxy(features: CandidateFeatures) -> float:
    """Research-only proxy in [0,1]; not a calibrated probability."""
    score, _ = score_candidate(features)
    return round(_positive(score / 100.0), 4)
