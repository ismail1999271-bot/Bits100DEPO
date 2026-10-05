"""Deterministic multi-layer ranking score.

LLM outputs are deliberately absent: AI agents may propose hypotheses, while
this numerical layer remains the decision arbiter for ranking and validation.

Missing components are never filled with zero. The weights of the components
that are actually present are renormalized, and the share of nominal weight
that is present is reported as ``coverage``. ``confidence`` additionally
discounts each present component by an optional point-in-time data-quality
factor (0..1, e.g. freshness or provider completeness).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class QuantScore:
    score: float
    confidence: float
    status: str
    reasons: tuple[str, ...]
    coverage: float = 0.0
    components: tuple[tuple[str, float], ...] = ()
    missing: tuple[str, ...] = ()

    def component(self, name: str) -> float | None:
        return dict(self.components).get(name)


WEIGHTS = {
    "auction": 0.25,
    "tavan_dna": 0.20,
    "technical": 0.10,
    "kap": 0.10,
    "news": 0.05,
    "fund": 0.10,
    "institutional": 0.08,
    "broker": 0.07,
    "smart_money": 0.05,
}
COMPONENTS = tuple(WEIGHTS)


def _clean(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not isfinite(value):
        return None
    if value < 0.0 or value > 100.0:
        raise ValueError("component scores must be between 0 and 100")
    return value


def _quality(name: str, quality: dict[str, float] | None) -> float:
    if quality is None or name not in quality:
        return 1.0
    q = float(quality[name])
    if not 0.0 <= q <= 1.0:
        raise ValueError("component quality must be between 0 and 1")
    return q


def calculate_quant_score(
    components: dict[str, float | None],
    *,
    min_score: float = 70.0,
    min_coverage: float = 0.40,
    quality: dict[str, float] | None = None,
) -> QuantScore:
    """Combine available point-in-time components with renormalized weights.

    Missing components reduce coverage/confidence but never silently become
    zero. The default coverage floor permits the two strongest early-session
    layers (auction + Tavan-DNA = 45% of nominal weight) to produce a research
    signal, while callers can require stricter coverage for production.
    """
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be in (0, 1]")
    unknown = set(components) - set(WEIGHTS)
    if unknown:
        raise ValueError(f"unknown quant components: {sorted(unknown)}")
    clean = {name: _clean(components.get(name)) for name in WEIGHTS}
    available = {name: value for name, value in clean.items() if value is not None}
    missing = tuple(name for name in WEIGHTS if name not in available)
    coverage = round(sum(WEIGHTS[name] for name in available), 4)
    confidence = round(sum(WEIGHTS[name] * _quality(name, quality) for name in available), 4)
    used = tuple((name, round(value, 4)) for name, value in available.items())
    if coverage < min_coverage or not available:
        return QuantScore(
            0.0, confidence, "BLOCKED_INSUFFICIENT_DATA",
            ("insufficient component coverage",), coverage, used, missing,
        )

    score = sum(WEIGHTS[name] * value for name, value in available.items()) / coverage
    reasons: list[str] = []
    for name, value in sorted(available.items(), key=lambda item: item[1], reverse=True):
        if value >= 75:
            reasons.append(f"strong_{name}")
    if score < min_score:
        reasons.append("below_quality_threshold")
    status = "SIGNAL" if score >= min_score else "NO_QUALITY_SIGNAL"
    return QuantScore(round(score, 2), confidence, status, tuple(reasons), coverage, used, missing)
