"""Deterministic multi-layer ranking score.

LLM outputs are deliberately absent: AI agents may propose hypotheses, while
this numerical layer remains the decision arbiter for ranking and validation.
"""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QuantScore:
    score: float
    confidence: float
    status: str
    reasons: tuple[str, ...]


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


def _clean(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if value < 0.0 or value > 100.0:
        raise ValueError("component scores must be between 0 and 100")
    return value


def calculate_quant_score(
    components: dict[str, float | None],
    *,
    min_score: float = 70.0,
    min_coverage: float = 0.40,
) -> QuantScore:
    """Combine available point-in-time components with renormalized weights.

    Missing components reduce confidence but do not silently become zero.
    The default coverage floor permits the two strongest early-session layers
    (auction + Tavan-DNA = 45% of nominal weight) to produce a research signal,
    while callers can require stricter coverage for production decisions.
    """
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be in (0, 1]")
    clean = {name: _clean(components.get(name)) for name in WEIGHTS}
    available = {name: value for name, value in clean.items() if value is not None}
    coverage = sum(WEIGHTS[name] for name in available)
    if coverage < min_coverage:
        return QuantScore(0.0, round(coverage, 4), "BLOCKED_INSUFFICIENT_DATA", ("insufficient component coverage",))

    score = sum(WEIGHTS[name] * value for name, value in available.items()) / coverage
    confidence = coverage
    reasons: list[str] = []
    for name, value in sorted(available.items(), key=lambda item: item[1], reverse=True):
        if value >= 75:
            reasons.append(f"strong_{name}")
    if score < min_score:
        reasons.append("below_quality_threshold")
    return QuantScore(round(score, 2), round(confidence, 4), "SIGNAL" if score >= min_score else "NO_QUALITY_SIGNAL", tuple(reasons))
