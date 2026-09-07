"""Signal attribution: persist why a rank happened and what happened next."""
from __future__ import annotations

from dataclasses import dataclass

from .research_memory import ResearchMemory, SignalAttribution


@dataclass(frozen=True, slots=True)
class AttributionInput:
    signal_id: str
    symbol: str
    timestamp: str
    score: float
    feature_scores: dict[str, float]
    source_ids: tuple[str, ...] = ()


def attribute_and_store(inp: AttributionInput, memory: ResearchMemory) -> SignalAttribution:
    total = sum(max(0.0, value) for value in inp.feature_scores.values())
    contributions = {
        key: round(value / total * inp.score, 6) if total else 0.0
        for key, value in inp.feature_scores.items()
    }
    attribution = SignalAttribution(
        inp.signal_id,
        inp.symbol,
        inp.timestamp,
        inp.score,
        None,
        None,
        contributions,
        inp.source_ids,
    )
    memory.record_signal(attribution)
    return attribution


def close_signal(memory: ResearchMemory, signal_id: str, outcome_return: float, event_threshold: float = 0.10) -> bool:
    """Append an outcome event; returns whether the signal hit the target."""
    hit = outcome_return >= event_threshold
    memory._append({
        "type": "signal_outcome",
        "signal_id": signal_id,
        "outcome_return": outcome_return,
        "hit": hit,
    })
    return hit
