"""Learn from closed signals without silently changing production weights."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .research_memory import ResearchMemory


@dataclass(frozen=True, slots=True)
class LearningReport:
    closed_signals: int
    hit_rate: float
    mean_return: float
    median_return: float
    return_std: float
    recommended_action: str


def build_learning_report(memory: ResearchMemory, hit_threshold: float = 0.10) -> LearningReport:
    rows = memory.load()
    outcomes = [r for r in rows if r.get("type") == "signal_outcome"]
    returns = [float(r["outcome_return"]) for r in outcomes]
    if not returns:
        return LearningReport(0, 0.0, 0.0, 0.0, 0.0, "COLLECT_MORE_OUTCOMES")
    ordered = sorted(returns)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    mean = sum(returns) / len(returns)
    variance = sum((x - mean) ** 2 for x in returns) / max(1, len(returns) - 1)
    hit_rate = sum(x >= hit_threshold for x in returns) / len(returns)
    action = "KEEP" if len(returns) >= 30 and hit_rate >= 0.50 and mean > 0 else "COLLECT_MORE_OUTCOMES"
    return LearningReport(len(returns), round(hit_rate, 4), round(mean, 6), round(median, 6), round(sqrt(variance), 6), action)
