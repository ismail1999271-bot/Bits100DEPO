"""Outcome-driven performance learning without automatic live-weight changes."""
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
    feature_edge: dict[str, float]
    recommended_action: str


def build_learning_report(memory: ResearchMemory, hit_threshold: float = 0.10) -> LearningReport:
    """Join immutable signal attributions with later outcomes and summarize OOS evidence."""
    rows = memory.load()
    signals = {str(r.get("signal_id")): r for r in rows if r.get("type") == "signal"}
    outcomes = [r for r in rows if r.get("type") == "signal_outcome" and r.get("outcome_return") is not None]
    returns = [float(r["outcome_return"]) for r in outcomes]
    if not returns:
        return LearningReport(0, 0.0, 0.0, 0.0, 0.0, {}, "COLLECT_MORE_OUTCOMES")

    ordered = sorted(returns)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    mean = sum(returns) / len(returns)
    variance = sum((x - mean) ** 2 for x in returns) / max(1, len(returns) - 1)
    hit_rate = sum(x >= hit_threshold for x in returns) / len(returns)

    # Feature edge is the return-weighted contribution sign. It is diagnostic,
    # not a request to mutate production weights.
    feature_values: dict[str, list[float]] = {}
    for outcome in outcomes:
        signal = signals.get(str(outcome.get("signal_id")))
        if not signal:
            continue
        outcome_return = float(outcome["outcome_return"])
        for name, contribution in dict(signal.get("feature_contributions", {})).items():
            feature_values.setdefault(str(name), []).append(float(contribution) * outcome_return)
    feature_edge = {k: round(sum(v) / len(v), 8) for k, v in feature_values.items() if v}

    action = "READY_FOR_RESEARCH" if len(returns) >= 30 else "COLLECT_MORE_OUTCOMES"
    return LearningReport(
        len(returns), round(hit_rate, 4), round(mean, 6), round(median, 6),
        round(sqrt(variance), 6), feature_edge, action,
    )


def research_actions(report: LearningReport) -> tuple[str, ...]:
    """Produce controlled experiment proposals; never alter live weights automatically."""
    if report.recommended_action != "READY_FOR_RESEARCH":
        return ("COLLECT_MORE_OUTCOMES",)
    if report.hit_rate < 0.50 or report.mean_return <= 0:
        return ("AUDIT_SIGNAL_QUALITY", "RUN_FEATURE_ABLATION", "RUN_WALK_FORWARD_REWEIGHTING_EXPERIMENT")
    return ("RUN_FEATURE_ABLATION", "RUN_WALK_FORWARD_REWEIGHTING_EXPERIMENT")
