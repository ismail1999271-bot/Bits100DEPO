"""Deterministic leaderboard aggregation for validated component returns."""
from dataclasses import dataclass
from .quant_validation import validate_returns, ValidationMetrics


@dataclass(frozen=True, slots=True)
class LeaderboardEntry:
    name: str
    metrics: ValidationMetrics


def rank_components(results: dict[str, list[float]], cost_bps: float = 0.0, slippage_bps: float = 0.0) -> list[LeaderboardEntry]:
    entries = [LeaderboardEntry(name, validate_returns(values, cost_bps, slippage_bps)) for name, values in results.items()]
    return sorted(entries, key=lambda e: (e.metrics.expectancy, e.metrics.profit_factor or -1.0), reverse=True)
