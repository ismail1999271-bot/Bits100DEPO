"""Provider-neutral daily ranking pipeline for normalized point-in-time signals."""
from dataclasses import dataclass

from .auction_trajectory import AuctionTrajectory, build_trajectory
from .quant_score import QuantScore, calculate_quant_score


@dataclass(frozen=True, slots=True)
class QuantCandidate:
    symbol: str
    score: QuantScore
    trajectory: AuctionTrajectory


def rank_quant_candidates(
    component_history: dict[str, list[dict[str, object]]],
    *,
    min_score: float = 70.0,
    min_coverage: float = 0.50,
) -> tuple[QuantCandidate, ...]:
    """Rank symbols using only supplied snapshots; blocked inputs never rank."""
    ranked: list[QuantCandidate] = []
    for symbol, rows in component_history.items():
        if not rows:
            continue
        latest = rows[-1]
        score = calculate_quant_score(latest.get("components", {}), min_score=min_score, min_coverage=min_coverage)
        points = latest.get("trajectory_points", [])
        trajectory = build_trajectory(symbol, points) if points else build_trajectory(symbol, [])
        if score.status == "SIGNAL" and trajectory.status != "BLOCKED":
            ranked.append(QuantCandidate(symbol, score, trajectory))
    ranked.sort(key=lambda item: (item.score.score, item.score.confidence), reverse=True)
    return tuple(ranked)
