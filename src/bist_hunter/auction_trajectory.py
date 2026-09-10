"""Point-in-time 09:40/09:45/09:50/09:55 auction trajectory."""
from dataclasses import dataclass


SNAPSHOT_MINUTES = (40, 45, 50, 55)


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    observed_at: str
    score: float
    indicative_change_pct: float


@dataclass(frozen=True, slots=True)
class AuctionTrajectory:
    symbol: str
    points: tuple[TrajectoryPoint, ...]
    slope: float | None
    acceleration: float | None
    status: str


def build_trajectory(symbol: str, points: list[TrajectoryPoint]) -> AuctionTrajectory:
    ordered = tuple(sorted(points, key=lambda p: p.observed_at))
    if not ordered:
        return AuctionTrajectory(symbol, (), None, None, "BLOCKED_MISSING_AUCTION_DATA")
    if len(ordered) == 1:
        return AuctionTrajectory(symbol, ordered, None, None, "PARTIAL")
    first, last = ordered[0], ordered[-1]
    slope = (last.score - first.score) / (len(ordered) - 1)
    acceleration = None
    if len(ordered) >= 3:
        early = ordered[1].score - ordered[0].score
        late = ordered[-1].score - ordered[-2].score
        acceleration = late - early
    return AuctionTrajectory(symbol, ordered, round(slope, 4), None if acceleration is None else round(acceleration, 4), "COMPLETE" if len(ordered) == 4 else "PARTIAL")
