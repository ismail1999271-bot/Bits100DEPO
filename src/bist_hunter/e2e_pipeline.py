"""End-to-end deterministic research pipeline.

OHLCV -> leakage-safe Tavan-DNA probability -> normalized components -> Quant
Score -> auction trajectory. The module only ranks supplied observations and
never places exchange orders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .auction_features import TZ as ISTANBUL
from .auction_features import AuctionTrajectoryFeatures
from .auction_trajectory import AuctionTrajectory, build_trajectory
from .quant_score import QuantScore, calculate_quant_score
from .tavan_model import TavanLogisticModel, build_tavan_dataset
from .walk_forward import purged_date_split


@dataclass(frozen=True, slots=True)
class E2ECandidate:
    symbol: str
    timestamp: str
    tavan_probability: float
    quant_score: QuantScore
    trajectory: AuctionTrajectory


def _external_component(latest: dict[str, Any], key: str) -> float | None:
    value = latest.get(key)
    if value is None:
        return None
    value = float(value)
    if not 0 <= value <= 100:
        raise ValueError(f"{key} must be between 0 and 100")
    return value


def session_date(timestamp) -> str:
    """Istanbul trading date of a timestamp (naive timestamps are taken as local)."""
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(ISTANBUL)
    return ts.date().isoformat()


def row_components(
    symbol: str,
    timestamp,
    tavan_probability: float,
    component_overrides: dict[str, dict[str, float]] | None,
    auction_features: dict[tuple[str, str], AuctionTrajectoryFeatures] | None,
) -> dict[str, float]:
    """Point-in-time Quant components for one symbol-day decision at the close."""
    components: dict[str, float] = {"tavan_dna": float(tavan_probability) * 100.0}
    for key, value in (component_overrides or {}).get(symbol, {}).items():
        components[key] = float(value)
    auction = (auction_features or {}).get((symbol, session_date(timestamp)))
    if auction is not None and auction.component_score is not None \
            and not auction.status.startswith("BLOCKED"):
        # Same-day 09:40-09:55 auction is known before the close-of-day decision.
        components["auction"] = auction.component_score
    return components


def run_e2e_backtest(
    frame: pd.DataFrame,
    *,
    trajectory_points: dict[str, list[dict[str, Any]]] | None = None,
    component_overrides: dict[str, dict[str, float]] | None = None,
    auction_features: dict[tuple[str, str], AuctionTrajectoryFeatures] | None = None,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    threshold: float = 0.50,
) -> tuple[E2ECandidate, ...]:
    """Fit Tavan-DNA chronologically and rank only the untouched holdout.

    Quant components other than Tavan-DNA come exclusively from caller-supplied
    point-in-time overrides and same-session auction features keyed by
    ``(symbol, YYYY-MM-DD)``. This prevents the backtest from manufacturing KAP,
    broker, fund or institutional evidence.
    """
    labeled = build_tavan_dataset(frame)
    if len(labeled) < 30:
        raise ValueError("at least 30 labeled observations are required")
    ordered = labeled.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    split = purged_date_split(
        ordered["timestamp"], train_fraction=train_fraction, validation_fraction=validation_fraction
    )
    train = ordered[ordered["timestamp"].isin(split.train_dates)]
    holdout = ordered[ordered["timestamp"].isin(split.holdout_dates)]
    if train["target"].nunique() < 2:
        raise ValueError("training data must contain both target classes")
    model = TavanLogisticModel().fit(train)
    probabilities = model.predict_proba(holdout)
    trajectory_points = trajectory_points or {}
    component_overrides = component_overrides or {}
    output: list[E2ECandidate] = []
    for index, (_, row) in enumerate(holdout.iterrows()):
        symbol = str(row["symbol"]).upper()
        probability = float(probabilities[index])
        components = row_components(
            symbol, row["timestamp"], probability, component_overrides, auction_features
        )
        score = calculate_quant_score(components, min_coverage=0.20)
        points = trajectory_points.get(symbol, [])
        trajectory = build_trajectory(symbol, points)
        if score.status == "SIGNAL" and not trajectory.status.startswith("BLOCKED"):
            output.append(E2ECandidate(symbol, row["timestamp"].isoformat(), probability, score, trajectory))
    output.sort(key=lambda item: (item.quant_score.score, item.quant_score.confidence), reverse=True)
    return tuple(output)
