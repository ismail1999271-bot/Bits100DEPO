"""End-to-end deterministic research pipeline.

OHLCV -> leakage-safe Tavan-DNA probability -> normalized components -> Quant
Score -> auction trajectory. The module only ranks supplied observations and
never places exchange orders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .auction_trajectory import AuctionTrajectory, build_trajectory
from .quant_score import QuantScore, calculate_quant_score
from .tavan_model import TavanLogisticModel, build_tavan_dataset


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


def run_e2e_backtest(
    frame: pd.DataFrame,
    *,
    trajectory_points: dict[str, list[dict[str, Any]]] | None = None,
    component_overrides: dict[str, dict[str, float]] | None = None,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    threshold: float = 0.50,
) -> tuple[E2ECandidate, ...]:
    """Fit Tavan-DNA chronologically and rank only the untouched holdout.

    Quant components other than Tavan-DNA come exclusively from caller-supplied
    point-in-time overrides. This prevents the backtest from manufacturing KAP,
    broker, fund or institutional evidence.
    """
    labeled = build_tavan_dataset(frame)
    if len(labeled) < 30:
        raise ValueError("at least 30 labeled observations are required")
    ordered = labeled.sort_values("timestamp").reset_index(drop=True)
    train_end = max(1, int(len(ordered) * train_fraction))
    validation_end = max(train_end + 1, int(len(ordered) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(ordered) - 1)
    train = ordered.iloc[:train_end]
    holdout = ordered.iloc[validation_end:]
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
        components: dict[str, float] = {"tavan_dna": probability * 100.0}
        for key, value in component_overrides.get(symbol, {}).items():
            components[key] = float(value)
        score = calculate_quant_score(components, min_coverage=0.20)
        points = trajectory_points.get(symbol, [])
        trajectory = build_trajectory(symbol, points)
        if score.status == "SIGNAL" and trajectory.status != "BLOCKED":
            output.append(E2ECandidate(symbol, row["timestamp"].isoformat(), probability, score, trajectory))
    output.sort(key=lambda item: (item.quant_score.score, item.quant_score.confidence), reverse=True)
    return tuple(output)
