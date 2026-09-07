"""Institutional-intelligence primitives inspired by institutional research workflows.

The module deliberately contains no vendor-specific SDK calls. Vendor feeds are adapted into
point-in-time records, then scored with deterministic, auditable functions. Missing data is
never fabricated and stale observations are surfaced through freshness penalties.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


class EvidenceKind(StrEnum):
    MARKET = "market"
    FUNDAMENTAL = "fundamental"
    ANALYST = "analyst"
    NEWS = "news"
    FILING = "filing"
    TRANSCRIPT = "transcript"
    INSTITUTIONAL_HOLDING = "institutional_holding"
    FUND_FLOW = "fund_flow"
    PRIVATE_MARKET = "private_market"
    ALTERNATIVE = "alternative"


@dataclass(frozen=True, slots=True)
class Evidence:
    symbol: str
    observed_at: datetime
    source: str
    kind: EvidenceKind
    value: float
    confidence: float = 1.0
    as_of: datetime | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.as_of is not None and self.as_of > self.observed_at:
            raise ValueError("as_of cannot be later than observed_at")


@dataclass(frozen=True, slots=True)
class InstitutionalWeights:
    market: float = 0.50
    smart_money: float = 0.20
    consensus: float = 0.10
    research: float = 0.10
    fundamentals: float = 0.10

    def __post_init__(self) -> None:
        values = (self.market, self.smart_money, self.consensus, self.research, self.fundamentals)
        if any(v < 0 for v in values) or not np.isclose(sum(values), 1.0):
            raise ValueError("institutional weights must be non-negative and sum to 1")


def _clip(series: pd.Series | float, low: float = 0.0, high: float = 100.0):
    return np.clip(series, low, high)


def validate_point_in_time(frame: pd.DataFrame, *, as_of_col: str = "as_of", observed_col: str = "observed_at") -> pd.DataFrame:
    """Reject future information and return a defensive copy."""
    result = frame.copy()
    if as_of_col not in result or observed_col not in result:
        raise ValueError(f"required columns: {as_of_col}, {observed_col}")
    result[as_of_col] = pd.to_datetime(result[as_of_col], utc=True)
    result[observed_col] = pd.to_datetime(result[observed_col], utc=True)
    bad = result[as_of_col] > result[observed_col]
    if bool(bad.any()):
        raise ValueError("lookahead detected: as_of is later than observed_at")
    return result


def institutional_consensus(positions: pd.DataFrame, *, as_of: datetime | None = None) -> pd.DataFrame:
    """Score cross-manager ownership concentration and position changes (0-100).

    Expected columns: symbol, manager, weight, change, as_of. ``change`` is the change in
    portfolio weight; it is not interpreted as a daily trade ledger.
    """
    required = {"symbol", "manager", "weight", "change", "as_of"}
    missing = required.difference(positions.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    data = validate_point_in_time(positions, as_of_col="as_of", observed_col="as_of")
    if as_of is not None:
        cutoff = pd.Timestamp(as_of, tz="UTC") if pd.Timestamp(as_of).tz is None else pd.Timestamp(as_of)
        data = data[data["as_of"] <= cutoff]
    if data.empty:
        return pd.DataFrame(columns=["symbol", "manager_count", "fund_count", "consensus_score"])

    grouped = data.groupby("symbol", sort=False)
    out = grouped.agg(
        manager_count=("manager", "nunique"),
        fund_count=("manager", "size"),
        mean_weight=("weight", "mean"),
        max_weight=("weight", "max"),
        mean_change=("change", "mean"),
        positive_managers=("change", lambda s: int((s > 0).sum())),
        exits=("change", lambda s: int((s < -0.005).sum())),
    ).reset_index()
    breadth = _clip(out["manager_count"] / 10 * 100)
    weight = _clip(out["mean_weight"] / 0.10 * 100)
    change = _clip(50 + out["mean_change"] / 0.02 * 50)
    entry = _clip(out["positive_managers"] / out["manager_count"].replace(0, 1) * 100)
    exit_penalty = _clip(out["exits"] / out["manager_count"].replace(0, 1) * 100)
    out["consensus_score"] = _clip(0.30 * breadth + 0.25 * weight + 0.25 * change + 0.20 * entry - 0.20 * exit_penalty).round(2)
    return out.sort_values("consensus_score", ascending=False).reset_index(drop=True)


def estimate_exposure(positions: pd.DataFrame) -> pd.DataFrame:
    """Estimate position value from AUM and portfolio weight without inventing AUM."""
    required = {"symbol", "manager", "weight"}
    missing = required.difference(positions.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out = positions.copy()
    if "aum_try" in out:
        out["estimated_exposure_try"] = out["aum_try"] * out["weight"]
    else:
        out["estimated_exposure_try"] = np.nan
    return out


def analyst_revision_score(estimates: pd.DataFrame) -> pd.DataFrame:
    """Score analyst estimate revisions without using future observations."""
    required = {"symbol", "period", "estimate", "observed_at"}
    missing = required.difference(estimates.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    data = estimates.sort_values(["symbol", "period", "observed_at"]).copy()
    data["revision"] = data.groupby(["symbol", "period"])["estimate"].pct_change()
    score = data.groupby("symbol", sort=False)["revision"].last().fillna(0)
    return pd.DataFrame({"symbol": score.index, "analyst_revision_score": _clip(50 + score * 500).round(2)}).reset_index(drop=True)


def research_signal_score(documents: pd.DataFrame) -> pd.DataFrame:
    """Aggregate document/transcript sentiment weighted by source confidence and freshness."""
    required = {"symbol", "sentiment", "confidence", "observed_at"}
    missing = required.difference(documents.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    data = documents.copy()
    data["weight"] = pd.to_numeric(data["confidence"], errors="coerce").fillna(0).clip(0, 1)
    grouped = data.groupby("symbol", sort=False)
    weighted = grouped.apply(lambda g: np.average(g["sentiment"], weights=g["weight"]) if g["weight"].sum() else 0.0, include_groups=False)
    return pd.DataFrame({"symbol": weighted.index, "research_score": _clip(50 + weighted * 50).round(2)}).reset_index(drop=True)


def build_institutional_score(frame: pd.DataFrame, *, weights: InstitutionalWeights = InstitutionalWeights()) -> pd.DataFrame:
    """Build a single auditable 0-100 score from optional institutional layers.

    Columns are optional except ``symbol`` and ``score``. This makes the function safe to use
    while vendor feeds are being connected incrementally.
    """
    if "symbol" not in frame or "score" not in frame:
        raise ValueError("frame requires symbol and score columns")
    out = frame.copy()
    layers = {
        "market": out["score"],
        "smart_money": out.get("smart_money_score", pd.Series(50.0, index=out.index)),
        "consensus": out.get("consensus_score", pd.Series(50.0, index=out.index)),
        "research": out.get("research_score", pd.Series(50.0, index=out.index)),
        "fundamentals": out.get("fundamental_score", pd.Series(50.0, index=out.index)),
    }
    for name, values in layers.items():
        out[f"{name}_layer"] = pd.to_numeric(values, errors="coerce").fillna(50).clip(0, 100)
    out["institutional_score"] = (
        weights.market * out["market_layer"]
        + weights.smart_money * out["smart_money_layer"]
        + weights.consensus * out["consensus_layer"]
        + weights.research * out["research_layer"]
        + weights.fundamentals * out["fundamentals_layer"]
    ).round(2)
    out["institutional_data_coverage"] = out[[f"{n}_layer" for n in layers]].ne(50).mean(axis=1).round(2)
    return out.sort_values("institutional_score", ascending=False).reset_index(drop=True)


def portfolio_risk_summary(positions: pd.DataFrame) -> pd.DataFrame:
    """Compute concentration metrics suitable for an institutional portfolio dashboard."""
    required = {"portfolio", "symbol", "weight"}
    missing = required.difference(positions.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for portfolio, group in positions.groupby("portfolio", sort=False):
        weights = pd.to_numeric(group["weight"], errors="coerce").fillna(0).clip(lower=0)
        total = weights.sum()
        normalized = weights / total if total else weights
        rows.append({
            "portfolio": portfolio,
            "positions": int(len(group)),
            "top1_weight": float(normalized.nlargest(1).sum()),
            "top5_weight": float(normalized.nlargest(5).sum()),
            "herfindahl": float((normalized**2).sum()),
            "effective_positions": float(1 / (normalized**2).sum()) if (normalized**2).sum() else 0.0,
        })
    return pd.DataFrame(rows)
