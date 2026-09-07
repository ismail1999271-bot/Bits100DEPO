"""Technical-follow-list ingestion and risk/reward features.

This module converts broker research watchlists (e.g. ticker, entry range, target,
stop) into deterministic, auditable features. It does not treat a broker list as
an independent trading signal; the resulting features are inputs to the Bits100
ranking/risk engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"symbol", "last_price", "entry_low", "entry_high", "target", "stop"}


def validate_watchlist(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out = frame.copy()
    numeric = ["last_price", "entry_low", "entry_high", "target", "stop"]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[numeric].isna().any().any():
        raise ValueError("watchlist contains non-numeric or missing price levels")
    invalid = (
        (out["entry_low"] > out["entry_high"])
        | (out["stop"] >= out["entry_low"])
        | (out["target"] <= out["entry_high"])
        | (out["last_price"] <= 0)
    )
    if bool(invalid.any()):
        raise ValueError("invalid watchlist levels: expected stop < entry range < target")
    return out


def add_risk_reward(frame: pd.DataFrame) -> pd.DataFrame:
    """Add entry-range distance, target upside and stop downside features."""
    out = validate_watchlist(frame)
    reference = out["last_price"]
    out["target_upside_pct"] = (out["target"] / reference - 1.0) * 100
    out["stop_downside_pct"] = (1.0 - out["stop"] / reference) * 100
    out["entry_distance_pct"] = np.where(
        reference < out["entry_low"],
        (out["entry_low"] / reference - 1.0) * 100,
        np.where(reference > out["entry_high"], (reference / out["entry_high"] - 1.0) * 100, 0.0),
    )
    risk = out["entry_high"] - out["stop"]
    reward = out["target"] - out["entry_high"]
    out["risk_reward"] = (reward / risk).replace([np.inf, -np.inf], np.nan)
    out["risk_reward"] = out["risk_reward"].round(3)
    out["technical_watch_score"] = (
        50
        + np.clip(out["target_upside_pct"], -20, 30) * 1.2
        - np.clip(out["stop_downside_pct"], 0, 20) * 1.0
        + np.clip(out["risk_reward"] - 1.0, -1, 3) * 7
    ).clip(0, 100).round(2)
    return out


def merge_technical_watchlist(base: pd.DataFrame, watchlist: pd.DataFrame) -> pd.DataFrame:
    """Attach broker-watchlist features to the main ranking frame by symbol."""
    if "symbol" not in base.columns:
        raise ValueError("base requires symbol")
    technical = add_risk_reward(watchlist)
    features = technical[["symbol", "technical_watch_score", "target_upside_pct", "stop_downside_pct", "risk_reward"]]
    out = base.drop(columns=[c for c in features.columns if c != "symbol" and c in base.columns], errors="ignore").merge(
        features, on="symbol", how="left"
    )
    return out
