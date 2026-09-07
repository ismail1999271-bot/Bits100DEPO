"""Daily BIST opportunity ranking from leakage-safe OHLCV and institutional features."""
from dataclasses import dataclass

import pandas as pd

from .institutional_intelligence import InstitutionalWeights, build_institutional_score


@dataclass(frozen=True, slots=True)
class RankingConfig:
    min_score: float = 65.0
    top_k: int = 20
    institutional_weights: InstitutionalWeights = InstitutionalWeights()


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    g = result.groupby("symbol", sort=False)
    result["ret_1"] = g["close"].pct_change()
    result["ret_5"] = g["close"].pct_change(5)
    result["vol_mean_20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    result["volume_ratio"] = result["volume"] / result["vol_mean_20"].replace(0, pd.NA)
    result["high_20_prev"] = g["high"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).max())
    result["breakout"] = result["close"] / result["high_20_prev"] - 1
    result["score"] = (
        50
        + result["ret_1"].fillna(0).clip(-0.05, 0.10) * 250
        + result["ret_5"].fillna(0).clip(-0.15, 0.30) * 100
        + (result["volume_ratio"].fillna(1).clip(0, 5) - 1) * 8
        + result["breakout"].fillna(0).clip(-0.10, 0.10) * 150
    ).clip(0, 100)
    return result


def rank_latest(frame: pd.DataFrame, config: RankingConfig = RankingConfig()) -> pd.DataFrame:
    """Rank the latest observation per symbol.

    If institutional layer columns are present, they are blended into ``institutional_score``.
    Existing OHLCV-only callers retain the legacy score while receiving a 50-point neutral
    institutional baseline. This allows connectors to be rolled out incrementally.
    """
    enriched = add_features(frame)
    latest = enriched.sort_values("timestamp").groupby("symbol", as_index=False).tail(1)
    latest = build_institutional_score(latest, weights=config.institutional_weights)
    return (
        latest[latest["institutional_score"] >= config.min_score]
        .sort_values("institutional_score", ascending=False)
        .head(config.top_k)
        .reset_index(drop=True)
    )
