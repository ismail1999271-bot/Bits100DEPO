"""Daily BIST opportunity ranking from leakage-safe OHLCV and institutional features."""
from dataclasses import dataclass

import pandas as pd

from .institutional_intelligence import InstitutionalWeights, build_institutional_score
from .technical_watchlist import add_risk_reward


@dataclass(frozen=True, slots=True)
class RankingConfig:
    min_score: float = 65.0
    top_k: int = 20
    technical_weight: float = 0.10
    institutional_weights: InstitutionalWeights = InstitutionalWeights()

    def __post_init__(self) -> None:
        if not 0 <= self.technical_weight <= 1:
            raise ValueError("technical_weight must be between 0 and 1")


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
    """Rank latest observations and optionally incorporate broker technical risk/reward data."""
    enriched = add_features(frame)
    latest = enriched.sort_values("timestamp").groupby("symbol", as_index=False).tail(1)
    institutional_columns = {
        "smart_money_score",
        "consensus_score",
        "research_score",
        "fundamental_score",
    }
    if institutional_columns.intersection(latest.columns):
        latest = build_institutional_score(latest, weights=config.institutional_weights)
        score_column = "institutional_score"
    else:
        latest["institutional_score"] = latest["score"]
        latest["institutional_data_coverage"] = 0.0
        score_column = "score"

    watchlist_columns = {"entry_low", "entry_high", "target", "stop", "last_price"}
    if watchlist_columns.issubset(latest.columns):
        latest = add_risk_reward(latest)
        latest["institutional_score"] = (
            (1 - config.technical_weight) * latest["institutional_score"]
            + config.technical_weight * latest["technical_watch_score"]
        ).round(2)
        latest["technical_risk_gate"] = latest["risk_reward"] >= 1.0
        latest = latest[latest["technical_risk_gate"]]
        score_column = "institutional_score"
    else:
        latest["technical_watch_score"] = pd.NA
        latest["technical_risk_gate"] = pd.NA

    return (
        latest[latest[score_column] >= config.min_score]
        .sort_values(score_column, ascending=False)
        .head(config.top_k)
        .reset_index(drop=True)
    )
