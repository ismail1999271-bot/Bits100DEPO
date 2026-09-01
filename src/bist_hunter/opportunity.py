"""Single facade for daily BIST opportunity ranking.

The score is a ranking signal, not a guaranteed return forecast.
"""
from dataclasses import dataclass
import pandas as pd
from .ranking import RankingConfig, rank_latest

@dataclass(frozen=True, slots=True)
class OpportunityConfig:
    top_k: int = 20
    min_score: float = 65.0
    no_signal_below: float = 65.0

def rank_daily(frame: pd.DataFrame, config: OpportunityConfig = OpportunityConfig()) -> pd.DataFrame:
    ranked = rank_latest(frame, RankingConfig(min_score=config.min_score, top_k=config.top_k))
    if ranked.empty:
        return ranked.assign(signal="NO_QUALITY_SIGNAL")
    ranked = ranked.copy()
    ranked["signal"] = ranked["score"].apply(lambda x: "STRONG" if x >= 85 else "WATCH")
    return ranked
