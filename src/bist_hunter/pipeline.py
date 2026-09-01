"""Historical OHLCV normalization, labeling, and candidate scoring facade."""
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable
import pandas as pd

from .scoring import CandidateFeatures, score_candidate as _score_features

REQUIRED_COLUMNS = ("symbol", "timestamp", "open", "high", "low", "close", "volume")

@dataclass(frozen=True, slots=True)
class LabelConfig:
    limit_pct: float = 0.10
    horizon_bars: int = 1

@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train: pd.DataFrame
    test: pd.DataFrame

@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """Compatibility facade for tests and simple rule-based ranking."""
    symbol: str
    pattern: float = 0.0
    catalyst: float = 0.0
    volume_anomaly: float = 0.0
    manipulation_risk: float = 0.0


def normalize_ohlcv(rows: Iterable[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    frame = frame[list(REQUIRED_COLUMNS)].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    for col in REQUIRED_COLUMNS[2:]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if (frame[["open", "high", "low", "close", "volume"]] < 0).any().any():
        raise ValueError("OHLCV values cannot be negative")
    return (frame.drop_duplicates(["symbol", "timestamp"], keep="last")
            .sort_values(["symbol", "timestamp"]).reset_index(drop=True))


def add_forward_labels(frame: pd.DataFrame, config: LabelConfig = LabelConfig()) -> pd.DataFrame:
    if config.limit_pct <= 0 or config.horizon_bars < 1:
        raise ValueError("invalid label configuration")
    result = frame.copy()
    future_high = result.groupby("symbol", sort=False)["high"].shift(-config.horizon_bars)
    result["hit_limit_forward"] = (future_high >= result["close"] * (1 + config.limit_pct)).fillna(False)
    result["label_available"] = future_high.notna()
    return result


def time_split(frame: pd.DataFrame, test_start: date | datetime) -> DatasetSplit:
    cutoff = pd.Timestamp(test_start, tz="UTC")
    return DatasetSplit(frame[frame["timestamp"] < cutoff].copy(), frame[frame["timestamp"] >= cutoff].copy())


def score_candidate(candidate: CandidateEvidence) -> float:
    features = CandidateFeatures(
        volume_anomaly=candidate.volume_anomaly,
        price_momentum=candidate.pattern,
        catalyst_strength=candidate.catalyst,
        manipulation_risk=candidate.manipulation_risk,
    )
    return _score_features(features)[0]


def rank_candidates(candidates: list[CandidateEvidence]) -> list[tuple[str, float]]:
    ranked = [(c.symbol, score_candidate(c)) for c in candidates]
    return sorted(ranked, key=lambda item: item[1], reverse=True)
