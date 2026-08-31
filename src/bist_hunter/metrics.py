"""Evaluation metrics for the daily opportunity ranking objective."""
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True, slots=True)
class RankingMetrics:
    precision: float
    recall: float
    daily_hit_day_rate: float
    average_daily_hits: float


def evaluate_ranking(y_true: pd.Series, y_pred: pd.Series, days: pd.Series | None = None) -> RankingMetrics:
    true = y_true.astype(bool)
    pred = y_pred.astype(bool)
    tp = int((true & pred).sum())
    fp = int((~true & pred).sum())
    fn = int((true & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if days is None:
        hit_rate = 1.0 if tp else 0.0
        avg_hits = float(tp)
    else:
        frame = pd.DataFrame({"day": pd.to_datetime(days).dt.date, "pred": pred})
        daily = frame.groupby("day")["pred"].any()
        hit_rate = float(daily.mean()) if len(daily) else 0.0
        avg_hits = float(frame.groupby("day")["pred"].sum().mean()) if len(daily) else 0.0
    return RankingMetrics(precision, recall, hit_rate, avg_hits)
