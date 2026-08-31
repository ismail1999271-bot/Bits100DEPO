"""Minimal leakage-aware evaluation utilities."""
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True, slots=True)
class Metrics:
    signals: int
    positives: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    months_with_10_hits: int
    average_monthly_hits: float

def evaluate(predictions: pd.DataFrame) -> Metrics:
    required = {"timestamp", "prediction", "actual"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    df = predictions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["prediction"] = df["prediction"].astype(bool)
    df["actual"] = df["actual"].astype(bool)
    tp = int((df.prediction & df.actual).sum())
    fp = int((df.prediction & ~df.actual).sum())
    fn = int((~df.prediction & df.actual).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    monthly = df.loc[df.prediction & df.actual].groupby(df.timestamp.dt.to_period("M")).size()
    avg = float(monthly.mean()) if len(monthly) else 0.0
    ten = int((monthly >= 10).sum()) if len(monthly) else 0
    return Metrics(len(df), int(df.actual.sum()), tp, fp, fn, precision, recall, ten, avg)
