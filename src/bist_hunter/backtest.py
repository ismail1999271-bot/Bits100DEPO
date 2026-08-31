"""Leakage-aware walk-forward evaluation and daily opportunity metrics."""
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
    hit_days: int
    total_days: int
    daily_hit_day_rate: float
    average_daily_hits: float
    months_with_10_hits: int
    average_monthly_hits: float


def evaluate(predictions: pd.DataFrame, threshold: float = 0.5) -> Metrics:
    required = {"timestamp", "prediction", "actual"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    df = predictions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "score" in df:
        df["prediction"] = df["score"] >= threshold
    else:
        df["prediction"] = df["prediction"].astype(bool)
    df["actual"] = df["actual"].astype(bool)
    tp = int((df.prediction & df.actual).sum())
    fp = int((df.prediction & ~df.actual).sum())
    fn = int((~df.prediction & df.actual).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    daily = df.groupby(df.timestamp.dt.date).apply(lambda x: int((x.prediction & x.actual).sum()), include_groups=False)
    total_days = len(daily)
    hit_days = int((daily > 0).sum())
    rate = hit_days / total_days if total_days else 0.0
    monthly = df.loc[df.prediction & df.actual].groupby(df.loc[df.prediction & df.actual, "timestamp"].dt.to_period("M")).size()
    avg_month = float(monthly.mean()) if len(monthly) else 0.0
    ten = int((monthly >= 10).sum()) if len(monthly) else 0
    return Metrics(len(df), int(df.actual.sum()), tp, fp, fn, precision, recall, hit_days, total_days, rate, float(daily.mean()) if total_days else 0.0, ten, avg_month)


def walk_forward_splits(frame: pd.DataFrame, train_days: int = 252, test_days: int = 21):
    """Yield chronological train/test windows; never shuffle market observations."""
    if train_days < 1 or test_days < 1:
        raise ValueError("train_days and test_days must be positive")
    x = frame.sort_values("timestamp").reset_index(drop=True)
    i = train_days
    while i < len(x):
        yield x.iloc[i - train_days:i].copy(), x.iloc[i:min(i + test_days, len(x))].copy()
        i += test_days
