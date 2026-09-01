"""Cost-aware, chronological backtesting primitives."""
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True, slots=True)
class TradeResult:
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    gross_return: float
    net_return: float

@dataclass(frozen=True, slots=True)
class CostModel:
    commission_bps: float = 10.0
    slippage_bps: float = 5.0
    @property
    def round_trip_rate(self) -> float:
        return 2 * (self.commission_bps + self.slippage_bps) / 10_000

def simulate_trade(symbol: str, entry_price: float, exit_price: float, capital: float, costs: CostModel) -> TradeResult:
    if min(entry_price, exit_price, capital) <= 0:
        raise ValueError("prices and capital must be positive")
    gross = exit_price / entry_price - 1
    return TradeResult(symbol, entry_price, exit_price, capital / entry_price, gross, gross - costs.round_trip_rate)

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
    if "score" in df.columns:
        df["prediction"] = df["score"] >= threshold
    else:
        df["prediction"] = df["prediction"].astype(bool)
    df["actual"] = df["actual"].astype(bool)
    tp = int((df["prediction"] & df["actual"]).sum())
    fp = int((df["prediction"] & ~df["actual"]).sum())
    fn = int((~df["prediction"] & df["actual"]).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    daily = df.groupby(df["timestamp"].dt.date)["prediction"].sum()
    hits = df[df["prediction"] & df["actual"]].assign(month=lambda x: x["timestamp"].dt.to_period("M"))
    monthly = hits.groupby("month").size()
    return Metrics(len(df), int(df["actual"].sum()), tp, fp, fn, precision, recall, int((daily > 0).sum()), len(daily), float((daily > 0).mean()) if len(daily) else 0.0, float(daily.mean()) if len(daily) else 0.0, int((monthly >= 10).sum()) if len(monthly) else 0, float(monthly.mean()) if len(monthly) else 0.0)

def walk_forward_splits(frame: pd.DataFrame, train_days: int = 252, test_days: int = 21):
    if train_days < 1 or test_days < 1:
        raise ValueError("train_days and test_days must be positive")
    x = frame.sort_values("timestamp").reset_index(drop=True)
    i = train_days
    while i < len(x):
        yield x.iloc[i - train_days:i].copy(), x.iloc[i:i + test_days].copy()
        i += test_days
