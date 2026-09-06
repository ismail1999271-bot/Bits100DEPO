"""Trading-signal evaluation metrics used for model selection."""
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True, slots=True)
class OpportunityMetrics:
    top_k_precision: float
    hit_day_rate: float
    recall: float
    mean_lead_days: float
    net_pnl: float
    max_drawdown: float


def evaluate_opportunities(
    predictions: pd.DataFrame,
    top_k: int = 5,
    event_threshold: float = 0.10,
) -> OpportunityMetrics:
    required = {"timestamp", "symbol", "score", "future_return"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    df = predictions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["timestamp", "score"], ascending=[True, False])
    df["event"] = df["future_return"] >= event_threshold
    selected = df.groupby("timestamp", sort=True).head(top_k)
    tp = int(selected["event"].sum())
    selected_events = int(selected.shape[0])
    event_total = int(df["event"].sum())
    top_k_precision = tp / selected_events if selected_events else 0.0
    recall = tp / event_total if event_total else 0.0
    day_hits = selected.groupby("timestamp")["event"].any()
    total_days = df["timestamp"].nunique()
    hit_day_rate = float(day_hits.mean()) if total_days else 0.0
    event_rows = df[df["event"]]
    first_hit = selected[selected["event"]].groupby("timestamp").size()
    mean_lead_days = float(first_hit.index.to_series().diff().dt.total_seconds().dropna().mean() / 86400) if len(first_hit) > 1 else 0.0
    daily_pnl = selected.groupby("timestamp")["future_return"].mean().fillna(0.0)
    equity = (1.0 + daily_pnl).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return OpportunityMetrics(
        round(top_k_precision, 6),
        round(hit_day_rate, 6),
        round(recall, 6),
        round(max(0.0, mean_lead_days), 6),
        round(float(equity.iloc[-1] - 1.0) if len(equity) else 0.0, 6),
        round(float(drawdown.min()) if len(drawdown) else 0.0, 6),
    )
