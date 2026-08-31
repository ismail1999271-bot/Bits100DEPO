"""End-to-end smoke pipeline used by CI and local research runs."""
from .backtest.engine import CostModel, simulate_trade
from .pipeline import LabelConfig, add_forward_labels, normalize_ohlcv
from .synthetic import make_dataset


def run_smoke() -> dict:
    frame = normalize_ohlcv(make_dataset())
    labeled = add_forward_labels(frame, LabelConfig(limit_pct=0.10, horizon_bars=1))
    usable = labeled[labeled["label_available"]]
    predictions = usable[["timestamp", "hit_limit_forward"]].rename(columns={"hit_limit_forward": "actual"})
    predictions["prediction"] = predictions["actual"]
    return {
        "rows": len(frame),
        "labeled_rows": len(usable),
        "positive_events": int(usable["actual"].sum()),
        "example_trade_net_return": simulate_trade("AAA", 100, 110, 10_000, CostModel()).net_return,
    }
