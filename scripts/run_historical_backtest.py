"""Run the leakage-safe historical Tavan-DNA backtest on provider data."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from bist_hunter.historical_backtest import load_ohlcv_csv, run_historical_backtest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="UTF-8 CSV with symbol,timestamp,open,high,low,close,volume")
    parser.add_argument("--output", default="artifacts/tavan_backtest/latest.json")
    args = parser.parse_args()
    frame = load_ohlcv_csv(args.csv)
    report = run_historical_backtest(frame)
    payload = {
        "rows": report.rows,
        "labeled_rows": report.labeled_rows,
        "train_rows": report.train_rows,
        "validation_rows": report.validation_rows,
        "holdout_rows": report.holdout_rows,
        "train_positive_rate": report.train_positive_rate,
        "validation_precision": report.validation_precision,
        "holdout_precision": report.holdout_precision,
        "holdout_recall": report.holdout_recall,
        "holdout_returns": asdict(report.holdout_returns),
        "status": "VALIDATED",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
