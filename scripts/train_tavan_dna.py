#!/usr/bin/env python3
"""Tavan-DNA expanding walk-forward + untouched final holdout on a REAL OHLCV CSV.

The CSV must come from a licensed BIST historical provider (columns: symbol,
timestamp, open, high, low, close, volume). Results produced from synthetic
data are not performance evidence.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from bist_hunter.tavan_validation import report_to_rows, run_tavan_walk_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Historical BIST OHLCV CSV (real data)")
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--holdout", type=float, default=0.15)
    parser.add_argument("--tune-threshold", action="store_true",
                        help="choose probability cut-off on development folds only")
    args = parser.parse_args()
    frame = pd.read_csv(args.csv, parse_dates=["timestamp"])
    report = run_tavan_walk_forward(
        frame, folds=args.folds, final_holdout_fraction=args.holdout,
        threshold=None if args.tune_threshold else 0.50,
    )
    print(json.dumps({
        "threshold": report.threshold,
        "development_period": report.development_period,
        "holdout_period": report.holdout_period,
        "mean_fold_precision": report.mean_fold_precision,
        "mean_fold_expectancy": report.mean_fold_expectancy,
        "rows": report_to_rows(report),
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
