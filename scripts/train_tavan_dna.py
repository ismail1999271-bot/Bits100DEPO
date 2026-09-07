#!/usr/bin/env python3
"""Train/evaluate Tavan-DNA from a real historical OHLCV CSV."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from bist_hunter.tavan_model import build_tavan_dataset, chronological_train_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Historical BIST OHLCV CSV")
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    result = chronological_train_test(build_tavan_dataset(frame))
    print(json.dumps({
        "train_rows": result.train_rows,
        "test_rows": result.test_rows,
        "test_precision": result.test_precision,
        "test_recall": result.test_recall,
        "coefficients": result.coefficients,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
