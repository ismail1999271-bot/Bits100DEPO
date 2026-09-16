"""Historical Tavan-DNA training and event-driven backtest utilities.

The module accepts provider-exported OHLCV data only. It never invents market
observations. Labels are computed from bars strictly after the observation
used for the prediction, and chronological splits keep the final holdout
untouched until the end.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .quant_validation import ValidationMetrics, validate_returns
from .tavan_model import FEATURES, TavanLogisticModel, build_tavan_dataset


REQUIRED_COLUMNS = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}


@dataclass(frozen=True, slots=True)
class BacktestReport:
    rows: int
    labeled_rows: int
    train_rows: int
    validation_rows: int
    holdout_rows: int
    train_positive_rate: float
    validation_precision: float
    holdout_precision: float
    holdout_recall: float
    holdout_returns: ValidationMetrics


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load a UTF-8 provider export and reject incomplete market data."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("duplicate symbol/timestamp observations")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("non-positive OHLC observation")
    if (frame["volume"] < 0).any():
        raise ValueError("negative volume observation")
    return frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _chronological_three_way(frame: pd.DataFrame, train_fraction: float = 0.60, validation_fraction: float = 0.20):
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("fractions must be positive and leave a holdout")
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    train_end = max(1, int(len(ordered) * train_fraction))
    validation_end = max(train_end + 1, int(len(ordered) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(ordered) - 1)
    return ordered.iloc[:train_end], ordered.iloc[train_end:validation_end], ordered.iloc[validation_end:]


def _precision(actual: pd.Series, probability: pd.Series, threshold: float) -> float:
    predicted = probability >= threshold
    tp = int((predicted & actual.astype(bool)).sum())
    fp = int((predicted & ~actual.astype(bool)).sum())
    return tp / (tp + fp) if tp + fp else 0.0


def run_historical_backtest(frame: pd.DataFrame, *, limit_pct: float = 0.10, threshold: float = 0.50) -> BacktestReport:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    labeled = build_tavan_dataset(frame, limit_pct=limit_pct)
    if len(labeled) < 30:
        raise ValueError("at least 30 labeled observations are required")
    train, validation, holdout = _chronological_three_way(labeled)
    model = TavanLogisticModel().fit(train)
    validation_probability = pd.Series(model.predict_proba(validation), index=validation.index)
    holdout_probability = pd.Series(model.predict_proba(holdout), index=holdout.index)
    validation_precision = _precision(validation["target"], validation_probability, threshold)
    holdout_prediction = holdout_probability >= threshold
    # Entry is made at the current close and exited at the next bar close only
    # for rows whose next close is available inside the untouched holdout.
    next_close = holdout.groupby("symbol")["close"].shift(-1)
    valid_returns = (next_close / holdout["close"] - 1).where(holdout_prediction & next_close.notna()).dropna()
    metrics = validate_returns(valid_returns.tolist(), cost_bps=10.0, slippage_bps=5.0)
    return BacktestReport(
        rows=len(frame),
        labeled_rows=len(labeled),
        train_rows=len(train),
        validation_rows=len(validation),
        holdout_rows=len(holdout),
        train_positive_rate=round(float(train["target"].mean()), 6),
        validation_precision=round(validation_precision, 6),
        holdout_precision=round(_precision(holdout["target"], holdout_probability, threshold), 6),
        holdout_recall=round(float(((holdout_prediction & holdout["target"].astype(bool)).sum()) / max(1, int(holdout["target"].sum()))), 6),
        holdout_returns=metrics,
    )
