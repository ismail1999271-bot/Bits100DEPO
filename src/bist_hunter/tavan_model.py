"""Leakage-safe Tavan-DNA feature extraction and baseline model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = ("ret1", "ret5", "volume_ratio", "close_location", "range_pct", "breakout_20", "volatility_20", "gap")


def build_tavan_dataset(frame: pd.DataFrame, limit_pct: float = 0.10) -> pd.DataFrame:
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    df = frame.copy().sort_values(["symbol", "timestamp"])
    grouped = df.groupby("symbol", group_keys=False)
    df["ret1"] = grouped["close"].pct_change()
    df["ret5"] = grouped["close"].pct_change(5)
    df["volume_ratio"] = df["volume"] / grouped["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean()).replace(0, np.nan)
    df["close_location"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["breakout_20"] = df["close"] / grouped["high"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).max()) - 1
    df["volatility_20"] = grouped["close"].transform(lambda s: s.pct_change().rolling(20, min_periods=10).std())
    df["gap"] = df["open"] / grouped["close"].shift(1) - 1
    future_high = grouped["high"].shift(-1)
    df["target"] = (future_high >= df["close"] * (1 + limit_pct)).astype(float)
    return df.loc[future_high.notna()].dropna(subset=list(FEATURES)).copy()


@dataclass(frozen=True, slots=True)
class TavanModelResult:
    train_rows: int
    test_rows: int
    test_precision: float
    test_recall: float
    coefficients: tuple[float, ...]


class TavanLogisticModel:
    def __init__(self, learning_rate: float = 0.05, epochs: int = 400, l2: float = 0.01) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "TavanLogisticModel":
        if frame.empty:
            raise ValueError("cannot fit on empty dataset")
        x = frame[list(FEATURES)].to_numpy(float)
        y = frame["target"].to_numpy(float)
        self.mean_ = x.mean(axis=0)
        self.std_ = np.where(x.std(axis=0) == 0, 1.0, x.std(axis=0))
        z = (x - self.mean_) / self.std_
        design = np.c_[np.ones(len(z)), z]
        weights = np.zeros(design.shape[1])
        for _ in range(self.epochs):
            p = 1 / (1 + np.exp(-np.clip(design @ weights, -30, 30)))
            gradient = design.T @ (p - y) / max(1, len(y))
            gradient[1:] += self.l2 * weights[1:]
            weights -= self.learning_rate * gradient
        self.weights_ = weights
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.weights_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("model is not fitted")
        x = frame[list(FEATURES)].to_numpy(float)
        z = (x - self.mean_) / self.std_
        return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(z)), z] @ self.weights_, -30, 30)))


def _precision_recall(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    tp = int(np.sum(predicted & actual))
    return tp / max(1, int(predicted.sum())), tp / max(1, int(actual.sum()))


def chronological_train_test(frame: pd.DataFrame, test_fraction: float = 0.2) -> TavanModelResult:
    ordered = frame.sort_values("timestamp")
    if len(ordered) < 2:
        raise ValueError("at least two labeled observations are required")
    cut = max(1, min(len(ordered) - 1, int(len(ordered) * (1 - test_fraction))))
    train, test = ordered.iloc[:cut], ordered.iloc[cut:]
    model = TavanLogisticModel().fit(train)
    precision, recall = _precision_recall(test["target"].to_numpy(bool), model.predict_proba(test) >= 0.5)
    return TavanModelResult(len(train), len(test), round(float(precision), 6), round(float(recall), 6), tuple(np.round(model.weights_[1:], 6)))


def walk_forward_precision(frame: pd.DataFrame, folds: int = 5) -> tuple[float, ...]:
    """Expanding-window OOS precision; each fold trains strictly before its test window."""
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    if len(ordered) < max(20, folds * 4):
        raise ValueError("insufficient observations for walk-forward validation")
    edges = np.linspace(0, len(ordered), folds + 2, dtype=int)
    scores: list[float] = []
    for i in range(1, len(edges) - 1):
        train_end, test_end = edges[i], edges[i + 1]
        train, test = ordered.iloc[:train_end], ordered.iloc[train_end:test_end]
        if train.empty or test.empty or train["target"].nunique() < 2:
            continue
        model = TavanLogisticModel().fit(train)
        precision, _ = _precision_recall(test["target"].to_numpy(bool), model.predict_proba(test) >= 0.5)
        scores.append(round(float(precision), 6))
    return tuple(scores)
