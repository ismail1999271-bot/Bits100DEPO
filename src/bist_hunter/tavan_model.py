"""Leakage-safe Tavan-DNA feature extraction and small logistic model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = ("ret1", "ret5", "volume_ratio", "close_location", "range_pct")


def build_tavan_dataset(frame: pd.DataFrame, limit_pct: float = 0.10) -> pd.DataFrame:
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    df = frame.copy().sort_values(["symbol", "timestamp"])
    grouped = df.groupby("symbol", group_keys=False)
    df["ret1"] = grouped["close"].pct_change()
    df["ret5"] = grouped["close"].pct_change(5)
    df["volume_ratio"] = df["volume"] / grouped["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    df["close_location"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    future_high = grouped["high"].shift(-1)
    df["target"] = (future_high >= df["close"] * (1 + limit_pct)).astype(float)
    return df.dropna(subset=list(FEATURES)).copy()


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


def chronological_train_test(frame: pd.DataFrame, test_fraction: float = 0.2) -> TavanModelResult:
    ordered = frame.sort_values("timestamp")
    cut = max(1, min(len(ordered) - 1, int(len(ordered) * (1 - test_fraction))))
    train, test = ordered.iloc[:cut], ordered.iloc[cut:]
    model = TavanLogisticModel().fit(train)
    probabilities = model.predict_proba(test)
    predicted = probabilities >= 0.5
    actual = test["target"].to_numpy(bool)
    tp = int(np.sum(predicted & actual))
    precision = tp / max(1, int(predicted.sum()))
    recall = tp / max(1, int(actual.sum()))
    return TavanModelResult(len(train), len(test), round(float(precision), 6), round(float(recall), 6), tuple(np.round(model.weights_[1:], 6)))
