"""Chronological benchmark and model rejection rules."""
from dataclasses import dataclass
from typing import Callable
import pandas as pd

from .backtest.engine import walk_forward_splits


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    train_days: int = 252
    test_days: int = 21
    min_precision: float = 0.35
    min_hit_day_rate: float = 0.25


def walk_forward_benchmark(
    frame: pd.DataFrame,
    predictor: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
    config: BenchmarkConfig = BenchmarkConfig(),
) -> list[pd.DataFrame]:
    """Run a predictor only on future folds; predictor owns feature fitting."""
    results = []
    for train, test in walk_forward_splits(frame, config.train_days, config.test_days):
        predicted = predictor(train, test.copy())
        if not isinstance(predicted, pd.DataFrame):
            raise TypeError("predictor must return a DataFrame")
        results.append(predicted)
    return results


def accept_model(precision: float, hit_day_rate: float, config: BenchmarkConfig = BenchmarkConfig()) -> bool:
    return precision >= config.min_precision and hit_day_rate >= config.min_hit_day_rate
