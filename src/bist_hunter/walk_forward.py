"""Chronological walk-forward and untouched final-holdout helpers."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimeSplit:
    train: tuple[int, ...]
    test: tuple[int, ...]


def walk_forward_splits(n_samples: int, train_size: int, test_size: int, step: int | None = None) -> tuple[TimeSplit, ...]:
    """Create chronological expanding-window splits; future rows never enter train."""
    if min(n_samples, train_size, test_size) <= 0:
        raise ValueError("sample sizes must be positive")
    step = test_size if step is None else step
    if step <= 0:
        raise ValueError("step must be positive")
    splits: list[TimeSplit] = []
    start = train_size
    while start + test_size <= n_samples:
        splits.append(TimeSplit(tuple(range(start)), tuple(range(start, start + test_size))))
        start += step
    return tuple(splits)


def final_holdout(n_samples: int, holdout_size: int) -> TimeSplit:
    """Return the final contiguous block as test data and all prior data as train."""
    if holdout_size <= 0 or holdout_size >= n_samples:
        raise ValueError("holdout_size must be between 1 and n_samples-1")
    cut = n_samples - holdout_size
    return TimeSplit(tuple(range(cut)), tuple(range(cut, n_samples)))
