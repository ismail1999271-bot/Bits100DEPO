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


# ---------------------------------------------------------------------------
# Date-aware, purged splits for panel data (many symbols per timestamp).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DateSplit:
    """Boolean-free description of a chronological panel split by timestamp."""

    train_dates: tuple
    validation_dates: tuple
    holdout_dates: tuple
    purged_dates: tuple


def _unique_sorted(timestamps) -> list:
    return sorted(set(timestamps))


def purged_date_split(
    timestamps,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    embargo_periods: int = 1,
) -> DateSplit:
    """Split unique dates into train / validation / final holdout.

    A symbol-day never straddles two segments (splits are made on dates, not
    rows), and the last ``embargo_periods`` dates before each boundary are
    purged because their labels look ``embargo_periods`` bars into the next
    segment. This removes label leakage across the boundary.
    """
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("fractions must be in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("fractions must leave a final holdout")
    if embargo_periods < 0:
        raise ValueError("embargo_periods must be non-negative")
    dates = _unique_sorted(timestamps)
    n = len(dates)
    if n < 3 + 2 * embargo_periods + 2:
        raise ValueError("insufficient distinct dates for a purged split")
    train_end = max(1 + embargo_periods, int(n * train_fraction))
    validation_end = int(n * (train_fraction + validation_fraction))
    validation_end = max(train_end + 1 + embargo_periods, min(validation_end, n - 1))
    train = dates[: train_end - embargo_periods]
    purged_a = dates[train_end - embargo_periods: train_end]
    validation = dates[train_end: validation_end - embargo_periods]
    purged_b = dates[validation_end - embargo_periods: validation_end]
    holdout = dates[validation_end:]
    if not train or not validation or not holdout:
        raise ValueError("insufficient distinct dates for a purged split")
    return DateSplit(tuple(train), tuple(validation), tuple(holdout), tuple(purged_a + purged_b))


def expanding_date_folds(
    dates,
    *,
    folds: int,
    min_train_dates: int,
    embargo_periods: int = 1,
) -> tuple[tuple[tuple, tuple], ...]:
    """Expanding-window (train_dates, test_dates) folds with an embargo gap."""
    ordered = _unique_sorted(dates)
    if folds < 1 or min_train_dates < 1:
        raise ValueError("folds and min_train_dates must be positive")
    remaining = len(ordered) - min_train_dates
    if remaining < folds:
        raise ValueError("insufficient dates for the requested folds")
    size = remaining // folds
    output = []
    for fold in range(folds):
        test_start = min_train_dates + fold * size
        test_end = len(ordered) if fold == folds - 1 else test_start + size
        train = ordered[: max(0, test_start - embargo_periods)]
        test = ordered[test_start:test_end]
        if train and test:
            output.append((tuple(train), tuple(test)))
    return tuple(output)
