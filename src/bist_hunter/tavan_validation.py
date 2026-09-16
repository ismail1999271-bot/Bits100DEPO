"""Expanded Tavan-DNA walk-forward validation with an untouched final holdout."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .quant_validation import validate_returns
from .tavan_model import TavanLogisticModel, build_tavan_dataset


@dataclass(frozen=True, slots=True)
class TavanFold:
    fold: int
    train_rows: int
    test_rows: int
    precision: float
    recall: float
    selected_rows: int
    selected_return_metrics: object


@dataclass(frozen=True, slots=True)
class TavanValidationReport:
    folds: tuple[TavanFold, ...]
    final_holdout_rows: int
    final_holdout_precision: float
    final_holdout_recall: float
    final_holdout_selected: int
    final_holdout_returns: object


def _metrics(actual: pd.Series, probability, threshold: float):
    predicted = probability >= threshold
    tp = int((predicted & actual.astype(bool)).sum())
    selected = int(predicted.sum())
    positives = int(actual.sum())
    precision = tp / max(1, selected)
    recall = tp / max(1, positives)
    return precision, recall, predicted


def run_tavan_walk_forward(
    frame: pd.DataFrame,
    *,
    folds: int = 8,
    final_holdout_fraction: float = 0.15,
    threshold: float = 0.50,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> TavanValidationReport:
    if folds < 3:
        raise ValueError("at least three walk-forward folds are required")
    labeled = build_tavan_dataset(frame).sort_values("timestamp").reset_index(drop=True)
    if len(labeled) < folds * 10 + 30:
        raise ValueError("insufficient observations for expanded walk-forward validation")
    holdout_start = int(len(labeled) * (1.0 - final_holdout_fraction))
    holdout_start = max(folds * 5, min(holdout_start, len(labeled) - 10))
    development = labeled.iloc[:holdout_start]
    holdout = labeled.iloc[holdout_start:]
    edges = [int(i) for i in pd.Series(range(folds + 1)).map(lambda x: 0)]
    # Equal chronological test windows with an expanding training prefix.
    test_edges = [int(v) for v in __import__("numpy").linspace(max(20, len(development) // (folds + 1)), len(development), folds + 1)]
    fold_reports: list[TavanFold] = []
    for fold in range(folds):
        train_end = test_edges[fold]
        test_end = test_edges[fold + 1]
        train = development.iloc[:train_end]
        test = development.iloc[train_end:test_end]
        if len(test) == 0 or train["target"].nunique() < 2:
            continue
        model = TavanLogisticModel().fit(train)
        probability = model.predict_proba(test)
        precision, recall, selected = _metrics(test["target"], probability, threshold)
        next_close = test.groupby("symbol")["close"].shift(-1)
        returns = (next_close / test["close"] - 1).where(selected & next_close.notna()).dropna()
        fold_reports.append(TavanFold(fold + 1, len(train), len(test), round(precision, 6), round(recall, 6), int(selected.sum()), validate_returns(returns.tolist(), cost_bps=cost_bps, slippage_bps=slippage_bps)))
    if not fold_reports:
        raise ValueError("no valid walk-forward folds")
    # The final holdout is fit once using development data and is never used for model selection.
    if development["target"].nunique() < 2:
        raise ValueError("development data must contain both target classes")
    final_model = TavanLogisticModel().fit(development)
    final_probability = final_model.predict_proba(holdout)
    precision, recall, selected = _metrics(holdout["target"], final_probability, threshold)
    next_close = holdout.groupby("symbol")["close"].shift(-1)
    returns = (next_close / holdout["close"] - 1).where(selected & next_close.notna()).dropna()
    return TavanValidationReport(tuple(fold_reports), len(holdout), round(precision, 6), round(recall, 6), int(selected.sum()), validate_returns(returns.tolist(), cost_bps=cost_bps, slippage_bps=slippage_bps))
