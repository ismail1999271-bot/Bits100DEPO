"""Expanded Tavan-DNA walk-forward validation with an untouched final holdout.

* Development and final holdout are split on distinct dates with an embargo,
  because the Tavan label looks one bar ahead.
* Walk-forward folds are expanding windows over development dates only, each
  with the same embargo between train and test.
* The probability threshold is fixed by the caller or chosen from the
  development folds only. The final holdout is scored exactly once and is
  never used for tuning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .quant_validation import ValidationMetrics, validate_returns
from .tavan_model import TavanLogisticModel, build_tavan_dataset
from .walk_forward import expanding_date_folds

PROBABILITY_GRID = (0.30, 0.40, 0.50, 0.60, 0.70)


@dataclass(frozen=True, slots=True)
class TavanFold:
    fold: int
    train_rows: int
    test_rows: int
    precision: float
    recall: float
    selected_rows: int
    selected_return_metrics: ValidationMetrics
    train_period: tuple[str, str] = ("", "")
    test_period: tuple[str, str] = ("", "")
    positives: int = 0


@dataclass(frozen=True, slots=True)
class TavanValidationReport:
    folds: tuple[TavanFold, ...]
    final_holdout_rows: int
    final_holdout_precision: float
    final_holdout_recall: float
    final_holdout_selected: int
    final_holdout_returns: ValidationMetrics
    threshold: float = 0.50
    development_period: tuple[str, str] = ("", "")
    holdout_period: tuple[str, str] = ("", "")
    mean_fold_precision: float = 0.0
    mean_fold_expectancy: float = 0.0


def _metrics(actual: pd.Series, probability: np.ndarray, threshold: float):
    predicted = probability >= threshold
    actual_bool = actual.astype(bool).to_numpy()
    tp = int((predicted & actual_bool).sum())
    selected = int(predicted.sum())
    positives = int(actual_bool.sum())
    precision = tp / max(1, selected)
    recall = tp / max(1, positives)
    return precision, recall, predicted


def _period(frame: pd.DataFrame) -> tuple[str, str]:
    if frame.empty:
        return ("", "")
    return (str(frame["timestamp"].min()), str(frame["timestamp"].max()))


def _returns(frame: pd.DataFrame, selected: np.ndarray) -> list[float]:
    mask = pd.Series(selected, index=frame.index)
    next_close = frame.groupby("symbol")["close"].shift(-1)
    return (next_close / frame["close"] - 1).where(mask & next_close.notna()).dropna().tolist()


def _run_folds(development, fold_dates, threshold, cost_bps, slippage_bps):
    reports: list[TavanFold] = []
    for number, (train_dates, test_dates) in enumerate(fold_dates, start=1):
        train = development[development["timestamp"].isin(train_dates)]
        test = development[development["timestamp"].isin(test_dates)]
        if test.empty or train["target"].nunique() < 2:
            continue
        probability = TavanLogisticModel().fit(train).predict_proba(test)
        precision, recall, selected = _metrics(test["target"], probability, threshold)
        reports.append(
            TavanFold(
                number, len(train), len(test), round(precision, 6), round(recall, 6),
                int(selected.sum()),
                validate_returns(_returns(test, selected), cost_bps=cost_bps, slippage_bps=slippage_bps),
                _period(train), _period(test), int(test["target"].sum()),
            )
        )
    return reports


def run_tavan_walk_forward(
    frame: pd.DataFrame,
    *,
    folds: int = 8,
    final_holdout_fraction: float = 0.15,
    threshold: float | None = 0.50,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    embargo_periods: int = 1,
) -> TavanValidationReport:
    """Run expanding walk-forward on development data, then one final holdout.

    ``threshold=None`` picks the probability cut-off that maximizes mean fold
    expectancy on development folds only.
    """
    if folds < 3 or not 0.05 <= final_holdout_fraction < 0.40:
        raise ValueError("use at least 3 folds and a 5%-40% final holdout")
    labeled = build_tavan_dataset(frame).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if len(labeled) < folds * 10 + 30:
        raise ValueError("insufficient observations for expanded walk-forward validation")
    dates = sorted(labeled["timestamp"].unique())
    holdout_start = int(len(dates) * (1.0 - final_holdout_fraction))
    holdout_start = max(folds + 2 + embargo_periods, min(holdout_start, len(dates) - 2))
    development_dates = dates[: holdout_start - embargo_periods]
    holdout_dates = dates[holdout_start:]
    development = labeled[labeled["timestamp"].isin(development_dates)]
    holdout = labeled[labeled["timestamp"].isin(holdout_dates)]
    if development["target"].nunique() < 2:
        raise ValueError("development data must contain both target classes")
    min_train = max(5, len(development_dates) // (folds + 1))
    fold_dates = expanding_date_folds(
        development_dates, folds=folds, min_train_dates=min_train, embargo_periods=embargo_periods
    )

    if threshold is None:
        best: tuple[float, float] | None = None
        for candidate in PROBABILITY_GRID:
            candidate_reports = _run_folds(development, fold_dates, candidate, cost_bps, slippage_bps)
            traded = [r.selected_return_metrics.expectancy for r in candidate_reports
                      if r.selected_return_metrics.sample_size]
            if not traded:
                continue
            value = float(np.mean(traded))
            if best is None or value > best[1]:
                best = (candidate, value)
        threshold = 0.50 if best is None else best[0]

    fold_reports = _run_folds(development, fold_dates, threshold, cost_bps, slippage_bps)
    if len(fold_reports) < 3:
        raise ValueError("fewer than three valid walk-forward folds")

    # Final holdout: one fit on all development data, one evaluation, no tuning.
    final_model = TavanLogisticModel().fit(development)
    precision, recall, selected = _metrics(holdout["target"], final_model.predict_proba(holdout), threshold)
    holdout_metrics = validate_returns(_returns(holdout, selected), cost_bps=cost_bps, slippage_bps=slippage_bps)
    traded = [r for r in fold_reports if r.selected_return_metrics.sample_size]
    return TavanValidationReport(
        tuple(fold_reports),
        len(holdout),
        round(precision, 6),
        round(recall, 6),
        int(selected.sum()),
        holdout_metrics,
        float(threshold),
        _period(development),
        _period(holdout),
        round(float(np.mean([r.precision for r in fold_reports])), 6),
        round(float(np.mean([r.selected_return_metrics.expectancy for r in traded])) if traded else 0.0, 8),
    )


def report_to_rows(report: TavanValidationReport) -> list[dict[str, object]]:
    """Flatten a report into table rows (one per fold + final holdout)."""
    rows: list[dict[str, object]] = []

    def row(name, period, precision, recall, selected, metrics: ValidationMetrics):
        return {
            "segment": name, "period": " -> ".join(period), "precision": precision, "recall": recall,
            "selected": selected, "sample_size": metrics.sample_size, "hit_rate": metrics.hit_rate,
            "expectancy": metrics.expectancy, "max_drawdown": metrics.max_drawdown,
            "sharpe": metrics.sharpe, "sortino": metrics.sortino,
            "profit_factor": metrics.profit_factor, "total_return": metrics.total_return,
        }

    for fold in report.folds:
        rows.append(row(f"fold_{fold.fold}", fold.test_period, fold.precision, fold.recall,
                        fold.selected_rows, fold.selected_return_metrics))
    rows.append(row("final_holdout", report.holdout_period, report.final_holdout_precision,
                    report.final_holdout_recall, report.final_holdout_selected, report.final_holdout_returns))
    return rows
