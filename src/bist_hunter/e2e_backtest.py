"""Leakage-safe E2E backtest joining Tavan-DNA probabilities to Quant Score.

Chronology: Train -> (embargo) -> Validation -> (embargo) -> Final Holdout.

* Splits are made on distinct timestamps, so one trading day never lands in
  two segments (important for multi-symbol panels).
* The Tavan target looks one bar ahead, so the last date before each
  boundary is purged (``embargo_periods``).
* Tavan-DNA is fitted on train only. The optional Quant-Score threshold search
  uses validation only. The final holdout is evaluated exactly once.
* Non-Tavan components come only from point-in-time columns already present in
  the frame; missing values are left missing (never zero-filled).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .quant_score import COMPONENTS, QuantScore, calculate_quant_score
from .quant_validation import ValidationMetrics, validate_returns
from .tavan_model import TavanLogisticModel, build_tavan_dataset
from .walk_forward import purged_date_split

THRESHOLD_GRID = (50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0)
EXTERNAL_COMPONENTS = tuple(name for name in COMPONENTS if name != "tavan_dna")


@dataclass(frozen=True, slots=True)
class E2EBacktestReport:
    train_rows: int
    validation_rows: int
    holdout_rows: int
    validation_selected: int
    holdout_selected: int
    validation_quant_score: float
    holdout_quant_score: float
    validation_precision: float
    holdout_precision: float
    holdout_recall: float
    holdout_returns: ValidationMetrics
    threshold: float = 70.0
    purged_rows: int = 0
    train_period: tuple[str, str] = ("", "")
    validation_period: tuple[str, str] = ("", "")
    holdout_period: tuple[str, str] = ("", "")
    holdout_mean_coverage: float = 0.0
    holdout_mean_confidence: float = 0.0
    holdout_blocked: int = 0
    validation_returns: ValidationMetrics | None = None


def _precision(actual: pd.Series, selected: pd.Series) -> float:
    actual_bool = actual.astype(bool)
    selected_bool = selected.astype(bool)
    true_positive = int((actual_bool & selected_bool).sum())
    return true_positive / max(1, int(selected_bool.sum()))


def _recall(actual: pd.Series, selected: pd.Series) -> float:
    actual_bool = actual.astype(bool)
    true_positive = int((actual_bool & selected.astype(bool)).sum())
    return true_positive / max(1, int(actual_bool.sum()))


def _score_row(row: dict[str, object], tavan_probability: float) -> QuantScore:
    components: dict[str, float] = {"tavan_dna": float(tavan_probability) * 100.0}
    for name in EXTERNAL_COMPONENTS:
        value = row.get(name)
        if value is not None and pd.notna(value):
            components[name] = float(value)
    return calculate_quant_score(components, min_coverage=0.20)


def _period(frame: pd.DataFrame) -> tuple[str, str]:
    if frame.empty:
        return ("", "")
    return (str(frame["timestamp"].min()), str(frame["timestamp"].max()))


def _score_frame(model: TavanLogisticModel, frame: pd.DataFrame) -> list[QuantScore]:
    probabilities = model.predict_proba(frame)
    return [_score_row(row, p) for row, p in zip(frame.to_dict("records"), probabilities)]


def _next_bar_returns(frame: pd.DataFrame, selected: pd.Series) -> list[float]:
    """Close-to-next-close return of selected rows (entry at signal close)."""
    next_close = frame.groupby("symbol")["close"].shift(-1)
    returns = (next_close / frame["close"] - 1).where(selected & next_close.notna()).dropna()
    return returns.tolist()


def _choose_threshold(
    frame: pd.DataFrame, scores: pd.Series, cost_bps: float, slippage_bps: float, min_trades: int
) -> float | None:
    best: tuple[float, float] | None = None
    for candidate in THRESHOLD_GRID:
        metrics = validate_returns(
            _next_bar_returns(frame, scores >= candidate), cost_bps=cost_bps, slippage_bps=slippage_bps
        )
        if metrics.sample_size < min_trades:
            continue
        if best is None or metrics.expectancy > best[1]:
            best = (candidate, metrics.expectancy)
    return None if best is None else best[0]


def run_quant_tavan_e2e(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    threshold: float | None = 70.0,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    embargo_periods: int = 1,
    min_validation_trades: int = 5,
) -> E2EBacktestReport:
    """Fit Tavan-DNA chronologically, then rank OOS rows with Quant Score.

    ``threshold=None`` selects the Quant-Score cut-off on validation only.
    """
    labeled = build_tavan_dataset(frame).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if len(labeled) < 30:
        raise ValueError("at least 30 labeled observations are required")
    split = purged_date_split(
        labeled["timestamp"],
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        embargo_periods=embargo_periods,
    )
    train = labeled[labeled["timestamp"].isin(split.train_dates)]
    validation = labeled[labeled["timestamp"].isin(split.validation_dates)]
    holdout = labeled[labeled["timestamp"].isin(split.holdout_dates)]
    purged_rows = int(labeled["timestamp"].isin(split.purged_dates).sum())

    # Hard leakage invariants.
    if not (train["timestamp"].max() < validation["timestamp"].min()
            and validation["timestamp"].max() < holdout["timestamp"].min()):
        raise AssertionError("chronological split violated")
    if train["target"].nunique() < 2:
        raise ValueError("training data must contain both target classes")

    model = TavanLogisticModel().fit(train)

    validation_q = _score_frame(model, validation)
    validation_scores = pd.Series([q.score for q in validation_q], index=validation.index, dtype=float)
    if threshold is None:
        chosen = _choose_threshold(validation, validation_scores, cost_bps, slippage_bps, min_validation_trades)
        threshold = 70.0 if chosen is None else chosen
    validation_selected = validation_scores >= threshold

    holdout_q = _score_frame(model, holdout)
    holdout_scores = pd.Series([q.score for q in holdout_q], index=holdout.index, dtype=float)
    holdout_selected = holdout_scores >= threshold

    return E2EBacktestReport(
        train_rows=len(train),
        validation_rows=len(validation),
        holdout_rows=len(holdout),
        validation_selected=int(validation_selected.sum()),
        holdout_selected=int(holdout_selected.sum()),
        validation_quant_score=round(float(validation_scores.mean()), 6),
        holdout_quant_score=round(float(holdout_scores.mean()), 6),
        validation_precision=round(_precision(validation["target"], validation_selected), 6),
        holdout_precision=round(_precision(holdout["target"], holdout_selected), 6),
        holdout_recall=round(_recall(holdout["target"], holdout_selected), 6),
        holdout_returns=validate_returns(
            _next_bar_returns(holdout, holdout_selected), cost_bps=cost_bps, slippage_bps=slippage_bps
        ),
        threshold=float(threshold),
        purged_rows=purged_rows,
        train_period=_period(train),
        validation_period=_period(validation),
        holdout_period=_period(holdout),
        holdout_mean_coverage=round(sum(q.coverage for q in holdout_q) / max(1, len(holdout_q)), 4),
        holdout_mean_confidence=round(sum(q.confidence for q in holdout_q) / max(1, len(holdout_q)), 4),
        holdout_blocked=sum(q.status.startswith("BLOCKED") for q in holdout_q),
        validation_returns=validate_returns(
            _next_bar_returns(validation, validation_selected), cost_bps=cost_bps, slippage_bps=slippage_bps
        ),
    )
