"""Leakage-safe E2E backtest joining Tavan-DNA probabilities to Quant Score."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .quant_score import QuantScore, calculate_quant_score
from .quant_validation import ValidationMetrics, validate_returns
from .tavan_model import TavanLogisticModel, build_tavan_dataset


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


def _precision(actual: pd.Series, selected: pd.Series) -> float:
    actual_bool = actual.astype(bool)
    selected_bool = selected.astype(bool)
    true_positive = int((actual_bool & selected_bool).sum())
    return true_positive / max(1, int(selected_bool.sum()))


def _score_row(row: dict[str, object], tavan_probability: float) -> QuantScore:
    components = {"tavan_dna": float(tavan_probability) * 100.0}
    for name in (
        "auction",
        "technical",
        "kap",
        "news",
        "fund",
        "institutional",
        "broker",
        "smart_money",
    ):
        value = row.get(name)
        if value is not None and pd.notna(value):
            components[name] = float(value)
    return calculate_quant_score(components, min_coverage=0.20)


def run_quant_tavan_e2e(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    threshold: float = 70.0,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> E2EBacktestReport:
    """Fit Tavan-DNA chronologically, then rank OOS rows with Quant Score."""
    labeled = build_tavan_dataset(frame).sort_values("timestamp").reset_index(drop=True)
    if len(labeled) < 30:
        raise ValueError("at least 30 labeled observations are required")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("fractions must leave a final holdout")

    train_end = max(20, int(len(labeled) * train_fraction))
    validation_end = int(len(labeled) * (train_fraction + validation_fraction))
    validation_end = max(train_end + 1, min(validation_end, len(labeled) - 1))
    train = labeled.iloc[:train_end]
    validation = labeled.iloc[train_end:validation_end]
    holdout = labeled.iloc[validation_end:]
    if train["target"].nunique() < 2:
        raise ValueError("training data must contain both target classes")

    model = TavanLogisticModel().fit(train)

    def evaluate(test: pd.DataFrame) -> tuple[float, float, int]:
        probabilities = model.predict_proba(test)
        scores = pd.Series(
            [
                _score_row(row, probability).score
                for row, probability in zip(test.to_dict("records"), probabilities)
            ],
            index=test.index,
            dtype=float,
        )
        selected = scores >= threshold
        return float(scores.mean()), _precision(test["target"], selected), int(selected.sum())

    validation_mean, validation_precision, validation_selected = evaluate(validation)
    holdout_mean, holdout_precision, holdout_selected = evaluate(holdout)
    holdout_probabilities = model.predict_proba(holdout)
    holdout_scores = pd.Series(
        [
            _score_row(row, probability).score
            for row, probability in zip(holdout.to_dict("records"), holdout_probabilities)
        ],
        index=holdout.index,
        dtype=float,
    )
    holdout_selected_mask = holdout_scores >= threshold
    holdout_next_close = holdout.groupby("symbol")["close"].shift(-1)
    returns = (
        holdout_next_close / holdout["close"] - 1
    ).where(holdout_selected_mask & holdout_next_close.notna()).dropna()
    actual = holdout["target"].astype(bool)
    true_positive = int((actual & holdout_selected_mask).sum())
    recall = true_positive / max(1, int(actual.sum()))
    return E2EBacktestReport(
        len(train),
        len(validation),
        len(holdout),
        validation_selected,
        holdout_selected,
        round(validation_mean, 6),
        round(holdout_mean, 6),
        round(validation_precision, 6),
        round(holdout_precision, 6),
        round(recall, 6),
        validate_returns(returns.tolist(), cost_bps=cost_bps, slippage_bps=slippage_bps),
    )
