import pandas as pd

from bist_hunter.synthetic import make_dataset
from bist_hunter.tavan_validation import PROBABILITY_GRID, report_to_rows, run_tavan_walk_forward


def _frame(days=160):
    return pd.DataFrame(make_dataset(symbols=("AAA", "BBB", "CCC", "DDD"), days=days))


def test_folds_are_expanding_and_before_holdout():
    report = run_tavan_walk_forward(_frame(), folds=4)
    assert len(report.folds) >= 3
    train_sizes = [f.train_rows for f in report.folds]
    assert train_sizes == sorted(train_sizes)
    for fold in report.folds:
        assert fold.train_period[1] < fold.test_period[0]
        assert fold.test_period[1] < report.holdout_period[0]
    assert report.development_period[1] < report.holdout_period[0]


def test_every_fold_has_required_metrics():
    report = run_tavan_walk_forward(_frame(), folds=4)
    rows = report_to_rows(report)
    assert rows[-1]["segment"] == "final_holdout"
    required = {"precision", "recall", "hit_rate", "expectancy", "max_drawdown", "sharpe",
                "sortino", "profit_factor", "sample_size"}
    for row in rows:
        assert required <= set(row)
        assert 0.0 <= row["precision"] <= 1.0


def test_threshold_tuning_ignores_final_holdout():
    base = _frame()
    tuned = run_tavan_walk_forward(base, folds=4, threshold=None)
    assert tuned.threshold in PROBABILITY_GRID or tuned.threshold == 0.5
    tampered = base.copy()
    future = tampered["timestamp"] >= pd.Timestamp(tuned.holdout_period[0])
    tampered.loc[future, ["open", "high", "low", "close"]] *= 2.5
    tuned_again = run_tavan_walk_forward(tampered, folds=4, threshold=None)
    assert tuned_again.threshold == tuned.threshold
    assert [f.precision for f in tuned_again.folds] == [f.precision for f in tuned.folds]
