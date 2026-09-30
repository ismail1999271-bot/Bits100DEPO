import math

import pandas as pd
import pytest

from bist_hunter.e2e_backtest import THRESHOLD_GRID, run_quant_tavan_e2e
from bist_hunter.quant_score import calculate_quant_score
from bist_hunter.synthetic import make_dataset
from bist_hunter.walk_forward import expanding_date_folds, purged_date_split


def test_missing_components_are_not_zero_filled():
    only_strong = calculate_quant_score({"auction": 90, "tavan_dna": 80})
    with_none = calculate_quant_score({"auction": 90, "tavan_dna": 80, "kap": None, "news": math.nan})
    assert only_strong.score == with_none.score == 85.56
    assert with_none.coverage == 0.45
    assert set(with_none.missing) >= {"kap", "news", "fund"}


def test_confidence_discounts_quality_but_coverage_does_not():
    result = calculate_quant_score({"auction": 90, "tavan_dna": 80}, quality={"auction": 0.5})
    assert result.coverage == 0.45
    assert result.confidence == pytest.approx(0.25 * 0.5 + 0.20)
    assert result.component("auction") == 90


def test_unknown_component_is_rejected():
    with pytest.raises(ValueError):
        calculate_quant_score({"auction": 90, "made_up": 50})


def test_purged_date_split_has_embargo_and_no_overlap():
    dates = pd.date_range("2025-01-01", periods=50, freq="D")
    panel = list(dates) * 3  # three symbols share every date
    split = purged_date_split(panel, embargo_periods=1)
    assert max(split.train_dates) < min(split.validation_dates)
    assert max(split.validation_dates) < min(split.holdout_dates)
    assert len(split.purged_dates) == 2
    all_used = set(split.train_dates) | set(split.validation_dates) | set(split.holdout_dates)
    assert not all_used & set(split.purged_dates)


def test_expanding_folds_are_chronological_with_gap():
    dates = pd.date_range("2025-01-01", periods=40, freq="D")
    folds = expanding_date_folds(dates, folds=4, min_train_dates=20, embargo_periods=1)
    assert len(folds) == 4
    for train, test in folds:
        assert max(train) < min(test)
        assert (min(test) - max(train)).days == 2  # one embargoed date between
    assert len(folds[1][0]) > len(folds[0][0])


def test_e2e_reports_periods_coverage_and_purge():
    frame = pd.DataFrame(make_dataset(days=80))
    frame["technical"] = 75.0
    report = run_quant_tavan_e2e(frame)
    assert report.train_period[1] < report.validation_period[0] < report.holdout_period[0]
    assert report.purged_rows > 0
    assert report.holdout_mean_coverage == pytest.approx(0.30)
    assert report.validation_returns is not None


def test_e2e_threshold_is_chosen_on_validation_only():
    frame = pd.DataFrame(make_dataset(days=80))
    report = run_quant_tavan_e2e(frame, threshold=None, min_validation_trades=1)
    assert report.threshold in THRESHOLD_GRID or report.threshold == 70.0


def test_e2e_holdout_changes_do_not_affect_train_or_validation():
    base = pd.DataFrame(make_dataset(days=80))
    report_a = run_quant_tavan_e2e(base)
    tampered = base.copy()
    cutoff = pd.Timestamp(report_a.holdout_period[0])
    future = tampered["timestamp"] > cutoff
    tampered.loc[future, ["open", "high", "low", "close"]] *= 3.0
    report_b = run_quant_tavan_e2e(tampered)
    assert report_a.validation_quant_score == report_b.validation_quant_score
    assert report_a.validation_precision == report_b.validation_precision
    assert report_a.train_rows == report_b.train_rows
