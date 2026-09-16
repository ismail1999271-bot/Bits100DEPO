import pandas as pd

from bist_hunter.e2e_backtest import run_quant_tavan_e2e
from bist_hunter.synthetic import make_dataset


def test_quant_tavan_e2e_uses_oos_validation_and_holdout():
    frame = pd.DataFrame(make_dataset(days=60))
    frame["technical"] = 75.0
    report = run_quant_tavan_e2e(frame)

    assert report.train_rows > 0
    assert report.validation_rows > 0
    assert report.holdout_rows > 0
    assert report.train_rows + report.validation_rows + report.holdout_rows == len(frame) - len(frame["symbol"].unique())
    assert 0.0 <= report.validation_quant_score <= 100.0
    assert 0.0 <= report.holdout_quant_score <= 100.0
    assert 0.0 <= report.holdout_precision <= 1.0
    assert 0.0 <= report.holdout_recall <= 1.0


def test_quant_tavan_e2e_does_not_fill_missing_components_with_zero():
    frame = pd.DataFrame(make_dataset(days=50))
    report = run_quant_tavan_e2e(frame, threshold=70.0)

    assert report.holdout_quant_score > 0.0
    assert report.holdout_selected >= 0
