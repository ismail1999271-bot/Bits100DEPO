import numpy as np
import pandas as pd

from bist_hunter.overfit import (
    deflated_sharpe, label_regimes, overfit_verdict, parameter_sensitivity, regime_performance,
)


def test_dsr_more_trials_lowers_probability():
    one = deflated_sharpe(0.1, 250, 1)
    many = deflated_sharpe(0.1, 250, 200)
    assert one > many
    assert deflated_sharpe(0.1, 10, 1) is None


def test_regime_performance_flags_small_sample():
    trades = pd.DataFrame({"date": [pd.Timestamp("2024-01-0%d" % d) for d in range(1, 5)],
                           "status": ["FILLED"] * 4, "net_return": [0.02, -0.01, 0.03, -0.02]})
    regimes = pd.Series("BULL", index=trades["date"])
    out = regime_performance(trades, regimes)
    assert out.iloc[0]["regime"] == "BULL" and out.iloc[0]["flag"] == "INSUFFICIENT_SAMPLE"
    assert out.iloc[0]["win_rate"] == 0.5


def test_label_regimes_point_in_time():
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 300)), index=pd.date_range("2023-01-01", periods=300))
    full = label_regimes(close)
    part = label_regimes(close.iloc[:200])
    assert full.iloc[:200].fillna("x").tolist() == part.fillna("x").tolist()


def test_sensitivity_and_verdict():
    sens = parameter_sensitivity(lambda k: 1.0 if k == 5 else -1.0, {"k": [3, 5, 7]})
    assert sens["spread_min_minus_max"].iloc[0] == -2.0
    v = overfit_verdict(0.5, sens, None, 1.0, 0.1)
    assert v["status"] == "SUSPECT"
    assert {"DSR_BELOW_95", "HOLDOUT_DEGRADATION", "PARAMETER_SENSITIVE"} <= set(v["flags"])
