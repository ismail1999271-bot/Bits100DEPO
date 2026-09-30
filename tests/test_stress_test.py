import numpy as np
import pandas as pd

from bist_hunter.stress_test import Position, estimate_beta, stress_portfolio


def _rets(n=120, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.Series(rng.normal(0, 0.01, n))
    a = 1.5 * idx + rng.normal(0, 0.001, n)
    b = 0.5 * idx + rng.normal(0, 0.001, n)
    return idx, pd.DataFrame({"AAA": a, "BBB": b})


def test_beta_and_scenarios():
    idx, r = _rets()
    assert abs(estimate_beta(r["AAA"], idx) - 1.5) < 0.1
    rep = stress_portfolio([Position("AAA", 600, "Banka"), Position("BBB", 400, "Enerji")], r, idx)
    assert rep.status == "OK"
    s10 = rep.scenarios[0]
    assert s10.pnl_try < 0 and s10.covered_fraction == 1.0
    assert rep.scenarios[1].pnl_try < rep.scenarios[0].pnl_try
    assert "position_concentration" in rep.warnings


def test_missing_history_is_not_guessed():
    rep = stress_portfolio([Position("AAA", 100)], None, None)
    assert rep.status == "MISSING"
    assert all(s.pnl_try is None for s in rep.scenarios)
    assert "sector_unknown" in rep.warnings


def test_empty_positions():
    assert stress_portfolio([], None, None).status == "MISSING"
