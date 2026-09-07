import pandas as pd
import pytest

from bist_hunter.technical_watchlist import add_risk_reward, merge_technical_watchlist, validate_watchlist


def sample():
    return pd.DataFrame([{
        "symbol": "ISMEN",
        "last_price": 33.22,
        "entry_low": 32.62,
        "entry_high": 33.50,
        "target": 36.40,
        "stop": 32.00,
    }])


def test_risk_reward_and_percentages():
    result = add_risk_reward(sample()).iloc[0]
    assert result["target_upside_pct"] > 0
    assert result["stop_downside_pct"] > 0
    assert result["risk_reward"] > 1
    assert 0 <= result["technical_watch_score"] <= 100


def test_invalid_levels_are_rejected():
    frame = sample()
    frame.loc[0, "stop"] = 33.0
    with pytest.raises(ValueError, match="invalid watchlist levels"):
        validate_watchlist(frame)


def test_merge_attaches_features_without_changing_unmatched_rows():
    base = pd.DataFrame([{"symbol": "ISMEN", "score": 80}, {"symbol": "AAA", "score": 70}])
    result = merge_technical_watchlist(base, sample())
    assert result.loc[result["symbol"] == "ISMEN", "risk_reward"].notna().all()
    assert result.loc[result["symbol"] == "AAA", "risk_reward"].isna().all()
