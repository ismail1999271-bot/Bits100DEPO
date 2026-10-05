from bist_hunter.scenarios import build_scenarios, format_scenarios

UP = {"close": 110, "ema_20": 105, "sma_50": 100, "macd_hist": 0.8, "rsi_14": 62, "atr_14": 2.0,
      "support_20": 104, "resistance_20": 112, "relative_volume": 1.8, "adx_14": 28}
WEEK_UP = {"close": 110, "ema_20": 100, "sma_50": 95, "macd_hist": 1.0, "rsi_14": 60, "support_20": 98,
           "resistance_20": 120}


def test_bull_scenario_has_levels_and_alignment():
    res = build_scenarios("thyao", UP, WEEK_UP)
    assert res.status == "OK" and res.bias == "BULLISH" and res.alignment == "ALIGNED_UP"
    bull, neutral, bear = res.scenarios
    assert any("112.00" in c for c in bull.confirms_if)
    assert any("107.00" in c for c in bull.invalidated_if)  # max(110-3, support 104)
    assert any("98.00" in c for c in bull.invalidated_if)
    assert bear.reference_level == 104 and bull.reference_level == 112
    assert any("104.00–112.00" in c for c in neutral.confirms_if)
    assert "garanti değildir" in format_scenarios(res)


def test_bear_and_mixed():
    down = {"close": 90, "ema_20": 95, "sma_50": 100, "macd_hist": -1, "rsi_14": 38, "atr_14": 2}
    assert build_scenarios("A", down, None).bias == "BEARISH"
    assert build_scenarios("A", down, None).alignment == "UNKNOWN"
    assert build_scenarios("A", UP, {**WEEK_UP, "close": 90, "macd_hist": -1, "rsi_14": 30}).alignment == "MIXED"


def test_missing_data_blocks_and_omits_levels():
    assert build_scenarios("A", {"close": 10}).status == "BLOCKED_MISSING_DATA"
    partial = build_scenarios("A", {"close": 10, "ema_20": 9, "macd_hist": 1, "rsi_14": 60})
    assert partial.status == "OK"
    assert partial.scenarios[0].invalidated_if == ()  # no ATR/support -> no invented stop
