from bist_hunter.agents.technology_scout import TechnologyScout, build_scout_prompt


def test_scout_has_research_safety_rules() -> None:
    rules = TechnologyScout().research_rules()
    assert any("leakage" in rule.lower() for rule in rules)
    assert any("backtest" in rule.lower() for rule in rules)


def test_prompt_contains_bist_specific_decision_contract() -> None:
    prompt = build_scout_prompt(
        "BIST pre-limit-up ranking with news, smart money and walk-forward backtests.",
        "alternative data; agentic AI; market microstructure",
    )
    assert "BIST" in prompt
    assert "ADOPT, EXPERIMENT, WATCH, or REJECT" in prompt
    assert "look-ahead leakage" in prompt
    assert "out-of-sample" in prompt
