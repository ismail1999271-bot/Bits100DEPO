import dataclasses

import pytest

from bist_hunter.ai_research import AIHypothesis, ResearchResult, ValidationCriteria, evaluate_hypothesis
from bist_hunter.experiments import ExperimentTracker, experiment_id

GOOD = {"validation_expectancy": 0.002, "holdout_expectancy": 0.001, "holdout_trades": 50,
        "holdout_max_drawdown": 0.1, "train_period": ("2020-01-01", "2023-12-31"),
        "validation_period": ("2024-01-02", "2024-12-31"), "holdout_period": ("2025-01-02", "2025-12-31")}


def test_tracker_is_append_only_and_complete(tmp_path):
    tracker = ExperimentTracker(tmp_path / "e.jsonl")
    rec = tracker.new(dataset="bist_real_v1", features=["rsi_14"], parameters={"k": 5}, status="RUNNING")
    tracker.new(dataset="bist_real_v1", features=["rsi_14"], parameters={"k": 5}, status="COMPLETED",
                metrics={"sharpe": 1.2})
    history = tracker.history(rec.experiment_id)
    assert [h["status"] for h in history] == ["RUNNING", "COMPLETED"]
    for key in ("experiment_id", "timestamp", "dataset", "features", "parameters", "train_period",
                "validation_period", "holdout_period", "result", "metrics", "status"):
        assert key in history[0]
    assert tracker.latest()[rec.experiment_id]["metrics"] == {"sharpe": 1.2}
    assert experiment_id("d", ["a", "b"], {}) == experiment_id("d", ["b", "a"], {})
    with pytest.raises(ValueError):
        tracker.new(dataset="d", features=[], parameters={}, status="LIVE")


def test_ai_output_is_never_a_trading_signal(tmp_path):
    fields = {f.name for f in dataclasses.fields(ResearchResult)}
    assert not fields & {"side", "quantity", "order", "order_id", "price"}
    hyp = AIHypothesis("ollama:qwen2.5-coder", "feature", "relative volume > 3 precedes tavan", ("relative_volume",))
    tracker = ExperimentTracker(tmp_path / "e.jsonl")
    res = evaluate_hypothesis(hyp, dataset="d", provenance="REAL", quant_engine=lambda h: h,
                              backtest=lambda _: GOOD, tracker=tracker)
    assert res.status == "SUPPORTED" and res.is_trading_signal is False
    assert tracker.latest()[res.experiment_id]["holdout_period"] == ["2025-01-02", "2025-12-31"]


def test_ai_hypothesis_rejected_or_blocked(tmp_path):
    tracker = ExperimentTracker(tmp_path / "e.jsonl")
    hyp = AIHypothesis("claude", "hypothesis", "gap-ups continue")
    bad = dict(GOOD, holdout_expectancy=-0.001)
    assert evaluate_hypothesis(hyp, dataset="d", provenance="REAL", quant_engine=lambda h: h,
                               backtest=lambda _: bad, tracker=tracker).status == "REJECTED"
    blocked = evaluate_hypothesis(hyp, dataset="d", provenance="SYNTHETIC", quant_engine=lambda h: 1 / 0,
                                  backtest=lambda _: GOOD, tracker=tracker)
    assert blocked.status == "BLOCKED" and "REAL_DATA_REQUIRED" in blocked.reasons
    with pytest.raises(ValueError):
        AIHypothesis("x", "trade_now", "buy it")
    assert ValidationCriteria().require_real_data
