from bist_hunter.performance_learning import build_learning_report, research_actions
from bist_hunter.research_memory import ResearchMemory, SignalAttribution


def test_learning_joins_signal_attribution_to_outcomes(tmp_path):
    memory = ResearchMemory(tmp_path / "memory.jsonl")
    for i in range(30):
        memory.record_signal(SignalAttribution(
            signal_id=f"s{i}", symbol="AAA", timestamp=f"2026-01-{i + 1:02d}",
            score=80.0, outcome_return=None, hit=None,
            feature_contributions={"volume_anomaly": 20.0, "flow_strength": 10.0},
        ))
        memory.close_signal(f"s{i}", 0.12 if i % 2 == 0 else 0.04)
    report = build_learning_report(memory)
    assert report.closed_signals == 30
    assert report.hit_rate == 0.5
    assert report.recommended_action == "READY_FOR_RESEARCH"
    assert "volume_anomaly" in report.feature_edge
    assert research_actions(report)
