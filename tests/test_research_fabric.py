from bist_hunter.agents.quant_researcher import ExperimentSpec
from bist_hunter.agents.signal_auditor import SignalAuditor
from bist_hunter.research_fabric import DataSourceSpec, build_simulation_grid, summarize_coverage


def valid_experiment() -> ExperimentSpec:
    return ExperimentSpec(
        hypothesis="Institutional flow plus event surprise improves Top-K precision.",
        target="next-session +5% event within 1 trading day",
        features=("smart_money", "kap_event", "volume_anomaly"),
        baseline="current_signal_engine",
        validation="chronological walk-forward with final untouched holdout",
        acceptance_criteria=(
            "out-of-sample Top-K precision improves >= 5pp",
            "track multiple-testing across experiment count",
        ),
        cost_model="commission + spread + slippage",
        liquidity_filter="minimum traded value and concentration limits",
        leakage_controls=("point-in-time timestamps; reject future/look-ahead leakage",),
        experiment_id="exp-001",
    )


def test_signal_auditor_accepts_research_contract() -> None:
    report = SignalAuditor().audit(valid_experiment())
    assert report.passed
    assert not report.blocking_findings


def test_signal_auditor_blocks_missing_holdout() -> None:
    experiment = valid_experiment()
    broken = ExperimentSpec(
        hypothesis=experiment.hypothesis,
        target=experiment.target,
        features=experiment.features,
        baseline=experiment.baseline,
        validation="chronological walk-forward only",
        acceptance_criteria=experiment.acceptance_criteria,
        cost_model=experiment.cost_model,
        liquidity_filter=experiment.liquidity_filter,
        leakage_controls=experiment.leakage_controls,
    )
    report = SignalAuditor().audit(broken)
    assert not report.passed
    assert any(item.check == "holdout" for item in report.blocking_findings)


def test_simulation_grid_is_bounded() -> None:
    configs = build_simulation_grid(
        ["baseline", "model_a"], ["core", "core_plus_alt"], [1, 2], [0.7, 0.8], [10.0, 20.0], [1, 2], max_runs=10
    )
    assert len(configs) == 10


def test_data_coverage_separates_discovered_from_live() -> None:
    coverage = summarize_coverage(
        [
            DataSourceSpec("ohlcv", "market", "provider-a", point_in_time=True, enabled=True),
            DataSourceSpec("news", "news", "provider-b", point_in_time=False, enabled=False),
        ]
    )
    assert coverage.discovered == 2
    assert coverage.enabled == 1
    assert coverage.point_in_time_ready == 1
