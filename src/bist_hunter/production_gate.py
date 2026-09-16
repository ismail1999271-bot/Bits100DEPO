"""Fail-closed production readiness gate."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GateResult:
    ready: bool
    blockers: tuple[str, ...]


def evaluate_production_gate(*, data_health: bool, walk_forward: bool, paper_trading: bool, final_holdout: bool, min_sample: bool, drawdown_ok: bool, provider_health: bool, data_integrity: bool) -> GateResult:
    checks = {
        "DATA_HEALTH": data_health,
        "WALK_FORWARD": walk_forward,
        "PAPER_TRADING": paper_trading,
        "FINAL_HOLDOUT": final_holdout,
        "MIN_SAMPLE": min_sample,
        "DRAWDOWN_LIMIT": drawdown_ok,
        "PROVIDER_HEALTH": provider_health,
        "DATA_INTEGRITY": data_integrity,
    }
    blockers = tuple(name for name, passed in checks.items() if not passed)
    return GateResult(not blockers, blockers)
