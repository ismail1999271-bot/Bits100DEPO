"""Executable Scout/Quant/Audit -> backtest research loop."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from .quant_researcher import ExperimentSpec
from .research_orchestrator import ResearchLoop


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    experiment_id: str
    accepted: bool
    metrics: dict[str, float]
    audit_passed: bool


class ExecutableResearchLoop:
    """Turn approved experiment specs into chronological backtests."""

    def __init__(self, orchestrator: ResearchLoop | None = None) -> None:
        self.orchestrator = orchestrator or ResearchLoop()

    def run(
        self,
        experiment: ExperimentSpec,
        frame: pd.DataFrame,
        backtest: Callable[[ExperimentSpec, pd.DataFrame], dict[str, float]],
    ) -> ExperimentResult:
        audit = self.orchestrator.deterministic_audit(experiment)
        if not audit.passed:
            return ExperimentResult(experiment.experiment_id, False, {}, False)
        metrics = backtest(experiment, frame)
        accepted = self._accept(experiment, metrics)
        return ExperimentResult(experiment.experiment_id, accepted, metrics, True)

    @staticmethod
    def _accept(experiment: ExperimentSpec, metrics: dict[str, float]) -> bool:
        criteria = experiment.acceptance_criteria.lower()
        precision = metrics.get("top_k_precision", 0.0)
        pnl = metrics.get("net_pnl", 0.0)
        if "precision" in criteria and precision <= 0.0:
            return False
        if "positive pnl" in criteria and pnl <= 0.0:
            return False
        return bool(metrics)
