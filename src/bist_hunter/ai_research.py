"""AI research assistant flow: Hypothesis -> Quant Engine -> Backtest -> Validation -> Research Result.

AI (Claude, Astra, Ollama or any other model) may propose patterns,
hypotheses, features, backtest designs, anomaly explanations or code changes.
Its output is NEVER a trading signal: :class:`ResearchResult` has no order,
side or quantity field, ``is_trading_signal`` is always False, and acceptance
depends only on deterministic validation metrics (holdout included, never
used for tuning).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .experiments import ExperimentTracker

ALLOWED_KINDS = ("pattern", "hypothesis", "feature", "backtest_design", "anomaly_explanation", "code_change")


@dataclass(frozen=True, slots=True)
class AIHypothesis:
    source: str  # e.g. "claude", "astra", "ollama:qwen2.5-coder"
    kind: str
    statement: str
    features: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ALLOWED_KINDS:
            raise ValueError(f"kind must be one of {ALLOWED_KINDS}")
        if not self.statement.strip():
            raise ValueError("empty hypothesis")


@dataclass(frozen=True, slots=True)
class ValidationCriteria:
    min_holdout_trades: int = 30
    min_holdout_expectancy: float = 0.0
    min_validation_expectancy: float = 0.0
    max_holdout_drawdown: float = 0.25
    require_real_data: bool = True


@dataclass(frozen=True, slots=True)
class ResearchResult:
    experiment_id: str
    hypothesis: AIHypothesis
    status: str  # SUPPORTED / REJECTED / BLOCKED
    reasons: tuple[str, ...]
    metrics: dict[str, Any]
    is_trading_signal: bool = False


def evaluate_hypothesis(
    hypothesis: AIHypothesis,
    *,
    dataset: str,
    provenance: str,
    quant_engine: Callable[[AIHypothesis], Any],
    backtest: Callable[[Any], dict[str, Any]],
    criteria: ValidationCriteria = ValidationCriteria(),
    tracker: ExperimentTracker | None = None,
) -> ResearchResult:
    """Run the deterministic pipeline. ``backtest`` must return a dict with keys
    validation_expectancy, holdout_expectancy, holdout_trades, holdout_max_drawdown and
    optionally train/validation/holdout periods."""
    tracker = tracker or ExperimentTracker()
    params = {"source": hypothesis.source, "kind": hypothesis.kind, **hypothesis.parameters}
    reasons: list[str] = []
    if criteria.require_real_data and provenance.upper() != "REAL":
        record = tracker.new(dataset=dataset, features=hypothesis.features, parameters=params,
                             result="blocked: real historical data required", status="BLOCKED")
        return ResearchResult(record.experiment_id, hypothesis, "BLOCKED", ("REAL_DATA_REQUIRED",), {})
    engine_output = quant_engine(hypothesis)
    metrics = dict(backtest(engine_output))
    if metrics.get("holdout_trades", 0) < criteria.min_holdout_trades:
        reasons.append("insufficient_holdout_trades")
    if metrics.get("validation_expectancy", float("-inf")) <= criteria.min_validation_expectancy:
        reasons.append("validation_expectancy_not_positive")
    if metrics.get("holdout_expectancy", float("-inf")) <= criteria.min_holdout_expectancy:
        reasons.append("holdout_expectancy_not_positive")
    if metrics.get("holdout_max_drawdown", 1.0) > criteria.max_holdout_drawdown:
        reasons.append("holdout_drawdown_too_large")
    status = "REJECTED" if reasons else "SUPPORTED"
    record = tracker.new(
        dataset=dataset, features=hypothesis.features, parameters=params,
        train_period=metrics.get("train_period", ("", "")),
        validation_period=metrics.get("validation_period", ("", "")),
        holdout_period=metrics.get("holdout_period", ("", "")),
        result=f"{status}: {hypothesis.statement}", metrics=metrics,
        status="COMPLETED" if status == "SUPPORTED" else "REJECTED",
    )
    return ResearchResult(record.experiment_id, hypothesis, status, tuple(reasons), metrics)
