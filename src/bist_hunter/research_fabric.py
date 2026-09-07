"""Scalable research-fabric contracts for broad data and simulation fan-out.

The fabric is provider-neutral: it can orchestrate many registered sources and
many deterministic experiment configurations without pretending that external
data is already connected. Live providers must be registered explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class DataSourceSpec:
    source_id: str
    category: str
    provider: str
    endpoint: str = ""
    point_in_time: bool = False
    timestamp_required: bool = True
    enabled: bool = False
    priority: int = 100


@dataclass(frozen=True, slots=True)
class DataCoverage:
    """Coverage health, deliberately separate from connection status."""

    discovered: int
    enabled: int
    point_in_time_ready: int

    @property
    def capacity(self) -> int:
        return self.discovered


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """One independent experiment/scenario in the research grid."""

    model: str
    feature_set: str
    horizon: int
    threshold: float
    cost_bps: float
    seed: int


def build_simulation_grid(
    models: Iterable[str],
    feature_sets: Iterable[str],
    horizons: Iterable[int],
    thresholds: Iterable[float],
    cost_bps: Iterable[float],
    seeds: Iterable[int],
    max_runs: int = 48_000,
) -> list[SimulationConfig]:
    """Create a bounded Cartesian research grid.

    ``max_runs`` is a capacity target, not a claim that 48,000 simulations have
    actually run. Production execution should supply a parallel evaluator and
    record each run's inputs and outputs for reproducibility.
    """
    if max_runs < 1:
        raise ValueError("max_runs must be positive")
    grid = (
        SimulationConfig(*values)
        for values in product(models, feature_sets, horizons, thresholds, cost_bps, seeds)
    )
    return list(_take(grid, max_runs))


def _take(items: Iterable[T], limit: int) -> Iterable[T]:
    for index, item in enumerate(items):
        if index >= limit:
            break
        yield item


def summarize_coverage(sources: Iterable[DataSourceSpec]) -> DataCoverage:
    items = list(sources)
    return DataCoverage(
        discovered=len(items),
        enabled=sum(source.enabled for source in items),
        point_in_time_ready=sum(source.enabled and source.point_in_time for source in items),
    )


def execute_parallel(
    configs: Iterable[SimulationConfig], evaluator: Callable[[SimulationConfig], R],
    workers: int = 1,
) -> list[R]:
    """Execute independent simulations with optional thread parallelism.

    The evaluator must be deterministic for a fixed config/data snapshot. A
    process-based backend can replace this later without changing the contract.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    configs = list(configs)
    if workers == 1:
        return [evaluator(config) for config in configs]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(evaluator, configs))
