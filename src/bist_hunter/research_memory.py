"""Append-only research memory and signal attribution store."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SignalAttribution:
    signal_id: str
    symbol: str
    timestamp: str
    score: float
    outcome_return: float | None
    hit: bool | None
    feature_contributions: dict[str, float]
    source_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResearchMemoryEntry:
    experiment_id: str
    hypothesis: str
    result: str
    metrics: dict[str, float]
    created_at: str


class ResearchMemory:
    """JSONL persistence: durable, auditable, dependency-free research history."""

    def __init__(self, path: str | Path = "data/research_memory.jsonl") -> None:
        self.path = Path(path)

    def _append(self, payload: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def record_experiment(self, experiment_id: str, hypothesis: str, result: str, metrics: dict[str, float]) -> None:
        entry = ResearchMemoryEntry(experiment_id, hypothesis, result, metrics, datetime.now(UTC).isoformat())
        self._append({"type": "experiment", **asdict(entry)})

    def record_signal(self, attribution: SignalAttribution) -> None:
        self._append({"type": "signal", **asdict(attribution)})

    def load(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def summary(self) -> dict[str, int]:
        rows = self.load()
        return {
            "experiments": sum(row.get("type") == "experiment" for row in rows),
            "signals": sum(row.get("type") == "signal" for row in rows),
        }
