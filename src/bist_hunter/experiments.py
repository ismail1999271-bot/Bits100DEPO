"""Append-only experiment tracking (JSONL).

Each record: experiment_id, timestamp, dataset, features, parameters,
train_period, validation_period, holdout_period, result, metrics, status.
Records are never rewritten; a status change is a new line with the same id.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STATUSES = ("PROPOSED", "RUNNING", "COMPLETED", "REJECTED", "BLOCKED")


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    experiment_id: str
    timestamp: str
    dataset: str
    features: tuple[str, ...]
    parameters: dict[str, Any]
    train_period: tuple[str, str]
    validation_period: tuple[str, str]
    holdout_period: tuple[str, str]
    result: str
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "PROPOSED"

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")


def experiment_id(dataset: str, features, parameters: dict[str, Any]) -> str:
    payload = json.dumps({"d": dataset, "f": sorted(features), "p": parameters}, sort_keys=True, default=str)
    return "exp_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class ExperimentTracker:
    def __init__(self, path: str | Path = "data/experiments.jsonl") -> None:
        self.path = Path(path)

    def log(self, record: ExperimentRecord) -> ExperimentRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True, default=str) + "\n")
        return record

    def new(self, *, dataset: str, features, parameters: dict[str, Any], train_period=("", ""),
            validation_period=("", ""), holdout_period=("", ""), result: str = "",
            metrics: dict[str, Any] | None = None, status: str = "PROPOSED") -> ExperimentRecord:
        return self.log(ExperimentRecord(
            experiment_id(dataset, features, parameters), datetime.now(UTC).isoformat(), dataset,
            tuple(features), dict(parameters), tuple(train_period), tuple(validation_period),
            tuple(holdout_period), result, dict(metrics or {}), status,
        ))

    def history(self, experiment_id_: str | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [r for r in rows if experiment_id_ is None or r["experiment_id"] == experiment_id_]

    def latest(self) -> dict[str, dict[str, Any]]:
        """Latest record per experiment id."""
        out: dict[str, dict[str, Any]] = {}
        for row in self.history():
            out[row["experiment_id"]] = row
        return out
