"""Leakage-safe feedback ledger for research agents.

The agents learn from completed, timestamped experiments rather than future
market outcomes during signal generation. This is a research memory, not an
online trading policy updater.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExperimentFeedback:
    experiment_id: str
    observed_at: str
    accepted: bool
    top_k_precision: float
    net_pnl: float
    max_drawdown: float
    note: str = ""


class FeedbackLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, feedback: ExperimentFeedback) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(feedback), ensure_ascii=False) + "\n")

    def recent(self, limit: int = 50) -> list[ExperimentFeedback]:
        if limit < 1 or not self.path.exists():
            return []
        rows = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [ExperimentFeedback(**json.loads(row)) for row in rows]

    def summary(self) -> dict[str, float]:
        rows = self.recent(500)
        if not rows:
            return {"experiments": 0.0, "acceptance_rate": 0.0, "mean_precision": 0.0, "mean_pnl": 0.0}
        return {
            "experiments": float(len(rows)),
            "acceptance_rate": sum(r.accepted for r in rows) / len(rows),
            "mean_precision": sum(r.top_k_precision for r in rows) / len(rows),
            "mean_pnl": sum(r.net_pnl for r in rows) / len(rows),
        }
