"""Persistent paper-trading performance ledger."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EquitySnapshot:
    timestamp: str
    equity_try: float
    cash_try: float
    open_positions_value_try: float


class PaperPerformanceLedger:
    def __init__(self, path: str | Path = "data/paper_performance.jsonl") -> None:
        self.path = Path(path)

    def record(self, equity_try: float, cash_try: float, open_positions_value_try: float) -> EquitySnapshot:
        snapshot = EquitySnapshot(datetime.now(UTC).isoformat(), equity_try, cash_try, open_positions_value_try)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(snapshot), ensure_ascii=False) + "\n")
        return snapshot

    def snapshots(self) -> list[EquitySnapshot]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [EquitySnapshot(**json.loads(line)) for line in handle if line.strip()]

    def metrics(self, initial_equity: float) -> dict[str, float]:
        rows = self.snapshots()
        if initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if not rows:
            return {"net_return": 0.0, "max_drawdown": 0.0, "observations": 0.0}
        peak = initial_equity
        max_drawdown = 0.0
        for row in rows:
            peak = max(peak, row.equity_try)
            max_drawdown = min(max_drawdown, row.equity_try / peak - 1.0)
        return {
            "net_return": rows[-1].equity_try / initial_equity - 1.0,
            "max_drawdown": max_drawdown,
            "observations": float(len(rows)),
        }
