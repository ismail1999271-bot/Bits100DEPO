"""Real-market-data paper session runner with persistent mark-to-market equity."""
from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from .paper_performance import PaperPerformanceLedger
from .real_adapters import load_symbol_universe, yahoo_chart


def _state_path() -> Path:
    return Path(os.getenv("PAPER_STATE_PATH", "data/paper_state.json"))


def _load_state() -> dict[str, object]:
    path = _state_path()
    if not path.exists():
        return {"positions": {}, "last_entry_date": ""}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(state: dict[str, object]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def run_paper_session(top_k: int = 5) -> dict[str, float]:
    symbols = load_symbol_universe()
    today = datetime.now(UTC).date()
    rows: dict[str, list[dict[str, object]]] = {}
    for symbol in symbols:
        rows[symbol] = yahoo_chart(symbol, today - timedelta(days=45), today)
    latest = {symbol: data[-1] for symbol, data in rows.items() if data}
    ranked = sorted(latest.items(), key=lambda item: float(item[1]["close"]), reverse=True)
    state = _load_state()
    positions = state.get("positions", {})
    initial = float(os.getenv("PAPER_INITIAL_EQUITY_TRY", "5000000"))
    if not positions and ranked:
        selected = ranked[:max(1, top_k)]
        allocation = initial / len(selected)
        positions = {symbol: {"quantity": allocation / float(row["close"]), "entry": float(row["close"])} for symbol, row in selected}
        state["positions"] = positions
        state["last_entry_date"] = today.isoformat()
    invested = 0.0
    for symbol, position in positions.items():
        if symbol in latest:
            invested += float(position["quantity"]) * float(latest[symbol]["close"])
    cash = max(0.0, initial - sum(float(p["quantity"]) * float(p["entry"]) for p in positions.values()))
    equity = cash + invested
    ledger = PaperPerformanceLedger(os.getenv("PAPER_DATA_PATH", "data/paper_performance.jsonl"))
    ledger.record(equity, cash, invested)
    _save_state(state)
    return ledger.metrics(initial)
