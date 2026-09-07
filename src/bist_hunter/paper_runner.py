"""Real-market-data paper session runner with persistent mark-to-market equity."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from .daily_ranker import rank_latest
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
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        rows.extend(yahoo_chart(symbol, today - timedelta(days=45), today))
    frame = pd.DataFrame(rows)
    ranked = rank_latest(frame, config=None) if False else rank_latest(frame)
    state = _load_state()
    positions = state.get("positions", {})
    initial = float(os.getenv("PAPER_INITIAL_EQUITY_TRY", "5000000"))
    if not positions and not ranked.empty:
        selected = ranked.head(max(1, top_k))
        allocation = initial / len(selected)
        positions = {
            str(row.symbol): {"quantity": allocation / float(row.close), "entry": float(row.close)}
            for row in selected.itertuples()
        }
        state["positions"] = positions
        state["last_entry_date"] = today.isoformat()
    latest = {str(row.symbol): row for row in frame.sort_values("timestamp").groupby("symbol").tail(1).itertuples()}
    invested = sum(
        float(position["quantity"]) * float(latest[symbol].close)
        for symbol, position in positions.items()
        if symbol in latest
    )
    cash = max(0.0, initial - sum(float(p["quantity"]) * float(p["entry"]) for p in positions.values()))
    equity = cash + invested
    ledger = PaperPerformanceLedger(os.getenv("PAPER_DATA_PATH", "data/paper_performance.jsonl"))
    ledger.record(equity, cash, invested)
    _save_state(state)
    return ledger.metrics(initial)
