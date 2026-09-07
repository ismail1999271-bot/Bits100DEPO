"""Real-market-data paper session with persistent state, risk caps and costs."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from .daily_ranker import rank_latest
from .paper_performance import PaperPerformanceLedger
from .paper_trading import RiskLimits
from .real_adapters import historical_bars, load_symbol_universe


def _state_path() -> Path:
    return Path(os.getenv("PAPER_STATE_PATH", "data/paper_state.json"))


def _load_state() -> dict[str, object]:
    path = _state_path()
    if not path.exists():
        return {"positions": {}, "cash": None, "last_entry_date": ""}
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
        rows.extend(historical_bars(symbol, today - timedelta(days=90), today))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("paper session received no historical rows")
    ranked = rank_latest(frame)
    state = _load_state()
    positions = dict(state.get("positions", {}))
    initial = float(os.getenv("PAPER_INITIAL_EQUITY_TRY", "5000000"))
    limits = RiskLimits(
        max_position_pct=float(os.getenv("PAPER_MAX_POSITION_PCT", "0.10")),
        max_open_positions=int(os.getenv("PAPER_MAX_OPEN_POSITIONS", "5")),
        max_daily_loss_pct=float(os.getenv("PAPER_MAX_DAILY_LOSS_PCT", "0.03")),
        max_order_notional_try=float(os.getenv("PAPER_MAX_ORDER_NOTIONAL_TRY", "100000")),
    )
    commission = float(os.getenv("PAPER_COMMISSION_PCT", "0.001"))
    slippage = float(os.getenv("PAPER_SLIPPAGE_PCT", "0.001"))

    if not positions and not ranked.empty:
        selected = ranked.head(min(max(1, top_k), limits.max_open_positions))
        per_position = min(initial * limits.max_position_pct, limits.max_order_notional_try)
        for row in selected.itertuples():
            price = float(row.close) * (1 + slippage)
            quantity = per_position / price
            if quantity > 0:
                positions[str(row.symbol)] = {"quantity": quantity, "entry": price, "entry_cost": per_position * (1 + commission)}
        state["last_entry_date"] = today.isoformat()

    latest = {str(row.symbol): row for row in frame.sort_values("timestamp").groupby("symbol").tail(1).itertuples()}
    invested = sum(float(p["quantity"]) * float(latest[symbol].close) * (1 - slippage) for symbol, p in positions.items() if symbol in latest)
    entry_cost = sum(float(p.get("entry_cost", float(p["quantity"]) * float(p["entry"]))) for p in positions.values())
    cash = max(0.0, initial - entry_cost)
    equity = cash + invested
    ledger = PaperPerformanceLedger(os.getenv("PAPER_DATA_PATH", "data/paper_performance.jsonl"))
    ledger.record(equity, cash, invested)
    state["positions"] = positions
    state["cash"] = cash
    _save_state(state)
    return ledger.metrics(initial)
