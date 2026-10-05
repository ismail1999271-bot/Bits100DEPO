"""Realistic paper-trading ledger (simulation only - never routes an order).

Each fill records cash, position, entry, exit, PnL, fees, slippage, timestamp
and reason. Fills pay half the quoted spread plus a slippage estimate and a
commission (BIST brokerage + BSMV are folded into ``commission_bps``).
Timestamps must be timezone-aware and non-decreasing (no back-dated fills).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .paper_trading import RiskLimits


@dataclass(frozen=True, slots=True)
class CostModel:
    commission_bps: float = 4.0
    slippage_bps: float = 5.0
    min_spread_bps: float = 5.0

    def __post_init__(self) -> None:
        if min(self.commission_bps, self.slippage_bps, self.min_spread_bps) < 0:
            raise ValueError("costs must be non-negative")


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    timestamp: str
    action: str  # BUY / SELL / REJECT
    symbol: str
    quantity: float
    mid_price: float
    fill_price: float
    notional: float
    fees: float
    slippage: float  # TRY paid versus mid (spread + slippage)
    cash_after: float
    position_after: float
    entry_price: float | None
    exit_price: float | None
    realized_pnl: float | None
    reason: str


@dataclass(slots=True)
class OpenPosition:
    symbol: str
    quantity: float
    entry_price: float  # average fill incl. spread/slippage
    entry_fees: float
    opened_at: str


@dataclass
class PaperLedger:
    initial_cash: float
    limits: RiskLimits = field(default_factory=RiskLimits)
    costs: CostModel = field(default_factory=CostModel)
    max_participation: float = 0.02
    cash: float = 0.0
    positions: dict[str, OpenPosition] = field(default_factory=dict)
    entries: list[LedgerEntry] = field(default_factory=list)
    _last_ts: datetime | None = None
    _day_start_equity: float | None = None
    _day: str | None = None

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial cash must be positive")
        self.cash = float(self.initial_cash)

    # -- helpers -----------------------------------------------------------
    def _check_time(self, ts: datetime) -> None:
        if ts.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if self._last_ts is not None and ts < self._last_ts:
            raise ValueError("back-dated paper fill rejected")
        self._last_ts = ts
        day = ts.date().isoformat()
        if day != self._day:
            self._day = day
            self._day_start_equity = self.equity({})

    def _half_spread_bps(self, spread_bps: float | None) -> float:
        return max(self.costs.min_spread_bps, spread_bps or 0.0) / 2

    def equity(self, marks: dict[str, float]) -> float:
        value = self.cash
        for symbol, pos in self.positions.items():
            value += pos.quantity * marks.get(symbol, pos.entry_price)
        return value

    def _reject(self, ts, symbol, qty, mid, reason) -> bool:
        self.entries.append(LedgerEntry(ts.isoformat(), "REJECT", symbol, qty, mid, 0.0, 0.0, 0.0, 0.0,
                                        self.cash, self.positions.get(symbol).quantity if symbol in self.positions
                                        else 0.0, None, None, None, reason))
        return False

    # -- actions -----------------------------------------------------------
    def buy(self, ts: datetime, symbol: str, quantity: float, mid_price: float, reason: str, *,
            spread_bps: float | None = None, avg_daily_value_try: float | None = None,
            marks: dict[str, float] | None = None) -> bool:
        self._check_time(ts)
        symbol = symbol.upper()
        if quantity <= 0 or mid_price <= 0:
            return self._reject(ts, symbol, quantity, mid_price, "INVALID_ORDER")
        cost_bps = self._half_spread_bps(spread_bps) + self.costs.slippage_bps
        fill = mid_price * (1 + cost_bps / 10_000)
        notional = fill * quantity
        fees = notional * self.costs.commission_bps / 10_000
        equity = self.equity(marks or {})
        if notional + fees > self.cash:
            return self._reject(ts, symbol, quantity, mid_price, "INSUFFICIENT_CASH")
        if notional > self.limits.max_order_notional_try:
            return self._reject(ts, symbol, quantity, mid_price, "ORDER_NOTIONAL_LIMIT")
        held = self.positions.get(symbol)
        if (notional + (held.quantity * held.entry_price if held else 0.0)) > equity * self.limits.max_position_pct:
            return self._reject(ts, symbol, quantity, mid_price, "POSITION_SIZE_LIMIT")
        if held is None and len(self.positions) >= self.limits.max_open_positions:
            return self._reject(ts, symbol, quantity, mid_price, "MAX_OPEN_POSITIONS")
        if avg_daily_value_try is None:
            return self._reject(ts, symbol, quantity, mid_price, "LIQUIDITY_UNKNOWN")
        if notional > avg_daily_value_try * self.max_participation:
            return self._reject(ts, symbol, quantity, mid_price, "LIQUIDITY_LIMIT")
        if self._day_start_equity and equity < self._day_start_equity * (1 - self.limits.max_daily_loss_pct):
            return self._reject(ts, symbol, quantity, mid_price, "DAILY_LOSS_LIMIT")
        self.cash -= notional + fees
        if held:
            total = held.quantity + quantity
            held.entry_price = (held.entry_price * held.quantity + fill * quantity) / total
            held.quantity = total
            held.entry_fees += fees
        else:
            self.positions[symbol] = OpenPosition(symbol, quantity, fill, fees, ts.isoformat())
        pos = self.positions[symbol]
        self.entries.append(LedgerEntry(ts.isoformat(), "BUY", symbol, quantity, mid_price, round(fill, 6),
                                        round(notional, 4), round(fees, 4),
                                        round((fill - mid_price) * quantity, 4), round(self.cash, 4),
                                        pos.quantity, round(pos.entry_price, 6), None, None, reason))
        return True

    def sell(self, ts: datetime, symbol: str, mid_price: float, reason: str, *,
             quantity: float | None = None, spread_bps: float | None = None) -> bool:
        self._check_time(ts)
        symbol = symbol.upper()
        pos = self.positions.get(symbol)
        if pos is None or mid_price <= 0:
            return self._reject(ts, symbol, quantity or 0.0, mid_price, "NO_POSITION")
        qty = pos.quantity if quantity is None else min(quantity, pos.quantity)
        cost_bps = self._half_spread_bps(spread_bps) + self.costs.slippage_bps
        fill = mid_price * (1 - cost_bps / 10_000)
        notional = fill * qty
        fees = notional * self.costs.commission_bps / 10_000
        entry_fee_share = pos.entry_fees * qty / pos.quantity
        pnl = (fill - pos.entry_price) * qty - fees - entry_fee_share
        self.cash += notional - fees
        pos.quantity -= qty
        pos.entry_fees -= entry_fee_share
        remaining = pos.quantity
        entry_price = pos.entry_price
        if remaining <= 1e-12:
            del self.positions[symbol]
            remaining = 0.0
        self.entries.append(LedgerEntry(ts.isoformat(), "SELL", symbol, qty, mid_price, round(fill, 6),
                                        round(notional, 4), round(fees, 4),
                                        round((mid_price - fill) * qty, 4), round(self.cash, 4), remaining,
                                        round(entry_price, 6), round(fill, 6), round(pnl, 4), reason))
        return True

    # -- reporting ---------------------------------------------------------
    def summary(self, marks: dict[str, float] | None = None) -> dict[str, float]:
        trades = [e for e in self.entries if e.action == "SELL"]
        fills = [e for e in self.entries if e.action in ("BUY", "SELL")]
        equity = self.equity(marks or {})
        return {
            "cash": round(self.cash, 4),
            "equity": round(equity, 4),
            "return_pct": round((equity / self.initial_cash - 1) * 100, 4),
            "realized_pnl": round(sum(e.realized_pnl or 0.0 for e in trades), 4),
            "fees": round(sum(e.fees for e in fills), 4),
            "slippage": round(sum(e.slippage for e in fills), 4),
            "closed_trades": len(trades),
            "win_rate": round(sum((e.realized_pnl or 0) > 0 for e in trades) / len(trades), 4) if trades else 0.0,
            "open_positions": len(self.positions),
            "rejections": sum(e.action == "REJECT" for e in self.entries),
        }

    def write_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
