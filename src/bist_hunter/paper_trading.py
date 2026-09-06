"""Safe paper-trading ledger with hard fail-safe limits and audit records."""
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class RiskLimits:
    max_position_pct: float = 0.10
    max_open_positions: int = 5
    max_daily_loss_pct: float = 0.03
    max_order_notional_try: float = 100_000.0

    def __post_init__(self) -> None:
        if not 0 < self.max_position_pct <= 1 or self.max_open_positions < 1:
            raise ValueError("invalid position limits")
        if not 0 < self.max_daily_loss_pct <= 1 or self.max_order_notional_try <= 0:
            raise ValueError("invalid loss/order limits")


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    quantity: float
    entry_price: float


@dataclass(frozen=True, slots=True)
class AuditEvent:
    timestamp: datetime
    action: str
    symbol: str
    quantity: float
    price: float
    reason: str


@dataclass
class PaperBroker:
    equity_try: float
    limits: RiskLimits = field(default_factory=RiskLimits)
    positions: dict[str, Position] = field(default_factory=dict)
    audit: list[AuditEvent] = field(default_factory=list)
    day_start_equity_try: float | None = None

    def __post_init__(self) -> None:
        if self.equity_try <= 0:
            raise ValueError("equity must be positive")
        self.day_start_equity_try = self.equity_try

    def _daily_loss_breached(self) -> bool:
        start = self.day_start_equity_try or self.equity_try
        return self.equity_try < start * (1 - self.limits.max_daily_loss_pct)

    def can_buy(self, symbol: str, quantity: float, price: float) -> tuple[bool, str]:
        notional = quantity * price
        if quantity <= 0 or price <= 0:
            return False, "INVALID_ORDER"
        if notional > self.limits.max_order_notional_try:
            return False, "ORDER_NOTIONAL_LIMIT"
        if symbol not in self.positions and len(self.positions) >= self.limits.max_open_positions:
            return False, "MAX_OPEN_POSITIONS"
        if notional > self.equity_try * self.limits.max_position_pct:
            return False, "POSITION_SIZE_LIMIT"
        if self._daily_loss_breached():
            return False, "DAILY_LOSS_LIMIT"
        return True, "OK"

    def buy(self, symbol: str, quantity: float, price: float, reason: str) -> bool:
        allowed, _ = self.can_buy(symbol, quantity, price)
        if not allowed:
            return False
        self.positions[symbol] = Position(symbol, quantity, price)
        self.audit.append(AuditEvent(datetime.now(timezone.utc), "BUY", symbol, quantity, price, reason))
        return True

    def sell(self, symbol: str, price: float, reason: str) -> bool:
        position = self.positions.pop(symbol, None)
        if position is None or price <= 0:
            return False
        pnl = (price - position.entry_price) * position.quantity
        self.equity_try += pnl
        self.audit.append(AuditEvent(datetime.now(timezone.utc), "SELL", symbol, position.quantity, price, reason))
        return True
