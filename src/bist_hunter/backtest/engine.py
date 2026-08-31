from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TradeResult:
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    gross_return: float
    net_return: float


@dataclass(frozen=True, slots=True)
class CostModel:
    commission_bps: float = 10.0
    slippage_bps: float = 5.0

    @property
    def round_trip_rate(self) -> float:
        return 2 * (self.commission_bps + self.slippage_bps) / 10_000


def simulate_trade(symbol: str, entry_price: float, exit_price: float, capital: float, costs: CostModel) -> TradeResult:
    if min(entry_price, exit_price, capital) <= 0:
        raise ValueError("prices and capital must be positive")
    gross = exit_price / entry_price - 1
    net = gross - costs.round_trip_rate
    quantity = capital / entry_price
    return TradeResult(symbol, entry_price, exit_price, quantity, gross, net)
