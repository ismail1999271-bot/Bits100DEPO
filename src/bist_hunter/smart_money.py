"""Join institutional fund flows to fund holdings without pretending a live feed exists."""

from dataclasses import dataclass
from datetime import date
from typing import Iterable

from .fund_flow import FundFlow


@dataclass(frozen=True, slots=True)
class FundHolding:
    as_of: date
    fund_code: str
    symbol: str
    portfolio_weight: float

    def __post_init__(self) -> None:
        if not 0 <= self.portfolio_weight <= 1:
            raise ValueError("portfolio_weight must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class StockMoneyFlow:
    symbol: str
    estimated_flow_try: float
    contributing_funds: tuple[str, ...]


def estimate_stock_flows(
    flows: Iterable[FundFlow],
    holdings: Iterable[FundHolding],
    as_of: date | None = None,
) -> list[StockMoneyFlow]:
    """Allocate each fund's net flow across its reported holdings.

    This is an exposure proxy, not a claim that the fund bought/sold each stock
    by the allocated amount. It becomes meaningful only when holdings and flows
    share the same observation date.
    """
    flow_rows = list(flows)
    holding_rows = list(holdings)
    if not flow_rows or not holding_rows:
        raise ValueError("flows and holdings are required")
    day = as_of or max(max(f.as_of for f in flow_rows), max(h.as_of for h in holding_rows))
    flow_map = {f.fund_code: f for f in flow_rows if f.as_of == day}
    totals: dict[str, float] = {}
    funds: dict[str, set[str]] = {}
    for holding in holding_rows:
        if holding.as_of != day or holding.fund_code not in flow_map:
            continue
        flow = flow_map[holding.fund_code].net_flow_try * holding.portfolio_weight
        totals[holding.symbol] = totals.get(holding.symbol, 0.0) + flow
        funds.setdefault(holding.symbol, set()).add(holding.fund_code)
    return [
        StockMoneyFlow(symbol=symbol, estimated_flow_try=round(total, 2), contributing_funds=tuple(sorted(funds[symbol])))
        for symbol, total in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    ]


def stock_smart_money_score(estimated_flow_try: float, scale_try: float = 100_000_000) -> float:
    """Bound an estimated stock-level flow to [-1, 1]."""
    import math

    if scale_try <= 0:
        raise ValueError("scale_try must be positive")
    return round(math.tanh(estimated_flow_try / scale_try), 6)
