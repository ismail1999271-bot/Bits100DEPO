"""Join institutional fund flows to fund holdings without pretending a live feed exists."""

from dataclasses import dataclass
from datetime import date
from typing import Iterable
import math

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
    flows: Iterable[FundFlow], holdings: Iterable[FundHolding], as_of: date | None = None
) -> list[StockMoneyFlow]:
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
        StockMoneyFlow(symbol, round(total, 2), tuple(sorted(funds[symbol])))
        for symbol, total in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    ]


def stock_smart_money_score(estimated_flow_try: float, scale_try: float = 100_000_000) -> float:
    if scale_try <= 0:
        raise ValueError("scale_try must be positive")
    score = math.tanh(estimated_flow_try / scale_try)
    if score >= 1.0:
        score = 0.999999
    elif score <= -1.0:
        score = -0.999999
    return score
