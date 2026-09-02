"""Institutional fund-flow intelligence for the daily opportunity engine.

The module is provider-neutral: it accepts normalized daily fund flows and does
not pretend that a live feed exists. It turns fund inflow/outflow into a
repeatable feature that can later be joined with BIST symbols, news and social
signals.
"""

from dataclasses import dataclass
from datetime import date
from typing import Iterable


@dataclass(frozen=True, slots=True)
class FundFlow:
    """One fund's daily net money flow, in TRY."""

    as_of: date
    fund_code: str
    fund_name: str
    net_flow_try: float
    source: str = "unknown"

    @property
    def direction(self) -> str:
        if self.net_flow_try > 0:
            return "INFLOW"
        if self.net_flow_try < 0:
            return "OUTFLOW"
        return "FLAT"


@dataclass(frozen=True, slots=True)
class FundFlowSummary:
    as_of: date
    total_inflow_try: float
    total_outflow_try: float
    net_flow_try: float
    inflow_funds: tuple[str, ...]
    outflow_funds: tuple[str, ...]


def summarize_fund_flows(flows: Iterable[FundFlow], as_of: date | None = None) -> FundFlowSummary:
    rows = list(flows)
    if as_of is not None:
        rows = [row for row in rows if row.as_of == as_of]
    if not rows:
        raise ValueError("no fund-flow observations")

    day = as_of or max(row.as_of for row in rows)
    rows = [row for row in rows if row.as_of == day]
    inflows = sorted((r for r in rows if r.net_flow_try > 0), key=lambda r: r.net_flow_try, reverse=True)
    outflows = sorted((r for r in rows if r.net_flow_try < 0), key=lambda r: r.net_flow_try)
    return FundFlowSummary(
        as_of=day,
        total_inflow_try=round(sum(r.net_flow_try for r in inflows), 2),
        total_outflow_try=round(abs(sum(r.net_flow_try for r in outflows)), 2),
        net_flow_try=round(sum(r.net_flow_try for r in rows), 2),
        inflow_funds=tuple(r.fund_code for r in inflows),
        outflow_funds=tuple(r.fund_code for r in outflows),
    )


def smart_money_score(net_flow_try: float, scale_try: float = 1_000_000_000) -> float:
    """Map net fund flow to a bounded -1..1 feature for model input."""
    if scale_try <= 0:
        raise ValueError("scale_try must be positive")
    # Smooth saturation prevents a single extreme observation dominating the model.
    import math

    return round(math.tanh(net_flow_try / scale_try), 6)
