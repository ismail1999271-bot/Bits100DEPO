"""Institutional fund-flow intelligence for the daily opportunity engine.

The module is deliberately source-agnostic: TEFAS/KAP/vendor adapters should
normalize their observations into :class:`FundFlow` before the scoring layer.
"""

from dataclasses import dataclass
from datetime import date
from typing import Iterable
import math


@dataclass(frozen=True, slots=True)
class FundFlow:
    as_of: date
    fund_code: str
    fund_name: str
    net_flow_try: float
    source: str = "unknown"
    sector: str | None = None

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


@dataclass(frozen=True, slots=True)
class FundFlowAnomaly:
    fund_code: str
    as_of: date
    net_flow_try: float
    z_score: float
    is_anomalous: bool


@dataclass(frozen=True, slots=True)
class SectorFlow:
    as_of: date
    sector: str
    estimated_flow_try: float
    contributing_funds: tuple[str, ...]


def summarize_fund_flows(flows: Iterable[FundFlow], as_of: date | None = None) -> FundFlowSummary:
    rows = list(flows)
    if as_of is not None:
        rows = [row for row in rows if row.as_of == as_of]
    if not rows:
        raise ValueError("no fund-flow observations")
    day = as_of or max(row.as_of for row in rows)
    rows = [row for row in rows if row.as_of == day]
    inflows = sorted(
        (r for r in rows if r.net_flow_try > 0),
        key=lambda r: r.net_flow_try,
        reverse=True,
    )
    outflows = sorted((r for r in rows if r.net_flow_try < 0), key=lambda r: r.net_flow_try)
    return FundFlowSummary(
        day,
        round(sum(r.net_flow_try for r in inflows), 2),
        round(abs(sum(r.net_flow_try for r in outflows)), 2),
        round(sum(r.net_flow_try for r in rows), 2),
        tuple(r.fund_code for r in inflows),
        tuple(r.fund_code for r in outflows),
    )


def detect_flow_anomalies(
    flows: Iterable[FundFlow],
    lookback: int = 20,
    z_threshold: float = 2.0,
) -> list[FundFlowAnomaly]:
    """Flag fund-day flows that are unusually large for that fund.

    Each fund is evaluated independently and today's value is compared only
    with that fund's preceding observations, avoiding cross-fund contamination
    and look-ahead bias.
    """
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if z_threshold <= 0:
        raise ValueError("z_threshold must be positive")

    grouped: dict[str, list[FundFlow]] = {}
    for row in flows:
        grouped.setdefault(row.fund_code, []).append(row)

    result: list[FundFlowAnomaly] = []
    for fund_code, fund_rows in grouped.items():
        fund_rows.sort(key=lambda r: r.as_of)
        for index, row in enumerate(fund_rows):
            history = [r.net_flow_try for r in fund_rows[max(0, index - lookback):index]]
            if len(history) < 2:
                result.append(FundFlowAnomaly(fund_code, row.as_of, row.net_flow_try, 0.0, False))
                continue
            mean = sum(history) / len(history)
            variance = sum((value - mean) ** 2 for value in history) / len(history)
            std = math.sqrt(variance)
            if std == 0:
                if row.net_flow_try == mean:
                    z = 0.0
                    anomalous = False
                else:
                    z = math.copysign(math.inf, row.net_flow_try - mean)
                    anomalous = True
            else:
                z = (row.net_flow_try - mean) / std
                anomalous = abs(z) >= z_threshold
            result.append(FundFlowAnomaly(fund_code, row.as_of, row.net_flow_try, z, anomalous))
    return sorted(result, key=lambda r: (r.as_of, r.fund_code))


def summarize_sector_flows(
    flows: Iterable[FundFlow], as_of: date | None = None
) -> list[SectorFlow]:
    """Aggregate explicitly tagged fund flows by sector.

    For stock-level sector exposure, prefer ``estimate_sector_flows`` in
    ``smart_money.py`` because it weights a fund's flow by actual holdings.
    """
    rows = list(flows)
    if as_of is None:
        if not rows:
            raise ValueError("no fund-flow observations")
        as_of = max(row.as_of for row in rows)
    totals: dict[str, float] = {}
    funds: dict[str, set[str]] = {}
    for row in rows:
        if row.as_of != as_of or not row.sector:
            continue
        totals[row.sector] = totals.get(row.sector, 0.0) + row.net_flow_try
        funds.setdefault(row.sector, set()).add(row.fund_code)
    return [
        SectorFlow(as_of, sector, round(total, 2), tuple(sorted(funds[sector])))
        for sector, total in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    ]


def smart_money_score(net_flow_try: float, scale_try: float = 1_000_000_000) -> float:
    """Map net fund flow to a bounded open interval (-1, 1)."""
    if scale_try <= 0:
        raise ValueError("scale_try must be positive")
    score = math.tanh(net_flow_try / scale_try)
    if score >= 1.0:
        score = 0.999999
    elif score <= -1.0:
        score = -0.999999
    return score
