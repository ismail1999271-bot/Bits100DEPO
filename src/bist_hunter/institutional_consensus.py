"""Institutional overlap and consensus scoring across portfolio managers.

Several funds belonging to the same manager count as one institutional manager.
Holdings are portfolio snapshots, not trade ledgers.
"""

from dataclasses import dataclass
from datetime import date
from typing import Iterable

DEFAULT_MANAGERS: tuple[str, ...] = (
    "Ak Portföy", "İş Portföy", "Yapı Kredi Portföy", "Garanti Portföy",
    "QNB Portföy", "TEB Portföy", "Deniz Portföy", "Oyak Portföy",
    "Ünlü Portföy", "Tacirler Portföy", "İnfo Portföy", "Tera Portföy",
    "Pusula Portföy", "Ata Portföy", "Azimut Portföy", "Albaraka Portföy",
)

@dataclass(frozen=True, slots=True)
class InstitutionalHolding:
    as_of: date
    manager: str
    fund_code: str
    fund_name: str
    symbol: str
    portfolio_weight: float
    fund_aum_try: float | None = None
    sector: str | None = None

    def __post_init__(self) -> None:
        if not self.manager.strip() or not self.symbol.strip():
            raise ValueError("manager and symbol are required")
        if not 0 <= self.portfolio_weight <= 1:
            raise ValueError("portfolio_weight must be between 0 and 1")
        if self.fund_aum_try is not None and self.fund_aum_try < 0:
            raise ValueError("fund_aum_try cannot be negative")

    @property
    def estimated_value_try(self) -> float | None:
        return None if self.fund_aum_try is None else self.fund_aum_try * self.portfolio_weight

@dataclass(frozen=True, slots=True)
class InstitutionalPositionChange:
    symbol: str
    manager: str
    previous_weight: float
    current_weight: float
    change: float
    status: str

@dataclass(frozen=True, slots=True)
class InstitutionalConsensus:
    symbol: str
    manager_count: int
    fund_count: int
    total_weight: float
    average_weight: float
    estimated_value_try: float | None
    new_manager_count: int
    increasing_manager_count: int
    decreasing_manager_count: int
    score: float
    managers: tuple[str, ...]

def _recency_weight(as_of: date, reference_date: date, half_life_days: float) -> float:
    age = max(0, (reference_date - as_of).days)
    return 0.5 ** (age / half_life_days)

def _breadth_score(manager_count: int, universe_size: int) -> float:
    if universe_size <= 0:
        raise ValueError("universe_size must be positive")
    return min(100.0, 100.0 * manager_count / universe_size)

def build_position_changes(previous: Iterable[InstitutionalHolding], current: Iterable[InstitutionalHolding], tolerance: float = 0.005) -> list[InstitutionalPositionChange]:
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    prev: dict[tuple[str, str], float] = {}
    cur: dict[tuple[str, str], float] = {}
    for row in previous:
        prev[(row.manager, row.symbol)] = prev.get((row.manager, row.symbol), 0.0) + row.portfolio_weight
    for row in current:
        cur[(row.manager, row.symbol)] = cur.get((row.manager, row.symbol), 0.0) + row.portfolio_weight
    changes = []
    for manager, symbol in sorted(set(prev) | set(cur)):
        old, new = prev.get((manager, symbol), 0.0), cur.get((manager, symbol), 0.0)
        delta = new - old
        if old == 0 and new > 0:
            status = "NEW"
        elif new == 0 and old > 0:
            status = "EXIT"
        elif delta > tolerance:
            status = "INCREASE"
        elif delta < -tolerance:
            status = "DECREASE"
        else:
            status = "STABLE"
        changes.append(InstitutionalPositionChange(symbol, manager, old, new, delta, status))
    return changes

def calculate_institutional_consensus(holdings: Iterable[InstitutionalHolding], changes: Iterable[InstitutionalPositionChange] = (), reference_date: date | None = None, manager_universe: Iterable[str] = DEFAULT_MANAGERS, half_life_days: float = 45.0) -> list[InstitutionalConsensus]:
    rows = list(holdings)
    if not rows:
        return []
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    ref = reference_date or max(row.as_of for row in rows)
    universe = tuple(dict.fromkeys(manager_universe))
    if not universe:
        raise ValueError("manager_universe must not be empty")
    changes_by_key = {(c.manager, c.symbol): c for c in changes}
    grouped: dict[str, list[InstitutionalHolding]] = {}
    for row in rows:
        grouped.setdefault(row.symbol, []).append(row)
    results = []
    for symbol, symbol_rows in grouped.items():
        managers = sorted({row.manager for row in symbol_rows})
        funds = {row.fund_code for row in symbol_rows}
        manager_recency = sum(_recency_weight(max(r.as_of for r in symbol_rows if r.manager == manager), ref, half_life_days) for manager in managers)
        recency_breadth = min(100.0, 100.0 * manager_recency / max(1, len(universe)))
        weighted_rows = [(row, _recency_weight(row.as_of, ref, half_life_days)) for row in symbol_rows]
        total_weight = sum(row.portfolio_weight * recency for row, recency in weighted_rows)
        avg_weight = total_weight / max(1, len(weighted_rows))
        values = [row.estimated_value_try * recency for row, recency in weighted_rows if row.estimated_value_try is not None]
        estimated_value = sum(values) if values else None
        exposure_score = min(100.0, 100.0 * avg_weight / 0.25)
        relevant = [changes_by_key[(manager, symbol)] for manager in managers if (manager, symbol) in changes_by_key]
        new_count = sum(c.status == "NEW" for c in relevant)
        increase_count = sum(c.status == "INCREASE" for c in relevant)
        decrease_count = sum(c.status == "DECREASE" for c in relevant)
        change_score = min(100.0, 100.0 * (new_count + increase_count) / max(1, len(universe)))
        breadth = _breadth_score(len(managers), len(universe))
        score = 0.40 * breadth + 0.20 * exposure_score + 0.15 * change_score + 0.15 * recency_breadth + 0.10 * min(100.0, 100.0 * new_count / max(1, len(universe)))
        results.append(InstitutionalConsensus(symbol, len(managers), len(funds), round(total_weight, 6), round(avg_weight, 6), round(estimated_value, 2) if estimated_value is not None else None, new_count, increase_count, decrease_count, round(max(0.0, min(100.0, score)), 2), tuple(managers)))
    return sorted(results, key=lambda row: (row.score, row.manager_count, row.average_weight), reverse=True)

def consensus_map(rows: Iterable[InstitutionalConsensus]) -> dict[str, float]:
    return {row.symbol: row.score for row in rows}
