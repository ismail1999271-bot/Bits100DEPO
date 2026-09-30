"""BIST100+ research ranking: Quant Score + fail-closed gate + risk references.

The output is a *research ordering*, not a forecast and not an order list.
Symbols that fail the gate stay in the table with status BLOCKED and their
reasons, ranked after every PASS symbol, so the user sees exactly what is
missing instead of silently shrinking the universe.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

from .fail_closed import GateInput, evaluate_gate
from .quant_score import calculate_quant_score
from .risk import RiskInputs, assess_risk
from .universe import Universe

DISCLAIMER = "Araştırma sıralamasıdır; kesin gelecek tahmini veya yatırım tavsiyesi değildir. Otomatik emir yoktur."
COLUMNS = ("Rank", "Symbol", "Status", "Quant Score", "Coverage", "Confidence", "Tavan-DNA", "Auction",
           "Technical", "Volume", "Institutional", "Broker", "Risk", "Entry", "Stop", "Target", "R/R", "Reasons")


@dataclass(frozen=True, slots=True)
class SymbolResearch:
    symbol: str
    gate: GateInput
    components: dict[str, float | None] = field(default_factory=dict)
    quality: dict[str, float] | None = None
    relative_volume: float | None = None
    risk: RiskInputs | None = None


def rank_research(
    universe: Universe,
    items: list[SymbolResearch],
    *,
    min_score: float = 70.0,
    min_coverage: float = 0.40,
) -> pd.DataFrame:
    by_symbol = {item.symbol.upper(): item for item in items}
    outside = set(by_symbol) - set(universe.symbols)
    if outside:
        raise ValueError(f"symbols outside the BIST100+ universe: {sorted(outside)}")
    rows = []
    for symbol in universe.symbols:
        item = by_symbol.get(symbol)
        if item is None:
            rows.append({"Symbol": symbol, "Status": "BLOCKED", "Reasons": "MISSING_MARKET_DATA",
                         "_sort": (1, 0.0)})
            continue
        comps = {k: v for k, v in item.components.items()}
        score = calculate_quant_score(comps, min_score=min_score, min_coverage=min_coverage, quality=item.quality)
        gate = evaluate_gate(replace(item.gate, coverage=score.coverage, min_coverage=min_coverage))
        risk = assess_risk(item.risk) if item.risk is not None else None
        reasons = list(gate.reasons) + list(score.reasons)
        if risk is not None:
            reasons += list(risk.reasons)
        status = "BLOCKED" if not gate.passed else score.status
        rows.append({
            "Symbol": symbol,
            "Status": status,
            "Quant Score": score.score if gate.passed else None,
            "Coverage": score.coverage,
            "Confidence": score.confidence,
            "Tavan-DNA": score.component("tavan_dna"),
            "Auction": score.component("auction"),
            "Technical": score.component("technical"),
            "Volume": item.relative_volume,
            "Institutional": score.component("institutional"),
            "Broker": score.component("broker"),
            "Risk": None if risk is None else risk.overall,
            "Entry": None if risk is None else risk.entry_reference,
            "Stop": None if risk is None else risk.stop_reference,
            "Target": None if risk is None else risk.target_reference,
            "R/R": None if risk is None else risk.risk_reward,
            "Reasons": ", ".join(dict.fromkeys(reasons)),
            "_sort": (0 if gate.passed else 1, -(score.score if gate.passed else 0.0)),
        })
    table = pd.DataFrame(rows)
    table = table.sort_values("_sort", kind="stable").drop(columns="_sort")
    table.insert(0, "Rank", range(1, len(table) + 1))
    for column in COLUMNS:
        if column not in table.columns:
            table[column] = None
    table = table[list(COLUMNS)].reset_index(drop=True)
    table.attrs["disclaimer"] = DISCLAIMER
    table.attrs["universe"] = f"{universe.name} {universe.as_of} ({universe.source})"
    return table

