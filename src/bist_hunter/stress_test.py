"""Portfolio stress test (research only; real inputs or MISSING).

Betas/correlations are estimated from real return history supplied by the caller.
Nothing is assumed: a position without enough history gets ``beta=None`` and the
scenario reports it as uncovered instead of guessing 1.0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_OBS = 60


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    value_try: float
    sector: str | None = None


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    name: str
    index_shock: float
    pnl_try: float | None
    pnl_pct: float | None
    covered_fraction: float
    uncovered: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StressReport:
    status: str  # OK / MISSING
    total_value: float
    concentration_hhi: float | None
    top_position_pct: float | None
    sector_exposure: dict[str, float]
    sector_unknown_pct: float
    avg_pair_correlation: float | None
    betas: dict[str, float | None]
    scenarios: tuple[ScenarioResult, ...]
    warnings: tuple[str, ...]


def estimate_beta(asset: pd.Series, index: pd.Series, min_obs: int = MIN_OBS) -> float | None:
    both = pd.concat([asset, index], axis=1, keys=["a", "i"]).dropna()
    if len(both) < min_obs:
        return None
    var = both["i"].var(ddof=1)
    if not var or var <= 0:
        return None
    return float(both["a"].cov(both["i"]) / var)


def stress_portfolio(
    positions: list[Position],
    returns: pd.DataFrame | None,
    index_returns: pd.Series | None,
    *,
    shocks: dict[str, float] | None = None,
    max_position_pct: float = 0.25,
    max_sector_pct: float = 0.45,
    rate_shock_betas: dict[str, float] | None = None,
) -> StressReport:
    """``returns``: columns = symbols, daily returns. ``shocks``: scenario name -> index move."""
    shocks = shocks or {"XU100 -10%": -0.10, "XU100 -20%": -0.20, "Resesyon (-30%)": -0.30}
    total = float(sum(p.value_try for p in positions))
    if not positions or total <= 0:
        return StressReport("MISSING", 0.0, None, None, {}, 0.0, None, {}, (), ("no positions",))
    weights = {p.symbol: p.value_try / total for p in positions}
    hhi = round(sum(w * w for w in weights.values()), 6)
    top = round(max(weights.values()) * 100, 4)
    warnings: list[str] = []
    if max(weights.values()) > max_position_pct:
        warnings.append("position_concentration")

    sectors: dict[str, float] = {}
    unknown = 0.0
    for p in positions:
        if p.sector is None:
            unknown += weights[p.symbol]
        else:
            sectors[p.sector] = sectors.get(p.sector, 0.0) + weights[p.symbol]
    sectors = {k: round(v * 100, 4) for k, v in sorted(sectors.items(), key=lambda kv: -kv[1])}
    if any(v > max_sector_pct * 100 for v in sectors.values()):
        warnings.append("sector_concentration")
    if unknown > 0:
        warnings.append("sector_unknown")

    corr = None
    betas: dict[str, float | None] = {p.symbol: None for p in positions}
    if returns is not None and not returns.empty:
        cols = [p.symbol for p in positions if p.symbol in returns.columns]
        if len(cols) >= 2:
            c = returns[cols].dropna(how="all").corr(min_periods=MIN_OBS).to_numpy()
            iu = np.triu_indices_from(c, k=1)
            vals = c[iu][~np.isnan(c[iu])]
            corr = round(float(vals.mean()), 4) if len(vals) else None
            if corr is not None and corr > 0.7:
                warnings.append("high_correlation")
        if index_returns is not None:
            for s in cols:
                b = estimate_beta(returns[s], index_returns)
                betas[s] = None if b is None else round(b, 4)
    else:
        warnings.append("no_return_history")

    results = []
    for name, shock in shocks.items():
        covered = [p for p in positions if betas[p.symbol] is not None]
        uncovered = tuple(p.symbol for p in positions if betas[p.symbol] is None)
        if not covered:
            results.append(ScenarioResult(name, shock, None, None, 0.0, uncovered))
            continue
        pnl = sum(p.value_try * betas[p.symbol] * shock for p in covered)
        cov_val = sum(p.value_try for p in covered)
        results.append(ScenarioResult(name, shock, round(pnl, 2), round(pnl / total * 100, 4),
                                      round(cov_val / total, 4), uncovered))
    if rate_shock_betas:
        pnl_r = sum(p.value_try * rate_shock_betas[p.symbol] for p in positions if p.symbol in rate_shock_betas)
        cov = sum(p.value_try for p in positions if p.symbol in rate_shock_betas) / total
        results.append(ScenarioResult("Faiz şoku (+kullanıcı verisi)", float("nan"), round(pnl_r, 2),
                                      round(pnl_r / total * 100, 4), round(cov, 4),
                                      tuple(p.symbol for p in positions if p.symbol not in rate_shock_betas)))
    status = "OK" if any(r.pnl_try is not None for r in results) else "MISSING"
    return StressReport(status, round(total, 2), hhi, top, sectors, round(unknown * 100, 4), corr, betas,
                        tuple(results), tuple(warnings))
