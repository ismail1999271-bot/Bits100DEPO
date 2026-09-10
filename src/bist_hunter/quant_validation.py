"""Deterministic performance metrics for signal validation.

This layer is deliberately independent of any model or broker. It consumes
realized returns and applies explicit transaction costs/slippage supplied by
the caller. It never turns an LLM opinion into an order.
"""
from dataclasses import dataclass
from math import isfinite, sqrt


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    sample_size: int
    hit_rate: float
    expectancy: float
    max_drawdown: float
    sharpe: float | None
    sortino: float | None
    profit_factor: float | None


def _clean(values: list[float]) -> list[float]:
    return [float(v) for v in values if isfinite(float(v))]


def validate_returns(returns: list[float], cost_bps: float = 0.0, slippage_bps: float = 0.0) -> ValidationMetrics:
    values = _clean(returns)
    friction = (cost_bps + slippage_bps) / 10_000.0
    net = [v - friction for v in values]
    n = len(net)
    if not n:
        return ValidationMetrics(0, 0.0, 0.0, 0.0, None, None, None)
    wins = [v for v in net if v > 0]
    losses = [v for v in net if v < 0]
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in net:
        equity *= 1.0 + r
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
    mean = sum(net) / n
    variance = sum((v - mean) ** 2 for v in net) / (n - 1) if n > 1 else 0.0
    downside = [min(0.0, v) ** 2 for v in net]
    downside_dev = sqrt(sum(downside) / n)
    sharpe = mean / sqrt(variance) * sqrt(252.0) if variance > 0 else None
    sortino = mean / downside_dev * sqrt(252.0) if downside_dev > 0 else None
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    return ValidationMetrics(
        n,
        len(wins) / n,
        mean,
        max_dd,
        sharpe,
        sortino,
        profit_factor,
    )
