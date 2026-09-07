"""Historical performance leaderboard for daily reports.

This module turns the useful idea behind social-media performance boards into
machine-readable, leakage-safe research data. It is descriptive, not a
trading signal by itself.
"""
from __future__ import annotations

from dataclasses import dataclass

from .domain import MarketBar


@dataclass(frozen=True, slots=True)
class PerformanceRow:
    symbol: str
    period_return_pct: float
    previous_period_return_pct: float
    rank: int
    previous_rank: int

    @property
    def rank_change(self) -> int:
        """Positive means the stock improved its leaderboard rank."""
        return self.previous_rank - self.rank

    @property
    def acceleration_pct(self) -> float:
        """Change in period return, measured in percentage points."""
        return self.period_return_pct - self.previous_period_return_pct


def build_performance_board(
    bars_by_symbol: dict[str, list[MarketBar]],
    period_bars: int = 20,
    top_k: int = 20,
) -> tuple[PerformanceRow, ...]:
    """Build a trailing-period leaderboard plus rank acceleration.

    ``period_bars=20`` is a practical monthly proxy for BIST trading days.
    Symbols without enough history for two complete periods are excluded.
    """
    if period_bars < 1:
        raise ValueError("period_bars must be >= 1")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    current: dict[str, float] = {}
    previous: dict[str, float] = {}
    for symbol, bars in bars_by_symbol.items():
        ordered = sorted(bars, key=lambda bar: bar.timestamp)
        needed = 2 * period_bars + 1
        if len(ordered) < needed:
            continue
        end = ordered[-1].close
        period_start = ordered[-period_bars - 1].close
        previous_start = ordered[-2 * period_bars - 1].close
        if period_start <= 0 or previous_start <= 0:
            continue
        current[symbol] = (end / period_start - 1.0) * 100.0
        previous[symbol] = (period_start / previous_start - 1.0) * 100.0

    current_ranked = sorted(current, key=lambda symbol: (-current[symbol], symbol))
    previous_ranked = sorted(previous, key=lambda symbol: (-previous[symbol], symbol))
    previous_ranks = {symbol: i + 1 for i, symbol in enumerate(previous_ranked)}

    rows = [
        PerformanceRow(
            symbol=symbol,
            period_return_pct=round(current[symbol], 2),
            previous_period_return_pct=round(previous[symbol], 2),
            rank=i + 1,
            previous_rank=previous_ranks[symbol],
        )
        for i, symbol in enumerate(current_ranked)
        if symbol in previous_ranks
    ]
    return tuple(rows[:top_k])


def format_performance_board(
    rows: tuple[PerformanceRow, ...],
    title: str = "BIST PERFORMANS LİDERLERİ",
) -> str:
    """Format a compact Telegram/social-style leaderboard."""
    if not rows:
        return f"{title}\n• Yeterli geçmiş veri yok."
    lines = [title]
    for row in rows:
        direction = "↑" if row.rank_change > 0 else "↓" if row.rank_change < 0 else "→"
        lines.append(
            f"{row.rank:>2}. {row.symbol:<8} %{row.period_return_pct:>7.2f} "
            f"{direction}{abs(row.rank_change):02d}"
        )
    return "\n".join(lines)
