"""Tavan-specific (BIST limit-up) risk features and hard blocks.

Every input is optional and comes from a real provider. ``None`` stays
``None``: nothing is estimated. A hard block (VBTS measure that forbids
trading, trading halt, circuit breaker, IPO lock-up) makes the symbol
``BLOCKED`` regardless of its score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

# VBTS (Volatility Based Measure System) measures that change tavan dynamics.
HARD_BLOCK_MEASURES = {"TEK_FIYAT", "ISLEM_YASAGI", "ISLEM_DURDURULDU"}
WARN_MEASURES = {"BRUT_TAKAS", "KREDILI_ISLEM_YASAGI", "EMIR_PAKETI", "ACIGA_SATIS_YASAGI", "NET_TAKAS"}
HALT_STATES = {"HALTED", "CIRCUIT_BREAKER", "SUSPENDED", "AUCTION_EXTENDED"}
HYPE_TERMS = ("tavan", "roket", "uçacak", "kesin", "garanti", "patlayacak", "ay'a", "1000%")


@dataclass(frozen=True, slots=True)
class TavanRiskInputs:
    symbol: str
    now: datetime
    vbts_measures: tuple[str, ...] | None = None  # None = provider did not say
    trading_state: str | None = None
    ipo_lockup_until: date | None = None
    free_float_ratio: float | None = None  # 0..1
    daily_closes: tuple[float, ...] = ()
    daily_highs: tuple[float, ...] = ()
    prev_close: float | None = None
    limit_pct: float = 0.10
    bid_queue_qty: float | None = None  # quantity queued at the limit price
    ask_qty_at_limit: float | None = None  # remaining sell quantity at the limit price
    avg_daily_volume: float | None = None
    social_mentions_z: float | None = None
    hype_posts: int | None = None
    volume_z: float | None = None
    upcoming_events: tuple[tuple[str, date], ...] = ()  # (type, date)


@dataclass(frozen=True, slots=True)
class TavanRisk:
    symbol: str
    status: str  # OK / BLOCKED
    block_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    limit_up_streak: int | None
    next_day_break_risk: str  # LOW / MEDIUM / HIGH / UNKNOWN
    lock_strength: float | None  # 0..100 (queue / remaining supply), None if unknown
    lock_probability_hint: float | None  # 0..1, transparent heuristic, not a forecast
    free_float_flag: str  # LOW_FLOAT / NORMAL / UNKNOWN
    manipulation_risk: float | None  # 0..1
    calendar_flags: tuple[str, ...] = field(default_factory=tuple)


def limit_up_streak(closes: tuple[float, ...], limit_pct: float, tolerance: float = 0.002) -> int:
    """Consecutive trailing closes at (almost) the upper limit versus the prior close."""
    streak = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i - 1] > 0 and closes[i] / closes[i - 1] - 1 >= limit_pct - tolerance:
            streak += 1
        else:
            break
    return streak


def lock_strength(bid_queue: float | None, ask_at_limit: float | None) -> float | None:
    """How firmly the limit is held: 100 when nobody sells at the limit, 50 when supply == queue."""
    if bid_queue is None or ask_at_limit is None or bid_queue < 0 or ask_at_limit < 0:
        return None
    total = bid_queue + ask_at_limit
    if total == 0:
        return None
    return round(100.0 * bid_queue / total, 2)


def manipulation_risk(mentions_z: float | None, hype_posts: int | None, volume_z: float | None,
                      free_float: float | None) -> float | None:
    """0..1 research RISK indicator (never a buy signal). Needs >= 2 inputs."""
    parts: list[tuple[float, float]] = []
    if mentions_z is not None:
        parts.append((0.3, min(1.0, max(0.0, mentions_z / 4))))
    if hype_posts is not None:
        parts.append((0.25, min(1.0, hype_posts / 10)))
    if volume_z is not None:
        parts.append((0.25, min(1.0, max(0.0, volume_z / 5))))
    if free_float is not None:
        parts.append((0.2, min(1.0, max(0.0, (0.25 - free_float) / 0.25))))
    if len(parts) < 2:
        return None
    return round(sum(w * v for w, v in parts) / sum(w for w, _ in parts), 4)


def hype_count(texts: list[str]) -> int:
    return sum(any(term in t.casefold() for term in HYPE_TERMS) for t in texts)


def assess_tavan_risk(inp: TavanRiskInputs, *, low_float_below: float = 0.15,
                      calendar_window_days: int = 3) -> TavanRisk:
    blocks: list[str] = []
    warnings: list[str] = []
    today = inp.now.date()

    if inp.trading_state is not None and inp.trading_state.upper() in HALT_STATES:
        blocks.append("TRADING_HALT_OR_CIRCUIT_BREAKER")
    if inp.ipo_lockup_until is not None and today <= inp.ipo_lockup_until:
        blocks.append("IPO_LOCKUP")
    if inp.vbts_measures is None:
        warnings.append("vbts_unknown")
    else:
        measures = {m.upper() for m in inp.vbts_measures}
        if measures & HARD_BLOCK_MEASURES:
            blocks.append("VBTS_" + "_".join(sorted(measures & HARD_BLOCK_MEASURES)))
        for m in sorted(measures & WARN_MEASURES):
            warnings.append(f"vbts_{m.lower()}")

    streak = limit_up_streak(inp.daily_closes, inp.limit_pct) if len(inp.daily_closes) >= 2 else None
    if streak is None:
        break_risk = "UNKNOWN"
    elif streak >= 4:
        break_risk = "HIGH"
    elif streak >= 2:
        break_risk = "MEDIUM"
    else:
        break_risk = "LOW"
    if break_risk in ("MEDIUM", "HIGH"):
        warnings.append(f"limit_up_streak_{streak}")

    strength = lock_strength(inp.bid_queue_qty, inp.ask_qty_at_limit)
    hint = None if strength is None else round(strength / 100, 4)
    if strength is not None and strength < 60:
        warnings.append("weak_limit_lock")

    if inp.free_float_ratio is None:
        float_flag = "UNKNOWN"
    elif inp.free_float_ratio < low_float_below:
        float_flag = "LOW_FLOAT"
        warnings.append("low_free_float")
    else:
        float_flag = "NORMAL"

    manip = manipulation_risk(inp.social_mentions_z, inp.hype_posts, inp.volume_z, inp.free_float_ratio)
    if manip is not None and manip >= 0.6:
        warnings.append("manipulation_risk_high")

    calendar: list[str] = []
    for kind, when in inp.upcoming_events:
        delta = (when - today).days
        if 0 <= delta <= calendar_window_days:
            calendar.append(f"{kind}@{when.isoformat()}")
    if calendar:
        warnings.append("corporate_event_soon")

    return TavanRisk(inp.symbol.upper(), "BLOCKED" if blocks else "OK", tuple(blocks), tuple(warnings),
                     streak, break_risk, strength, hint, float_flag, manip, tuple(calendar))


def closes_from_frame(frame: pd.DataFrame, symbol: str) -> tuple[float, ...]:
    g = frame[frame["symbol"] == symbol].sort_values("timestamp")
    return tuple(float(x) for x in g["close"].tolist())
