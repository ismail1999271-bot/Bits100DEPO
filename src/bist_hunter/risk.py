"""Per-signal risk references: entry / stop / target, R/R, spread, slippage, liquidity.

These are *reference levels for research*, not orders. Missing inputs make
the corresponding field ``None`` and add a reason; nothing is estimated from
thin air.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

BIST_DAILY_LIMIT_PCT = 0.10  # nominal price band used for the tavan (limit-up) target cap


@dataclass(frozen=True, slots=True)
class RiskInputs:
    last_price: float | None
    atr: float | None = None
    support: float | None = None
    reference_close: float | None = None  # previous close -> limit-up cap
    spread_bps: float | None = None
    avg_daily_value_try: float | None = None
    top_of_book_depth_try: float | None = None
    position_value_try: float = 100_000.0


@dataclass(frozen=True, slots=True)
class RiskReport:
    status: str
    entry_reference: float | None
    stop_reference: float | None
    target_reference: float | None
    risk_reward: float | None
    spread_risk: str
    slippage_bps_estimate: float | None
    slippage_risk: str
    liquidity_risk: str
    overall: str
    reasons: tuple[str, ...]


def _ok(value) -> bool:
    return value is not None and isfinite(float(value)) and float(value) > 0


def _bucket(value: float | None, low: float, high: float) -> str:
    if value is None:
        return "UNKNOWN"
    return "LOW" if value <= low else "MEDIUM" if value <= high else "HIGH"


def assess_risk(
    inputs: RiskInputs,
    *,
    stop_atr: float = 1.5,
    target_atr: float = 3.0,
    max_participation: float = 0.02,
) -> RiskReport:
    reasons: list[str] = []
    if not _ok(inputs.last_price):
        return RiskReport("BLOCKED_MISSING_MARKET_DATA", None, None, None, None, "UNKNOWN", None,
                          "UNKNOWN", "UNKNOWN", "BLOCKED", ("no last price",))
    entry = float(inputs.last_price)

    stop = None
    if _ok(inputs.atr):
        stop = entry - stop_atr * float(inputs.atr)
    if _ok(inputs.support) and float(inputs.support) < entry:
        stop = float(inputs.support) if stop is None else max(stop, float(inputs.support) * 0.995)
    if stop is None or stop <= 0:
        stop = None
        reasons.append("no_stop_reference")

    target = entry + target_atr * float(inputs.atr) if _ok(inputs.atr) else None
    if _ok(inputs.reference_close):
        cap = float(inputs.reference_close) * (1 + BIST_DAILY_LIMIT_PCT)
        target = cap if target is None else min(target, cap)
    if target is None or target <= entry:
        reasons.append("no_upside_to_target" if target is not None else "no_target_reference")
        target = None if target is None else target

    rr = None
    if stop is not None and target is not None and entry > stop:
        rr = round((target - entry) / (entry - stop), 4)
        if rr < 1.0:
            reasons.append("risk_reward_below_1")

    spread_risk = _bucket(inputs.spread_bps, 15, 40)
    slippage = None
    if _ok(inputs.top_of_book_depth_try):
        # Consuming beyond visible top-of-book costs roughly half a spread per multiple of depth.
        multiple = inputs.position_value_try / float(inputs.top_of_book_depth_try)
        half_spread = (inputs.spread_bps or 0.0) / 2
        slippage = round(half_spread * (1 + multiple), 4)
    elif inputs.spread_bps is not None:
        slippage = round(inputs.spread_bps / 2, 4)
    slippage_risk = _bucket(slippage, 10, 30)

    liquidity_risk = "UNKNOWN"
    if _ok(inputs.avg_daily_value_try):
        participation = inputs.position_value_try / float(inputs.avg_daily_value_try)
        liquidity_risk = "LOW" if participation <= max_participation / 4 else \
            "MEDIUM" if participation <= max_participation else "HIGH"
        if liquidity_risk == "HIGH":
            reasons.append("position_exceeds_participation_limit")
    for name, level in (("spread", spread_risk), ("slippage", slippage_risk), ("liquidity", liquidity_risk)):
        if level == "UNKNOWN":
            reasons.append(f"{name}_unknown")

    levels = [spread_risk, slippage_risk, liquidity_risk]
    if "HIGH" in levels or (rr is not None and rr < 1.0):
        overall = "HIGH"
    elif "UNKNOWN" in levels or rr is None:
        overall = "UNKNOWN"
    elif "MEDIUM" in levels:
        overall = "MEDIUM"
    else:
        overall = "LOW"
    return RiskReport("OK", round(entry, 4), None if stop is None else round(stop, 4),
                      None if target is None else round(target, 4), rr, spread_risk, slippage,
                      slippage_risk, liquidity_risk, overall, tuple(reasons))
