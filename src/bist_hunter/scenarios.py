"""Bull / neutral / bear scenario builder from daily + weekly technical state.

Each scenario lists the conditions that confirm it and the price level that
invalidates it. Nothing here is a forecast or a guarantee; levels that cannot
be computed (missing inputs) are simply omitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping

DISCLAIMER = "Senaryolar garanti değildir; doğrulama ve geçersizlik koşulları araştırma referansıdır."


def _g(row: Mapping[str, Any] | None, key: str) -> float | None:
    if row is None:
        return None
    v = row.get(key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if isfinite(v) else None


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str  # BULL / NEUTRAL / BEAR
    summary: str
    confirms_if: tuple[str, ...]
    invalidated_if: tuple[str, ...]
    reference_level: float | None = None  # next reference level in the scenario direction


@dataclass(frozen=True, slots=True)
class ScenarioSet:
    symbol: str
    status: str  # OK / BLOCKED_MISSING_DATA
    bias: str  # BULLISH / NEUTRAL / BEARISH / UNKNOWN
    trend_points: int | None
    alignment: str  # ALIGNED_UP / ALIGNED_DOWN / MIXED / UNKNOWN (daily vs weekly)
    scenarios: tuple[Scenario, ...] = field(default_factory=tuple)
    disclaimer: str = DISCLAIMER


def _trend_points(row: Mapping[str, Any] | None) -> tuple[int, int] | None:
    """(points, checks) from close vs EMA20/SMA50, MACD hist, RSI; None if < 3 checks possible."""
    if row is None:
        return None
    close = _g(row, "close")
    if close is None:
        return None
    pts = checks = 0
    for ma in ("ema_20", "sma_50"):
        v = _g(row, ma)
        if v is not None:
            checks += 1
            pts += 1 if close > v else -1
    hist = _g(row, "macd_hist")
    if hist is not None:
        checks += 1
        pts += 1 if hist > 0 else -1
    rsi = _g(row, "rsi_14")
    if rsi is not None:
        checks += 1
        pts += 1 if rsi > 55 else -1 if rsi < 45 else 0
    return (pts, checks) if checks >= 3 else None


def _label(points: tuple[int, int] | None) -> str:
    if points is None:
        return "UNKNOWN"
    pts, checks = points
    ratio = pts / checks
    return "BULLISH" if ratio >= 0.5 else "BEARISH" if ratio <= -0.5 else "NEUTRAL"


def build_scenarios(symbol: str, daily: Mapping[str, Any] | None, weekly: Mapping[str, Any] | None = None,
                    *, stop_atr: float = 1.5) -> ScenarioSet:
    d = _trend_points(daily)
    if d is None:
        return ScenarioSet(symbol.upper(), "BLOCKED_MISSING_DATA", "UNKNOWN", None, "UNKNOWN")
    bias = _label(d)
    w = _label(_trend_points(weekly))
    alignment = ("ALIGNED_UP" if bias == "BULLISH" and w == "BULLISH"
                 else "ALIGNED_DOWN" if bias == "BEARISH" and w == "BEARISH"
                 else "UNKNOWN" if w == "UNKNOWN" else "MIXED")
    close = _g(daily, "close")
    atr = _g(daily, "atr_14")
    support, resistance = _g(daily, "support_20"), _g(daily, "resistance_20")
    rel_vol = _g(daily, "relative_volume")
    w_support, w_resistance = _g(weekly, "support_20"), _g(weekly, "resistance_20")

    bull_confirm, bull_invalid = [], []
    if resistance is not None:
        bull_confirm.append(f"Günlük kapanış {resistance:.2f} direncinin üzerinde")
    if rel_vol is not None:
        bull_confirm.append(f"Göreceli hacim 1.5x üzerinde (şu an {rel_vol:.2f}x)")
    if _g(daily, "macd_hist") is not None:
        bull_confirm.append("MACD histogramı pozitif kalıyor")
    stop = None
    if atr is not None and close is not None:
        stop = close - stop_atr * atr
    if support is not None:
        stop = support if stop is None else max(stop, support)
    if stop is not None:
        bull_invalid.append(f"Günlük kapanış {stop:.2f} altına iniyor")
    if w_support is not None:
        bull_invalid.append(f"Haftalık destek {w_support:.2f} kırılıyor")

    bear_confirm, bear_invalid = [], []
    if support is not None:
        bear_confirm.append(f"Günlük kapanış {support:.2f} desteğinin altında")
    if _g(daily, "macd_hist") is not None:
        bear_confirm.append("MACD histogramı negatif kalıyor")
    if resistance is not None:
        bear_invalid.append(f"Günlük kapanış {resistance:.2f} direncinin üzerine çıkıyor")
    if w_resistance is not None:
        bear_invalid.append(f"Haftalık direnç {w_resistance:.2f} aşılıyor")

    neutral_confirm, neutral_invalid = [], []
    if support is not None and resistance is not None:
        neutral_confirm.append(f"Fiyat {support:.2f}–{resistance:.2f} bandında kalıyor")
        neutral_invalid.append(f"Bandın dışında günlük kapanış ({support:.2f} altı / {resistance:.2f} üstü)")
    adx = _g(daily, "adx_14")
    if adx is not None:
        neutral_confirm.append(f"ADX düşük kalıyor (şu an {adx:.1f}; 20 altı zayıf trend)")

    scenarios = (
        Scenario("BULL", "Yükseliş senaryosu", tuple(bull_confirm), tuple(bull_invalid), resistance),
        Scenario("NEUTRAL", "Nötr / yatay senaryo", tuple(neutral_confirm), tuple(neutral_invalid), None),
        Scenario("BEAR", "Düşüş senaryosu", tuple(bear_confirm), tuple(bear_invalid), support),
    )
    return ScenarioSet(symbol.upper(), "OK", bias, d[0], alignment, scenarios)


def format_scenarios(result: ScenarioSet) -> str:
    if result.status != "OK":
        return f"{result.symbol}: {result.status}"
    lines = [f"{result.symbol} — eğilim {result.bias} (günlük/haftalık: {result.alignment})"]
    for s in result.scenarios:
        lines.append(f"• {s.summary}")
        lines += [f"    ✔ {c}" for c in s.confirms_if]
        lines += [f"    ✖ {c}" for c in s.invalidated_if]
    lines.append(result.disclaimer)
    return "\n".join(lines)
