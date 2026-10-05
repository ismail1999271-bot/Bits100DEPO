"""Chart analysis: point-in-time support/resistance, candlestick/price patterns and a 4-panel PNG.

Everything is computed from bars <= the last bar shown. Pivot levels need ``confirm``
bars on the right to be confirmed, so the most recent ``confirm`` bars never create
a level (no look-ahead). Patterns are descriptive flags of what already happened on
the chart; they are not forecasts and carry no probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .technical import add_technical_features

PATTERN_LABELS = {
    "BULLISH_ENGULFING": "Yükseliş yutan mum",
    "BEARISH_ENGULFING": "Düşüş yutan mum",
    "HAMMER": "Çekiç",
    "DOJI": "Doji",
    "BREAKOUT_20": "20g direnç kırılımı",
    "BREAKDOWN_20": "20g destek kırılımı",
    "HIGHER_HIGHS_LOWS": "Yükselen tepe/dip",
    "LOWER_HIGHS_LOWS": "Alçalan tepe/dip",
    "LIMIT_UP": "Tavan kapanışı",
}


@dataclass(frozen=True, slots=True)
class Levels:
    supports: tuple[float, ...]
    resistances: tuple[float, ...]


def pivots(df: pd.DataFrame, left: int = 3, confirm: int = 3) -> tuple[list[int], list[int]]:
    """Indices of confirmed pivot highs/lows (needs ``confirm`` later bars)."""
    highs, lows = df["high"].to_numpy(), df["low"].to_numpy()
    ph, pl = [], []
    for i in range(left, len(df) - confirm):
        if highs[i] == highs[i - left:i + confirm + 1].max() and highs[i] > highs[i - 1]:
            ph.append(i)
        if lows[i] == lows[i - left:i + confirm + 1].min() and lows[i] < lows[i - 1]:
            pl.append(i)
    return ph, pl


def _cluster(values: list[float], tol: float) -> list[float]:
    out: list[list[float]] = []
    for v in sorted(values):
        if out and abs(v - np.mean(out[-1])) / np.mean(out[-1]) <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    # stronger levels (more touches) first
    out.sort(key=lambda g: -len(g))
    return [round(float(np.mean(g)), 4) for g in out]


def support_resistance(df: pd.DataFrame, *, tol: float = 0.015, max_levels: int = 3, left: int = 3,
                       confirm: int = 3) -> Levels:
    price = float(df["close"].iloc[-1])
    ph, pl = pivots(df, left, confirm)
    res = _cluster([float(df["high"].iloc[i]) for i in ph], tol)
    sup = _cluster([float(df["low"].iloc[i]) for i in pl], tol)
    return Levels(tuple(sorted([s for s in sup if s < price], reverse=True)[:max_levels]),
                  tuple(sorted([r for r in res if r > price])[:max_levels]))


def detect_patterns(df: pd.DataFrame, *, limit_pct: float = 0.10) -> list[str]:
    """Flags on the LAST bar (and short trend structure). Needs >= 25 bars, else []."""
    if len(df) < 25:
        return []
    d = df.reset_index(drop=True)
    o, h, lo_, c = (d[k].astype(float) for k in ("open", "high", "low", "close"))
    flags: list[str] = []
    body = abs(c.iloc[-1] - o.iloc[-1])
    rng = max(h.iloc[-1] - lo_.iloc[-1], 1e-12)
    lower = min(o.iloc[-1], c.iloc[-1]) - lo_.iloc[-1]
    upper = h.iloc[-1] - max(o.iloc[-1], c.iloc[-1])
    if body / rng < 0.1:
        flags.append("DOJI")
    if lower >= 2 * body and upper <= body and body / rng >= 0.1 and c.iloc[-2] > c.iloc[-1] * 0.0:
        down = c.iloc[-6:-1].is_monotonic_decreasing or c.iloc[-1] < c.iloc[-6]
        if down:
            flags.append("HAMMER")
    p_o, p_c = o.iloc[-2], c.iloc[-2]
    if c.iloc[-1] > o.iloc[-1] and p_c < p_o and c.iloc[-1] >= p_o and o.iloc[-1] <= p_c:
        flags.append("BULLISH_ENGULFING")
    if c.iloc[-1] < o.iloc[-1] and p_c > p_o and c.iloc[-1] <= p_o and o.iloc[-1] >= p_c:
        flags.append("BEARISH_ENGULFING")
    if c.iloc[-1] > h.iloc[-21:-1].max():
        flags.append("BREAKOUT_20")
    if c.iloc[-1] < lo_.iloc[-21:-1].min():
        flags.append("BREAKDOWN_20")
    ph, pl = pivots(d)
    if len(ph) >= 2 and len(pl) >= 2:
        if h.iloc[ph[-1]] > h.iloc[ph[-2]] and lo_.iloc[pl[-1]] > lo_.iloc[pl[-2]]:
            flags.append("HIGHER_HIGHS_LOWS")
        elif h.iloc[ph[-1]] < h.iloc[ph[-2]] and lo_.iloc[pl[-1]] < lo_.iloc[pl[-2]]:
            flags.append("LOWER_HIGHS_LOWS")
    if c.iloc[-2] > 0 and c.iloc[-1] / c.iloc[-2] - 1 >= limit_pct - 0.002:
        flags.append("LIMIT_UP")
    return flags


def _bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def render_chart(bars: pd.DataFrame, symbol: str, out_path: str | Path, *, last_n: int = 120,
                 limit_pct: float = 0.10, title_note: str = "") -> Path | None:
    """Render candles+EMA/SMA+Bollinger+S/R, volume, RSI and MACD. Returns None if < 30 bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = bars[bars["symbol"] == symbol].sort_values("timestamp").reset_index(drop=True)
    if len(g) < 30:
        return None
    tech = add_technical_features(g)
    lo, mid, hi = _bollinger(tech["close"])
    levels = support_resistance(g)
    patterns = detect_patterns(g, limit_pct=limit_pct)
    view = tech.tail(last_n).reset_index(drop=True)
    off = len(tech) - len(view)
    x = np.arange(len(view))
    fig, (ax, axv, axr, axm) = plt.subplots(4, 1, figsize=(11, 10), sharex=True,
                                            gridspec_kw={"height_ratios": [5, 1.4, 1.4, 1.4]})
    up = view["close"] >= view["open"]
    for xi, row, is_up in zip(x, view.itertuples(), up):
        color = "#16a34a" if is_up else "#dc2626"
        ax.vlines(xi, row.low, row.high, color=color, linewidth=0.8)
        ax.add_patch(plt.Rectangle((xi - 0.3, min(row.open, row.close)), 0.6,
                                   max(abs(row.close - row.open), 1e-9), color=color))
    ax.plot(x, view["ema_20"], color="#2563eb", linewidth=1.1, label="EMA20")
    ax.plot(x, view["sma_50"], color="#9333ea", linewidth=1.1, label="SMA50")
    ax.plot(x, lo.iloc[off:].to_numpy(), color="#94a3b8", linewidth=0.7, linestyle="--")
    ax.plot(x, hi.iloc[off:].to_numpy(), color="#94a3b8", linewidth=0.7, linestyle="--", label="Bollinger")
    for s in levels.supports:
        ax.axhline(s, color="#16a34a", linestyle=":", linewidth=1)
        ax.text(x[-1] + 0.5, s, f"D {s:.2f}", color="#16a34a", fontsize=8, va="center")
    for r in levels.resistances:
        ax.axhline(r, color="#dc2626", linestyle=":", linewidth=1)
        ax.text(x[-1] + 0.5, r, f"R {r:.2f}", color="#dc2626", fontsize=8, va="center")
    prev = tech["close"].shift(1)
    limit_days = (tech["close"] / prev - 1 >= limit_pct - 0.002).iloc[off:].to_numpy()
    for xi in x[limit_days]:
        ax.annotate("T", (xi, view["high"].iloc[xi]), xytext=(0, 6), textcoords="offset points",
                    ha="center", color="#b45309", fontsize=9, fontweight="bold")
    note = ", ".join(PATTERN_LABELS[p] for p in patterns) or "belirgin formasyon yok"
    ax.set_title(f"{symbol} · günlük · {note}{(' · ' + title_note) if title_note else ''}", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    axv.bar(x, view["volume"], color=np.where(up, "#86efac", "#fca5a5"))
    axv.set_ylabel("Hacim", fontsize=8)
    axr.plot(x, view["rsi_14"], color="#0f172a", linewidth=1)
    axr.axhline(70, color="#dc2626", linewidth=0.6)
    axr.axhline(30, color="#16a34a", linewidth=0.6)
    axr.set_ylim(0, 100)
    axr.set_ylabel("RSI", fontsize=8)
    axm.bar(x, view["macd_hist"], color=np.where(view["macd_hist"] >= 0, "#86efac", "#fca5a5"))
    axm.plot(x, view["macd"], color="#2563eb", linewidth=0.9)
    axm.plot(x, view["macd_signal"], color="#f59e0b", linewidth=0.9)
    axm.set_ylabel("MACD", fontsize=8)
    ax.set_xlim(-1, len(x) + 9)
    ticks = x[:: max(1, len(x) // 8)]
    axm.set_xticks(ticks)
    axm.set_xticklabels([pd.Timestamp(view["timestamp"].iloc[i]).strftime("%d.%m.%y") for i in ticks],
                        fontsize=8)
    fig.text(0.01, 0.005, "Araştırma grafiği; yatırım tavsiyesi değildir. Formasyonlar geçmişi tanımlar, tahmin değildir.",
             fontsize=7, color="#64748b")
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def chart_report(bars: pd.DataFrame, symbol: str) -> dict:
    """Machine-readable chart analysis for the dashboard snapshot."""
    g = bars[bars["symbol"] == symbol].sort_values("timestamp").reset_index(drop=True)
    if len(g) < 30:
        return {"status": "MISSING", "reason": "fewer than 30 daily bars"}
    lv = support_resistance(g)
    pats = detect_patterns(g)
    return {"status": "OK", "supports": list(lv.supports), "resistances": list(lv.resistances),
            "patterns": [{"code": p, "label": PATTERN_LABELS[p]} for p in pats],
            "last_close": float(g["close"].iloc[-1])}


def build_charts(bars: pd.DataFrame, symbols: list[str], out_dir: str | Path) -> dict[str, dict]:
    """PNG + analysis for each symbol. Returns {symbol: report + 'image'} for the dashboard snapshot."""
    out: dict[str, dict] = {}
    for symbol in symbols:
        report = chart_report(bars, symbol)
        if report["status"] == "OK":
            path = render_chart(bars, symbol, Path(out_dir) / f"{symbol}.png")
            report["image"] = None if path is None else str(path)
        out[symbol] = report
    return out
