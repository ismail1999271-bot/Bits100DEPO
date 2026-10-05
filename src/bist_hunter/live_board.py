"""Live market board for ALL listed BIST symbols (display + research, no orders).

Pure functions. Quotes come from a real source (BorsaPy ``fast_info`` or a licensed
Level-1 feed). Nothing is invented: an unavailable quote is dropped and reported,
an unavailable indicator stays NaN. Every row carries the quote age; rows older than
``stale_after_s`` are flagged STALE. The board is a monitoring view, not a signal.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from .technical import add_technical_features

QUOTE_COLUMNS = ["symbol", "last", "prev_close", "open", "high", "low", "volume", "timestamp"]


def _num(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def quote_from_fast_info(symbol: str, info: Any, now: datetime) -> dict[str, Any] | None:
    """Map a BorsaPy ``fast_info`` object to a validated quote row; None when unusable."""
    def get(key: str):
        try:
            return info[key]
        except Exception:
            return getattr(info, key, None)

    last, prev = _num(get("last_price")), _num(get("previous_close"))
    if last is None or prev is None or last <= 0 or prev <= 0:
        return None
    o, hi, lo, vol = (_num(get(k)) for k in ("open", "day_high", "day_low", "volume"))
    if hi is not None and lo is not None and hi < lo:
        return None
    return {"symbol": symbol.upper(), "last": last, "prev_close": prev, "open": o, "high": hi, "low": lo,
            "volume": vol, "timestamp": now}


def fetch_quotes(symbols: list[str], ticker_factory: Callable[[str], Any], now: datetime,
                 max_workers: int = 16) -> tuple[pd.DataFrame, dict[str, str]]:
    skipped: dict[str, str] = {}

    def one(symbol: str):
        try:
            row = quote_from_fast_info(symbol, ticker_factory(symbol).fast_info, now)
        except Exception as exc:
            skipped[symbol] = str(exc)[:80]
            return None
        if row is None:
            skipped[symbol] = "unusable quote"
        return row

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        rows = [r for r in pool.map(one, symbols) if r is not None]
    return pd.DataFrame(rows, columns=QUOTE_COLUMNS), skipped


def _indicators(bars: pd.DataFrame | None) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()
    tech = add_technical_features(bars)
    last = tech.sort_values("timestamp").groupby("symbol").tail(1).set_index("symbol")
    g = bars.sort_values("timestamp").groupby("symbol")
    adv = g.apply(lambda d: (d["close"] * d["volume"]).tail(20).mean(), include_groups=False)
    avg_vol = g["volume"].apply(lambda s: s.tail(20).mean())
    out = pd.DataFrame(index=last.index)
    out["RSI"] = last["rsi_14"]
    out["ATR %"] = last["atr_14"] / last["close"] * 100
    out["Trend"] = np.where(last["ema_20"].isna() | last["sma_50"].isna(), None,
                            np.where((last["close"] > last["ema_20"]) & (last["ema_20"] > last["sma_50"]), "YUKARI",
                                     np.where((last["close"] < last["ema_20"]) & (last["ema_20"] < last["sma_50"]),
                                              "AŞAĞI", "YATAY")))
    out["Direnç 20g"] = last["resistance_20"]
    out["Destek 20g"] = last["support_20"]
    out["Ort. Hacim 20g"] = avg_vol
    out["Ort. Değer 20g (TL)"] = adv
    return out


def build_board(quotes: pd.DataFrame, bars: pd.DataFrame | None = None, *, now: datetime,
                limit_pct: float = 0.10, stale_after_s: float = 1200.0,
                scores: dict[str, float] | None = None) -> pd.DataFrame:
    """One row per symbol with change, range position, tavan/taban distance and (if bars) indicators."""
    if quotes is None or quotes.empty:
        return pd.DataFrame()
    q = quotes.copy()
    q["symbol"] = q["symbol"].astype(str).str.upper()
    q = q.drop_duplicates("symbol", keep="last").set_index("symbol")
    b = pd.DataFrame(index=q.index)
    b["Son"] = q["last"]
    b["Önc. Kapanış"] = q["prev_close"]
    b["Değişim %"] = (q["last"] / q["prev_close"] - 1) * 100
    b["Açılış"] = q["open"]
    b["Gap %"] = (q["open"] / q["prev_close"] - 1) * 100
    b["Yüksek"] = q["high"]
    b["Düşük"] = q["low"]
    span = (q["high"] - q["low"]).replace(0, np.nan)
    b["Gün İçi Konum"] = ((q["last"] - q["low"]) / span).clip(0, 1)  # 0 = günün dibi, 1 = günün tepesi
    b["Hacim"] = q["volume"]
    b["Değer (TL)"] = q["last"] * q["volume"]
    tavan_px = (q["prev_close"] * (1 + limit_pct)).round(2)
    taban_px = (q["prev_close"] * (1 - limit_pct)).round(2)
    b["Tavan Fiyatı"] = tavan_px
    b["Taban Fiyatı"] = taban_px
    b["Tavana Uzaklık %"] = (tavan_px / q["last"] - 1) * 100
    b["Tavan"] = q["last"] >= tavan_px * 0.998
    b["Taban"] = q["last"] <= taban_px * 1.002
    ind = _indicators(bars)
    if not ind.empty:
        b = b.join(ind)
        b["Rel. Hacim"] = b["Hacim"] / b["Ort. Hacim 20g"].replace(0, np.nan)
        b["Direnç Kırılımı"] = b["Son"] > b["Direnç 20g"]
    if scores:
        b["Quant Score"] = pd.Series(scores)
    age = (pd.Timestamp(now) - pd.to_datetime(q["timestamp"], utc=True)).dt.total_seconds()
    b["Veri Yaşı (sn)"] = age
    b["Durum"] = np.where(age > stale_after_s, "STALE", "CANLI")
    b.index.name = "Sembol"
    return b.reset_index().sort_values("Değer (TL)", ascending=False, na_position="last").reset_index(drop=True)


def summarize(board: pd.DataFrame) -> dict[str, Any]:
    if board is None or board.empty:
        return {"status": "MISSING"}
    ch = board["Değişim %"]
    up, down = int((ch > 0).sum()), int((ch < 0).sum())
    total = int(ch.notna().sum())
    return {
        "status": "OK", "count": total, "yükselen": up, "düşen": down, "değişmeyen": total - up - down,
        "tavan": int(board["Tavan"].sum()), "taban": int(board["Taban"].sum()),
        "genişlik %": round(100 * up / total, 1) if total else None,
        "medyan değişim %": round(float(ch.median()), 2),
        "toplam değer (TL)": float(board["Değer (TL)"].sum(skipna=True)),
        "stale": int((board["Durum"] == "STALE").sum()),
    }


def movers(board: pd.DataFrame, column: str, n: int = 15, ascending: bool = False) -> pd.DataFrame:
    if board is None or board.empty or column not in board:
        return pd.DataFrame()
    return board.dropna(subset=[column]).sort_values(column, ascending=ascending).head(n)


def change_histogram(board: pd.DataFrame, bins=(-10.5, -7, -5, -3, -1, 0, 1, 3, 5, 7, 10.5)) -> pd.DataFrame:
    """Counts per change bucket, in ascending order (``Sıra`` keeps the order for charting)."""
    labels = [f"{lo:g} … {hi:g}" for lo, hi in zip(bins[:-1], bins[1:])]
    labels[0], labels[-1] = f"≤ {bins[1]:g}", f"≥ {bins[-2]:g}"
    if board is None or board.empty:
        return pd.DataFrame({"Aralık": labels, "Adet": 0, "Sıra": range(len(labels))})
    cut = pd.cut(board["Değişim %"], bins=list(bins), include_lowest=True, labels=labels)
    counts = cut.value_counts().reindex(labels).fillna(0).astype(int)
    return pd.DataFrame({"Aralık": labels, "Adet": counts.to_numpy(), "Sıra": range(len(labels))})


def alerts(board: pd.DataFrame, *, near_tavan_pct: float = 1.0, relvol: float = 3.0) -> list[dict[str, str]]:
    """Event list for the current board (UI de-duplicates between refreshes)."""
    out: list[dict[str, str]] = []
    if board is None or board.empty:
        return out
    live = board[board["Durum"] == "CANLI"]
    for r in live.itertuples(index=False):
        d = r._asdict() if hasattr(r, "_asdict") else {}
        sym = d.get("Sembol")
        if d.get("Tavan"):
            out.append({"id": f"{sym}:TAVAN", "text": f"{sym} TAVANDA ({d['Son']:.2f})", "kind": "TAVAN"})
        elif d.get("Tavana Uzaklık %") is not None and 0 < d["Tavana Uzaklık %"] <= near_tavan_pct:
            out.append({"id": f"{sym}:YAKIN", "text": f"{sym} tavana %{d['Tavana Uzaklık %']:.2f} uzakta",
                        "kind": "TAVAN_YAKIN"})
        if d.get("Taban"):
            out.append({"id": f"{sym}:TABAN", "text": f"{sym} TABANDA ({d['Son']:.2f})", "kind": "TABAN"})
        rv = d.get("Rel. Hacim")
        if rv is not None and rv == rv and rv >= relvol:
            out.append({"id": f"{sym}:HACIM", "text": f"{sym} hacim {rv:.1f}x", "kind": "HACIM"})
        if d.get("Direnç Kırılımı") is True:
            out.append({"id": f"{sym}:KIRILIM", "text": f"{sym} 20g direnç kırılımı", "kind": "KIRILIM"})
    return out
