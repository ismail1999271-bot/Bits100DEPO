"""Ready-made candidate screens. They only NARROW the candidate pool for research;
they never produce a signal or a score. Each screen needs the columns it names; a
symbol whose required value is NaN simply does not match (nothing is imputed)."""
from __future__ import annotations

from typing import Callable

import pandas as pd

Screen = Callable[[pd.DataFrame], pd.Series]


def _last(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("timestamp").groupby("symbol").tail(1).set_index("symbol")


def _screen(fn: Screen, label: str, needs: tuple[str, ...]):
    fn.label = label  # type: ignore[attr-defined]
    fn.needs = needs  # type: ignore[attr-defined]
    return fn


SCREENS: dict[str, Screen] = {
    "volume_spike": _screen(lambda d: d["relative_volume"] >= 2.0, "Hacim patlaması (≥2x)", ("relative_volume",)),
    "breakout_20": _screen(lambda d: d["breakout_20"] == 1.0, "20 günlük direnç kırılımı", ("breakout_20",)),
    "trend_up": _screen(lambda d: (d["close"] > d["ema_20"]) & (d["ema_20"] > d["sma_50"]),
                        "Yükselen trend (kapanış>EMA20>SMA50)", ("ema_20", "sma_50")),
    "oversold_bounce": _screen(lambda d: (d["rsi_14"] < 30) & (d["macd_hist"] > 0),
                               "Aşırı satım + MACD dönüşü", ("rsi_14", "macd_hist")),
    "momentum": _screen(lambda d: (d["momentum_10"] > 0.05) & (d["adx_14"] > 25),
                        "Güçlü momentum (10g>%5, ADX>25)", ("momentum_10", "adx_14")),
    "volatility_expansion": _screen(lambda d: d["volatility_expansion"] >= 1.5,
                                    "Oynaklık genişlemesi", ("volatility_expansion",)),
    "sma_cross": _screen(lambda d: (d["close"] > d["sma_50"]) & (d["prev_close"] <= d["prev_sma_50"]),
                         "SMA50 yukarı kesişim", ("sma_50", "prev_close", "prev_sma_50")),
}


def run_screens(tech: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    """Boolean table symbol x screen over the LAST bar of ``tech`` (output of add_technical_features)."""
    data = tech.sort_values(["symbol", "timestamp"]).copy()
    g = data.groupby("symbol")
    data["prev_close"] = g["close"].shift(1)
    data["prev_sma_50"] = g["sma_50"].shift(1)
    last = _last(data)
    out = {}
    for name in names or list(SCREENS):
        fn = SCREENS[name]
        missing = [c for c in fn.needs if c not in last.columns]
        if missing:
            raise KeyError(f"screen {name} needs columns {missing}")
        mask = fn(last)
        out[name] = mask.fillna(False).astype(bool)
    return pd.DataFrame(out)


def matches(tech: pd.DataFrame, names: list[str] | None = None, *, min_hits: int = 1) -> pd.DataFrame:
    table = run_screens(tech, names)
    hits = table.sum(axis=1)
    result = table[hits >= min_hits].copy()
    result["hits"] = hits[hits >= min_hits]
    return result.sort_values("hits", ascending=False)
