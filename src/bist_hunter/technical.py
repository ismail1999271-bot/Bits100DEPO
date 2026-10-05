"""Point-in-time technical, volume and momentum features on 1W/1D/4H/1H/15M/5M bars.

Every indicator at row *t* uses only bars <= *t* (rolling / ewm / shift(1)
windows). Intraday bars are resampled into higher timeframes and the last,
still-forming bucket is dropped unless it is complete, so no partial future
bar leaks into features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TIMEFRAMES = {"5M": "5min", "15M": "15min", "1H": "1h", "4H": "4h", "1D": "1D", "1W": "7D"}
FEATURE_COLUMNS = (
    "rsi_14", "macd", "macd_signal", "macd_hist", "ema_20", "sma_50", "atr_14", "adx_14", "vwap",
    "relative_volume", "volume_acceleration", "momentum_10", "volume_momentum", "volatility_20",
    "volatility_expansion", "breakout_20", "breakout_strength", "support_20", "resistance_20",
)


def resample_bars(frame: pd.DataFrame, timeframe: str, *, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Aggregate intraday OHLCV per symbol; drop buckets that end after ``as_of``."""
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"unsupported timeframe {timeframe}; use one of {sorted(TIMEFRAMES)}")
    rule = TIMEFRAMES[timeframe]
    out = []
    for symbol, g in frame.sort_values("timestamp").groupby("symbol"):
        g = g.set_index(pd.to_datetime(g["timestamp"]))
        if timeframe == "1W":
            # Calendar weeks starting Monday 00:00 (BIST trades Mon-Fri).
            week_start = g.index.normalize() - pd.to_timedelta(g.index.weekday, unit="D")
            agg = g.groupby(week_start).agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna(subset=["open", "close"])
            agg.index.name = None
        else:
            agg = g.resample(rule, label="left", closed="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna(subset=["open", "close"])
        if as_of is not None:
            end = agg.index + pd.Timedelta(rule)
            agg = agg[end <= as_of]
        agg["symbol"] = symbol
        agg["timeframe"] = timeframe
        out.append(agg.reset_index(names="timestamp"))
    if not out:
        return frame.iloc[0:0].assign(timeframe=timeframe)
    return pd.concat(out, ignore_index=True)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.where(loss != 0, 100.0).where(gain.notna())


def _true_range(g: pd.DataFrame) -> pd.Series:
    prev = g["close"].shift(1)
    return pd.concat([g["high"] - g["low"], (g["high"] - prev).abs(), (g["low"] - prev).abs()], axis=1).max(axis=1)


def _adx(g: pd.DataFrame, n: int = 14) -> pd.Series:
    up = g["high"].diff()
    down = -g["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = _true_range(g).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _session_key(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.date


def _per_symbol(g: pd.DataFrame, intraday: bool) -> pd.DataFrame:
    g = g.sort_values("timestamp").copy()
    close, volume = g["close"], g["volume"]
    g["rsi_14"] = _rsi(close)
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    g["macd"] = ema12 - ema26
    g["macd_signal"] = g["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    g["macd_hist"] = g["macd"] - g["macd_signal"]
    g["ema_20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    g["sma_50"] = close.rolling(50, min_periods=50).mean()
    g["atr_14"] = _true_range(g).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    g["adx_14"] = _adx(g)
    typical = (g["high"] + g["low"] + close) / 3
    if intraday:
        session = _session_key(g["timestamp"])
        pv = (typical * volume).groupby(session).cumsum()
        vv = volume.groupby(session).cumsum()
    else:
        pv = (typical * volume).rolling(20, min_periods=5).sum()
        vv = volume.rolling(20, min_periods=5).sum()
    g["vwap"] = pv / vv.replace(0, np.nan)
    prior_vol = volume.shift(1).rolling(20, min_periods=5).mean()
    g["relative_volume"] = volume / prior_vol.replace(0, np.nan)
    g["volume_acceleration"] = g["relative_volume"] - g["relative_volume"].shift(1)
    g["momentum_10"] = close / close.shift(10) - 1
    g["volume_momentum"] = volume.rolling(5, min_periods=5).mean() / prior_vol.replace(0, np.nan) - 1
    returns = close.pct_change()
    g["volatility_20"] = returns.rolling(20, min_periods=10).std()
    g["volatility_expansion"] = returns.rolling(5, min_periods=5).std() / g["volatility_20"].replace(0, np.nan)
    g["resistance_20"] = g["high"].shift(1).rolling(20, min_periods=5).max()
    g["support_20"] = g["low"].shift(1).rolling(20, min_periods=5).min()
    g["breakout_20"] = (close > g["resistance_20"]).astype(float).where(g["resistance_20"].notna())
    g["breakout_strength"] = (close / g["resistance_20"] - 1) / g["volatility_20"].replace(0, np.nan)
    return g


def add_technical_features(frame: pd.DataFrame, *, intraday: bool | None = None) -> pd.DataFrame:
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    if intraday is None:
        intraday = "timeframe" in data.columns and str(data["timeframe"].iloc[0]) not in ("1D", "1W")
    parts = [_per_symbol(g, intraday) for _, g in data.groupby("symbol", sort=False)]
    return pd.concat(parts).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def multi_timeframe_features(intraday: pd.DataFrame, as_of: pd.Timestamp,
                             timeframes: tuple[str, ...] = ("5M", "15M", "1H", "4H", "1D", "1W")) -> pd.DataFrame:
    """Latest complete-bar features per symbol and timeframe as of ``as_of``."""
    rows = []
    for tf in timeframes:
        bars = resample_bars(intraday, tf, as_of=as_of)
        if bars.empty:
            continue
        feats = add_technical_features(bars, intraday=tf not in ("1D", "1W"))
        latest = feats.groupby("symbol").tail(1)
        rows.append(latest.assign(timeframe=tf))
    if not rows:
        return pd.DataFrame(columns=["symbol", "timeframe", *FEATURE_COLUMNS])
    return pd.concat(rows, ignore_index=True)[["symbol", "timeframe", "timestamp", *FEATURE_COLUMNS]]


def technical_score(row: pd.Series | dict) -> float | None:
    """Transparent 0-100 score from available features; None if too little data."""
    get = row.get
    parts: list[tuple[float, float]] = []

    def add(weight, value):
        if value is not None and pd.notna(value) and np.isfinite(value):
            parts.append((weight, float(max(0.0, min(100.0, value)))))

    rsi = get("rsi_14")
    if rsi is not None and pd.notna(rsi):
        add(0.20, 100 - abs(rsi - 62) * 2.5)  # constructive momentum zone, penalize overbought
    add(0.20, None if get("macd_hist") is None or pd.isna(get("macd_hist")) or not get("atr_14")
        else 50 + 50 * np.tanh(get("macd_hist") / get("atr_14")))
    add(0.20, None if get("relative_volume") is None or pd.isna(get("relative_volume"))
        else 50 + 25 * np.log2(max(get("relative_volume"), 1e-6)))
    add(0.20, None if get("breakout_strength") is None or pd.isna(get("breakout_strength"))
        else 50 + 25 * np.tanh(get("breakout_strength")))
    add(0.20, None if get("momentum_10") is None or pd.isna(get("momentum_10"))
        else 50 + 500 * get("momentum_10"))
    if len(parts) < 3:
        return None
    weight = sum(w for w, _ in parts)
    return round(sum(w * v for w, v in parts) / weight, 2)


@dataclass(frozen=True, slots=True)
class FeatureUsefulness:
    feature: str
    observations: int
    mean_ic: float | None
    ic_t_stat: float | None
    positive_ic_share: float | None


def feature_usefulness(frame: pd.DataFrame, feature: str, *, horizon: int = 1,
                       min_names: int = 3) -> FeatureUsefulness:
    """Cross-sectional rank IC of a feature vs forward return, averaged over dates.

    Forward returns are used only as the *label*; features stay point-in-time.
    A useful feature has a consistently positive IC (t-stat > ~2).
    """
    data = frame.sort_values(["symbol", "timestamp"]).copy()
    data["_fwd"] = data.groupby("symbol")["close"].shift(-horizon) / data["close"] - 1
    ics = []
    for _, day in data.dropna(subset=[feature, "_fwd"]).groupby("timestamp"):
        if len(day) < min_names or day[feature].nunique() < 2 or day["_fwd"].nunique() < 2:
            continue
        ics.append(day[feature].rank().corr(day["_fwd"].rank()))
    if not ics:
        return FeatureUsefulness(feature, 0, None, None, None)
    arr = np.array(ics, dtype=float)
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    t = float(arr.mean() / (std / np.sqrt(len(arr)))) if std > 0 else None
    return FeatureUsefulness(feature, len(arr), round(float(arr.mean()), 6),
                             None if t is None else round(t, 4), round(float((arr > 0).mean()), 4))
