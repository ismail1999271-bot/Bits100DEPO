"""Regime-based performance and overfitting diagnostics (research only)."""
from __future__ import annotations

from math import erf, log, sqrt

import numpy as np
import pandas as pd

EULER = 0.5772156649


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def _norm_ppf(p: float) -> float:
    # Acklam-free: bisection is plenty for diagnostics.
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def deflated_sharpe(sharpe_daily: float, n_obs: int, n_trials: int, skew: float = 0.0,
                    kurt: float = 3.0, trial_sharpe_var: float | None = None) -> float | None:
    """Probability the true Sharpe > 0 after deflating for ``n_trials`` tried strategies.

    Bailey & Lopez de Prado. ``sharpe_daily`` is per-period (not annualised).
    Returns None when inputs cannot support it (few observations / trials < 1).
    """
    if n_obs < 30 or n_trials < 1:
        return None
    var = trial_sharpe_var if trial_sharpe_var is not None else 1.0 / n_obs
    if n_trials == 1:
        sr0 = 0.0
    else:
        sr0 = sqrt(var) * ((1 - EULER) * _norm_ppf(1 - 1 / n_trials)
                           + EULER * _norm_ppf(1 - 1 / (n_trials * np.e)))
    denom = 1 - skew * sharpe_daily + (kurt - 1) / 4 * sharpe_daily ** 2
    if denom <= 0:
        return None
    z = (sharpe_daily - sr0) * sqrt(n_obs - 1) / sqrt(denom)
    return round(_norm_cdf(z), 6)


def regime_performance(trades: pd.DataFrame, regimes: pd.Series, min_trades: int = 20) -> pd.DataFrame:
    """Win/loss stats per regime. ``regimes`` is indexed by trade ``date`` (point-in-time labels).

    Regimes with fewer than ``min_trades`` filled trades are flagged INSUFFICIENT_SAMPLE.
    """
    filled = trades[trades["status"] == "FILLED"].copy()
    if filled.empty:
        return pd.DataFrame(columns=["regime", "trades", "win_rate", "mean_win", "mean_loss", "expectancy", "flag"])
    filled["regime"] = filled["date"].map(regimes)
    rows = []
    for name, g in filled.groupby(filled["regime"].fillna("UNKNOWN")):
        wins = g[g["net_return"] > 0]["net_return"]
        losses = g[g["net_return"] < 0]["net_return"]
        rows.append({
            "regime": name, "trades": len(g), "win_rate": round(float((g["net_return"] > 0).mean()), 4),
            "mean_win": round(float(wins.mean()), 6) if len(wins) else None,
            "mean_loss": round(float(losses.mean()), 6) if len(losses) else None,
            "expectancy": round(float(g["net_return"].mean()), 6),
            "flag": "OK" if len(g) >= min_trades else "INSUFFICIENT_SAMPLE"})
    return pd.DataFrame(rows)


def label_regimes(index_close: pd.Series, window: int = 50, vol_window: int = 20,
                  vol_quantile: float = 0.8) -> pd.Series:
    """Point-in-time regime: BULL/BEAR by close vs trailing SMA, plus HIGH_VOL by trailing vol rank."""
    sma = index_close.rolling(window, min_periods=window).mean()
    ret = index_close.pct_change()
    vol = ret.rolling(vol_window, min_periods=vol_window).std()
    thresh = vol.expanding(min_periods=vol_window * 2).quantile(vol_quantile)
    out = pd.Series(index=index_close.index, dtype=object)
    for ts in index_close.index:
        if pd.isna(sma[ts]):
            continue
        trend = "BULL" if index_close[ts] >= sma[ts] else "BEAR"
        hv = (not pd.isna(thresh[ts])) and vol[ts] > thresh[ts]
        out[ts] = trend + ("_HIGH_VOL" if hv else "")
    return out


def parameter_sensitivity(run_fn, grid: dict[str, list]) -> pd.DataFrame:
    """Evaluate ``run_fn(**params) -> float`` over a one-at-a-time grid.

    A strategy whose metric collapses away from the chosen value is brittle.
    Returns one row per (param, value) plus stability = min/max of the metric per param.
    """
    rows = []
    for name, values in grid.items():
        metrics = []
        for v in values:
            m = run_fn(**{name: v})
            metrics.append(m)
            rows.append({"param": name, "value": v, "metric": m})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    stab = frame.groupby("param")["metric"].agg(lambda s: None if s.isna().all() else float(s.min() - s.max()))
    frame["spread_min_minus_max"] = frame["param"].map(stab)
    return frame


def overfit_verdict(dsr: float | None, sensitivity: pd.DataFrame | None, regimes: pd.DataFrame | None,
                    train_metric: float | None, holdout_metric: float | None) -> dict:
    flags = []
    if dsr is None:
        flags.append("DSR_UNAVAILABLE")
    elif dsr < 0.95:
        flags.append("DSR_BELOW_95")
    if train_metric is not None and holdout_metric is not None and train_metric > 0 \
            and holdout_metric < 0.5 * train_metric:
        flags.append("HOLDOUT_DEGRADATION")
    if regimes is not None and not regimes.empty and (regimes["flag"] == "INSUFFICIENT_SAMPLE").any():
        flags.append("REGIME_SAMPLE_SMALL")
    if sensitivity is not None and not sensitivity.empty and "spread_min_minus_max" in sensitivity:
        worst = sensitivity["spread_min_minus_max"].dropna()
        if len(worst) and worst.min() < -abs(log(2)):  # metric swings by more than ~0.69 abs units
            flags.append("PARAMETER_SENSITIVE")
    return {"status": "SUSPECT" if flags else "NO_OVERFIT_FLAGS", "flags": flags,
            "note": "Tanı amaçlıdır; performans kanıtı değildir."}
